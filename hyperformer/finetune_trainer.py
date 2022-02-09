import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path
import dataclasses

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy

from modeling.adapter_t5 import (
    T5WithAdapterConfig,
    T5ForConditionalGenerationWithAdapter,
)
from modeling.adapter_bart import (
    BartWithAdapterConfig,
    BartForConditionalGenerationWithAdapter,
)
from third_party.trainers import T5Trainer
from data import AutoTask
from third_party.utils import TaskCollator, check_output_dir, MrqaTaskCollator
from metrics import build_compute_metrics_fn
from training_args import (
    Seq2SeqTrainingArguments,
    ModelArguments,
    DataTrainingArguments,
    AdapterTrainingArguments,
)
from utils import (
    get_last_checkpoint_path,
    freeze_model,
    unfreeze_adapter_params_encoder,
    unfreeze_adapter_params_decoder,
    unfreeze_encoder,
    unfreeze_decoder,
)

logger = logging.getLogger(__name__)


def remove_rank_info_from_argv(args):
    extra_parameters = {}
    if args[1].startswith("--local_rank"):
        extra_parameters.update({"local_rank": int(args[1].split("=")[-1])})
        del args[1]
    return extra_parameters


def main():
    # See all possible arguments in src/transformers/training_args.py or by passing
    # the --help flag to this script. We now keep distinct sets of args, for a cleaner
    # separation of concerns.
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            AdapterTrainingArguments,
        )
    )

    # For running on multiple gpus with torch.distributed.launch, it adds a local_rank paramter, to allow the parser
    # still use the config file, we add the local_rank to the config file.
    if (
        len(sys.argv) > 2
        and sys.argv[1].startswith("--local_rank")
        and (sys.argv[2].endswith(".json"))
    ):
        rank_info = remove_rank_info_from_argv(sys.argv)
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        args_dict.update(rank_info)
        model_args, data_args, training_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        logger.warning("config path: %s", sys.argv[1])
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()
    check_output_dir(training_args)

    # Setup logging
    # logfile output folders must exist before telling the logger to output there
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        filename=os.path.join(training_args.output_dir, "log.txt"),
        filemode="w+",
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if "bart" in (
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path
    ):
        model_class = BartForConditionalGenerationWithAdapter
        config_class = BartWithAdapterConfig
    else:
        from transformers import T5Config, T5ForConditionalGeneration

        model_class = T5ForConditionalGenerationWithAdapter
        config_class = T5WithAdapterConfig

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = config_class.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.update(dataclasses.asdict(adapter_args))
    all_tasks = list(set(data_args.tasks + data_args.eval_tasks))
    # mrqa is a single 'task' with many sub-tasks
    if "mrqa" or "mrqa_reg" in data_args.tasks + data_args.eval_tasks:
        all_tasks += [
            "HotpotQA",
            "NaturalQuestionsShort",
            "NewsQA",
            "SearchQA",
            "SQuAD",
            "TriviaQA-web",
        ]
    config.update({"tasks": all_tasks})

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if model_args.not_load_t5_checkpoint:
        model = model_class(config=config)
    else:
        last_checkpoint_path = training_args.output_dir
        model_path = (
            model_args.model_name_or_path
            if (
                (
                    training_args.optimize_from_scratch
                    and not training_args.optimize_from_scratch_with_loading_model
                )
                or not os.path.exists(
                    os.path.join(last_checkpoint_path, "pytorch_model.bin")
                )
            )
            else last_checkpoint_path
        )
        logger.warning("model path loaded from : %s", model_path)
        model = model_class.from_pretrained(
            model_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if model_args.freeze_model:
        freeze_model(model)
    if model_args.unfreeze_encoder_adapters:
        unfreeze_adapter_params_encoder(model)
    if model_args.unfreeze_decoder_adapters:
        unfreeze_adapter_params_decoder(model)
    if model_args.unfreeze_encoder:
        unfreeze_encoder(model)
    if model_args.unfreeze_decoder:
        unfreeze_decoder(model)

    if training_args.print_num_parameters:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
    # Gets the training/test/validation datasets.
    dataset_class = AutoTask
    if training_args.do_train:
        train_datasets = [
            dataset_class.get(task, seed=data_args.data_seed).get_dataset(
                split="train",
                n_obs=data_args.n_train,
                add_prefix=True,
                split_validation_test=training_args.split_validation_test,
            )
            for task in data_args.tasks
        ]
        if "mrqa" in data_args.tasks and data_args.filter_nulls:
            mrqa = train_datasets[data_args.tasks.index("mrqa")]
            mrqa.toggle_null_filter()
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = (
        {
            task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
                split="validation",
                n_obs=data_args.n_val,
                add_prefix=True,
                split_validation_test=training_args.split_validation_test,
            )
            for task in data_args.eval_tasks
        }
        if training_args.do_eval
        or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        {
            task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
                split="test",
                n_obs=data_args.n_test,
                add_prefix=True,
                split_validation_test=training_args.split_validation_test,
            )
            for task in data_args.eval_tasks
        }
        if training_args.do_test
        else None
    )

    # Defines the metrics for evaluation.
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer)
        if training_args.predict_with_generate
        else None
    )

    collator_class = TaskCollator
    compute_gen_probs = False
    if "mrqa" in eval_datasets:
        collator_class = MrqaTaskCollator
        compute_gen_probs = True
    elif "mrqa_reg" in eval_datasets:
        compute_gen_probs = True

    # Defines the trainer.
    trainer = T5Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=collator_class(
            tokenizer,
            data_args,
            tpu_num_cores=training_args.tpu_num_cores,
        ),
        tokenizer=tokenizer,
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        compute_gen_probs=compute_gen_probs,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
    )

    # Trains the model.
    if training_args.do_train:
        if trainer.is_world_process_zero():
            last_checkpoint_path = training_args.output_dir
            model_path = (
                model_args.model_name_or_path
                if (
                    training_args.optimize_from_scratch
                    or not os.path.exists(
                        os.path.join(last_checkpoint_path, "pytorch_model.bin")
                    )
                )
                else last_checkpoint_path
            )
        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for move to complete
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        trainer.train(
            # get_last_checkpoint_path(training_args.output_dir) \
            model_path=model_path
            if (
                os.path.exists(training_args.output_dir)
                and not training_args.optimize_from_scratch
            )
            else None,
        )
        if training_args.compute_time:
            torch.cuda.synchronize()  # wait for all_reduce to complete
            end.record()
            total_time = {"total_time": start.elapsed_time(end)}
            print("###### total_time ", total_time)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_test:
        # to avoid overwriting
        config.mean_task_embeddings = True
        trainer.answer_output_file = "predicted_answers_test.json"
        trainer.evaluate(test_dataset)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
