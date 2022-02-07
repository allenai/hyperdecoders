import glob
import os
from dataclasses import asdict
from logging import getLogger
from third_party.utils import assert_all_frozen, freeze_embeds, freeze_params, save_json
from transformers.models.t5.modeling_t5 import T5LayerNorm

from data import TASK_MAPPING

logger = getLogger(__name__)


def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def handle_metrics(split, metrics, output_dir):  # , gcs_bucket=None):
    """
    Prints and saves metrics or a general dictionary of results.

    Args:
        split: one of train, val, test, or training arguments.
        metrics: metrics dict
        output_dir: where to save the metrics, if gcs_bucket is given
        we save the results also on the given bucket.
    """
    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json_file(metrics, f"{split}_results.json", output_dir)


def save_json_file(json_dict, outfile_name, output_dir):
    """
    Saves the given dictionary as a json file to output_dir and also
    the given bucket if given.
    """
    save_json(json_dict, os.path.join(output_dir, outfile_name))


def get_training_args(arguments_list):
    """
    Concatenate all training arguments except evaluation strategy which
    is not Json serializable.
    Args:
        arguments_list: list of dataclasses.
    Return:
        arguments: concatenated arguments.
    """
    all_arguments = {}
    for arguments in arguments_list:
        all_arguments.update(asdict(arguments))
    all_arguments.pop("evaluation_strategy")
    return all_arguments


def get_last_checkpoint_path(output_dir):
    """
    Finds the path for the last checkpoint saved in the output_dir
    Args:
        output_dir:  output_dir
    Returns:
        path to the last checkpoint saved in the output dir.
    """
    paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if len(paths) == 0:
        return output_dir
    else:
        checkpoints = [int(checkpoint.split("-")[-1]) for checkpoint in paths]
        max_checkpoint = max(checkpoints)
        return os.path.join(output_dir, "checkpoint-" + str(max_checkpoint))


def use_task_specific_params(model, task):
    """Update config with task specific params during evaluation."""
    task_dataset = TASK_MAPPING[task]
    task_specific_config = task_dataset.task_specific_config
    if task_specific_config is not None:
        logger.info(f"using task specific params for {task}: {task_specific_config}")
        model.config.update(task_specific_config)


def reset_config(model, config):
    """Resets the config file to the one provided."""
    model.config = model.config.from_dict(config)
    logger.info(f"config is reset to the initial values.")


def freeze_model(model):
    """Freezes the model weights."""
    freeze_params(model)


def unfreeze_adapter_params_encoder(model):
    for name, param in model.named_parameters():
        if (
            "adapter" in name or "mlp" in name or "param_gen" in name
        ) and "encoder" in name:
            param.requires_grad = True


def unfreeze_adapter_params_decoder(model):
    for name, param in model.named_parameters():
        if (
            "adapter" in name or "mlp" in name or "param_gen" in name
        ) and "decoder" in name:
            param.requires_grad = True


def unfreeze_encoder(model):
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = True


def unfreeze_decoder(model):
    for name, param in model.named_parameters():
        if "decoder" in name:
            param.requires_grad = True
