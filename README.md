# Hyperdecoders

Code for [the Hyperdecoders paper](https://arxiv.org/abs/2203.08304). This is built off the [hyperformer codebase](https://github.com/rabeehk/hyperformer), with the following major changes:
- Added several tasks and relevant preprocessing, including MRQA (with and without sliding windows), xsum, CNN/Daily Mail, Wiki Lingua, abductive NLI, and adversarial NLI.
- Fixed some minor bugs including the 'split validation test' not being applied to the training set.
- Added new adapter and parameter generation code in `hyperformer/modeling`, and removed the old adapter code. Added relevant training arguments for these setups (encoder/decoder adapter sizes, etc).
- Updated the trainer to save copies of generated answers along with likelihood scores for MRQA evaluation.

## Installation

Install pytorch (1.10 recommended). Install required packages, preferably in a virtualenv: `pip install -r requirements.txt`.

Navigate into the `hyperformer` directory, and then you can run any configuration with `python finetune_trainer.py configs/<config>`.

For example, for GLUE, the Hyperdecoder model can be run with `python finetune_trainer.py configs/glue_ablations/glue_manual_generated.json`. The trained model will be placed in `hyperformer/output`, and the evaluation logs can be found in `hyperformer/output/log.txt`. You can control how often the model is evaluated and saved with `eval_steps` and `save_steps` in the config.

## Config

Some useful config items:
- `{de,en}coder_adapter`: controls how we adapt the encoder/decoder. Can be `none` (no adapters), `manual` (regular adapters), `generated` (generated adapters). Note `generated` in the encoder results in the encoder being run twice: once with adapters to produce an embedding that is then used to adapt the encoder for a second run (the output of which is passed to the decoder as usual).
- `freeze_model/unfreeze_{en,de}coder_adapters/unfreeze_{en,de}coder`: freeze/unfreeze the relevant parts of the model for training. This is accomplished through the `requires_grad` flag. Usually we freeze the whole model and then unfreeze the encoder/decoder adapter bits.
- `max_steps`: controls how many training steps. Note that `num_train_epochs` is ignored, we just train based on steps and do not distinguish any sort of epoch boundary.
- `{en,de}coder_adapter_dim`: controls the adapter bottleneck size. You can control separately for encoder/decoder.
- `hypernetwork_bottleneck`: controls the hypernetwork bottleneck size (see paper for details on this).
- `split_validation_test`: split the validation sets of datasets into validation and test splits, so we can early-stop based on validation metrics and then eval on the test split. This is what we do for most experiments in our paper.

Most other config options are hopefully either straightforward or do not need to be changed. Note that the hyperdecoder model is achieved by setting `encoder_adapter: manual, decoder_adapter: generated`.

### MRQA Evaluation

Due to the sliding window nature of MRQA, evaluation is separately to running the model. When running evaluation with MRQA, the model will at the end output answer files for the validation and test sets as `<step_num>predicted_answers.json` and `<step_num>predicted_answers_test.json`. 

After getting these files, navigate into `mrqa_eval` and run the `construct_eval_folders.sh` script, which will download the MRQA evaluation data for you and place it in useful folders. You can then run evaluation on *in-domain* data as follows:

`for file in in-domain/*.gz; do; echo $file; python eval.py $file <predicted_answers.json file>; done`

The *out-domain* data can be evaluated similarly:

`for file in out-domain/*.gz; do; echo $file; python eval.py $file <predicted_answers_test.json file>; done`

In both cases, you will get terminal output that prints (a) the name of the dataset being evaluated, and then (b) the performance on that particular dataset. Note our evaluation script is the same as the original MRQA evaluation script but with some extra code to handle picking the highest likelihood answer (as the model output saves these scores but does not filter on them). As such, it is fairly simple to convert our `predicted_answer.json` files to the format needed for the original MRQA evaluation script.

## Training

Navigate into the `hyperformer` directory, and run the appropriate config.

We only used one GPU during finetuning, so the code is almost definitely broken for distributed setups. Sorry!


## Citation

If you found this code or our paper useful, please cite us:
```
@inproceedings{hamish-et-al-hyperdecoder,
    title = "Hyperdecoders: Instance-specific decoders for multi-task NLP",
    author = "Ivison, Hamish and Peters, Matthew E.",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
}
```