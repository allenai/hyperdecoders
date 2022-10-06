# Hyperdecoders

Code for the hyperdecoders paper. This is built off the [hyperformer codebase](https://github.com/rabeehk/hyperformer), with the following major changes:
- Added several tasks and relevant preprocessing, including MRQA (with and without sliding windows), xsum, CNN/Daily Mail, Wiki Lingua, abductive NLI, and adversarial NLI.
- Fixed some minor bugs including the 'split validation test' not being applied to the training set.
- Added new adapter and parameter generation code in `hyperformer/modeling`, and removed the old adapter code. Added relevant training arguments for these setups (encoder/decoder adapter sizes, etc).
- Updated the trainer to save copies of generated answers along with likelihood scores for MRQA evaluation.

## Installation

Install pytorch (1.10 recommended). Install required packages, preferably in a virtualenv: `pip install -r requirements.txt`.

Navigate into the `hyperformer` directory, and then you can run any configuration with `python finetune_trainer.py configs/<config>`.

For example, for GLUE, the Hyperdecoder model can be run with `python finetune_trainer.py configs/glue_ablations/glue_manual_generated.json`.

## Evaluation

Navigate into the `hyperformer` directory, and run the appropriate config.

### MRQA Evaluation

Due to the sliding window nature of MRQA, evaluation is separately to running the model. When running evaluation with MRQA, the model will at the end output answer files for the validation and test sets as `<step_num>predicted_answers.json` and `<step_num>predicted_answers_test.json`. 

After getting these files, navigate into `mrqa_eval` and run the `construct_eval_folders.sh` script, which will download the MRQA evaluation data for you and place it in useful folders. You can then run evaluation on *in-domain* data as follows:

`for file in in-domain/*.gz; do; echo $file; python eval.py $file <predicted_answers.json file>; done`

The *out-domain* data can be evaluated similarly:

`for file in out-domain/*.gz; do; echo $file; python eval.py $file <predicted_answers_test.json file>; done`

In both cases, you will get terminal output that prints (a) the name of the dataset being evaluated, and then (b) the performance on that particular dataset. Note our evaluation script is the same as the original MRQA evaluation script but with some extra code to handle picking the highest likelihood answer (as the model output saves these scores but does not filter on them). As such, it is fairly simple to convert our `predicted_answer.json` files to the format needed for the original MRQA evaluation script.

## Training

Navigate into the `hyperformer` directory, and run the appropriate config.

We only used one GPU during finetuning, so this library is not guaranteed (in fact, it may just break) for distributed setups.
