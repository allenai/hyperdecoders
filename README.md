# Param-efficient Self-conditioning

Basically, models self-steering themself. This repo is built off the hyperformer repo, and draws a little bit from the hypter repo.

## How to run the models

Install pytorch (1.10.1 recommended), then the required packages in requirements.txt. Then `cd` into `hyperformer`.

Then just run `python finetune_trainer.py configs/<config>` with the config setup of your choice. Currently I usually hack in changes I want to test but I'm in the process of slowly adding proper configuration options for this.
