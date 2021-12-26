'''small script to just generate the enc/dec variants for ablations'''
import json

with open('glue_none_none.json') as f:
    base_config = json.load(f)

for enc_setup in ['none', 'manual', 'task', 'generated']:
    for dec_setup in ['none', 'manual', 'task', 'generated']:
        config = base_config.copy()
        config['encoder_adapter'] = enc_setup
        # if ff, unfreeze all, else, just adapter stuff
        if enc_setup == 'none':
            config['unfreeze_encoder_adapters'] = False
            config['unfreeze_encoder'] = True
        else:
            config['unfreeze_encoder_adapters'] = True
            config['unfreeze_encoder'] = False
        if enc_setup in ['task', 'generated']:
            config['adapter_norm_input'] = True
        config['decoder_adapter'] = dec_setup
        # if ff, unfreeze all, else, just adapter stuff
        if dec_setup == 'none':
            config['unfreeze_decoder_adapters'] = False
            config['unfreeze_decoder'] = True
        else:
            config['unfreeze_decoder_adapters'] = True
            config['unfreeze_decoder'] = False
        # if both none, then dont freeze at all
        if enc_setup == 'none' and dec_setup == 'none':
            config['freeze_model'] = False
        else:
            config['freeze_model'] = True
        config['output_dir'] = f'glue_{enc_setup}_{dec_setup}'
        with open(f'glue_{enc_setup}_{dec_setup}.json', 'w') as f:
            json.dump(config, f, indent=4)