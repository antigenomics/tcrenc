from pathlib import Path

import torch
from torchtune import config

from utils.argparsers import run_argparser
from utils.basic import read_config, filter_input, set_device
from utils.run import input_process, model_process, saving_results

SCRIPT_TYPE = 'run'


def main():
    args = run_argparser()

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    inp_data = input_process(args.input, gen_config)

    data = filter_input(inp_data, gen_config)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Loading model
    if args.embed_type == 'onehot':
        script_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

    elif args.embed_type == 'kidera':
        script_config = read_config('./configs/config_kidera.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

    elif args.embed_type == 'atchley':
        script_config = read_config('./configs/config_atchley.yaml', script_type=SCRIPT_TYPE)
        # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    # Special configurations
    gen_config.update(script_config)
    loss_function = config.instantiate(gen_config['LOSS_FUNCTION'])

    # Processing model based on request
    if gen_config['cdr3_ex'] is True:
        model_cdr3 = Model(gen_config, seq_type='cdr3')

        model_cdr3.load_state_dict(torch.load(gen_config['WEIGHTS_CDR3'],
                                              weights_only=True))
        model_cdr3.to(device)

        cdr3_dataloader = model_cdr3.input_data_process(inp_data=data['cdr3'])

        cdr3_output = model_process(input_dataloader=cdr3_dataloader,
                                    device=device, model=model_cdr3,
                                    criterion=loss_function)

        saving_results(cdr3_output, output_path, args.embed_type, gen_config, data['cdr3'])

    if gen_config['epitope_ex'] is True:
        model_epitope = Model(gen_config, seq_type='antigen_epitope')

        model_epitope.load_state_dict(torch.load(gen_config['WEIGHTS_EPIOPE'],
                                                 weights_only=True))
        model_epitope.to(device)

        epitope_dataloader = model_epitope.input_data_process(inp_data=data['antigen_epitope'])

        epitope_output = model_process(input_dataloader=epitope_dataloader,
                                       device=device, model=model_epitope,
                                       criterion=loss_function)

        saving_results(epitope_output, output_path, args.embed_type, gen_config, data['antigen_epitope'])

    print("All files saved!")


if __name__ == '__main__':
    main()
