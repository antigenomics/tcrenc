from pathlib import Path

import torch
from torchtune import config as torchtune_config

from utils.argparsers import validate_argrapser
from utils.basic import read_config, filter_input, set_device, input_process
from utils.validate import make_report


SCRIPT_TYPE = 'run'


def load_model(args):
    """
    Here you can add model
    """
    if args.embed_type == 'onehot':
        model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

    elif args.embed_type == 'kidera':
        model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

    elif args.embed_type == 'atchley':
        model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=SCRIPT_TYPE)
        # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model
    return Model, model_config


def main():
    args = validate_argrapser()

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    Model, model_config = load_model(args)

    # Special configurations
    gen_config.update(model_config)
    loss_function = torchtune_config.instantiate(gen_config['LOSS_FUNCTION'])

    inp_data = input_process(args.input, gen_config)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Processing model based on request
    if gen_config['cdr3_ex']:

        inp_data_cdr3 = inp_data.cdr3.to_frame()
        data_cdr3 = filter_input(inp_data_cdr3, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3')

        model_cdr3.load_state_dict(torch.load(gen_config['WEIGHTS_CDR3'],
                                              map_location=device,
                                              weights_only=True))
        model_cdr3.to(device)

        input_seqs, output_seqs, loss_value = model_cdr3.validation_on_seqs(
            input_data=data_cdr3,
            loss_function=loss_function,
            device=device,
        )

        make_report(input_seqs,
                    output_seqs,
                    output_path,
                    embed_type=args.embed_type,
                    seq_type='cdr3',
                    loss_value=loss_value
                    )

    if gen_config['epitope_ex']:

        inp_data_epitope = inp_data.antigen_epitope.to_frame()
        data_epitope = filter_input(inp_data_epitope, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope')

        model_epitope.load_state_dict(torch.load(gen_config['WEIGHTS_EPIOPE'],
                                                 map_location=device,
                                                 weights_only=True))
        model_epitope.to(device)

        input_seqs, output_seqs, loss_value = model_epitope.validation_on_seqs(
            input_data=data_epitope,
            loss_function=loss_function,
            device=device,
        )

        make_report(input_seqs,
                    output_seqs,
                    output_path,
                    embed_type=args.embed_type,
                    seq_type='antigen_epitope',
                    loss_value=loss_value
                    )

    print("All files saved!")


if __name__ == '__main__':
    main()
