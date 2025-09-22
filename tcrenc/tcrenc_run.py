from pathlib import Path

import torch

from utils.argparsers import run_argparser
from utils.basic import read_config, filter_input, set_device, input_process
from utils.run import saving_results
from models.models_list import load_model


SCRIPT_TYPE = 'run'


def main():
    args = run_argparser()

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    Model, model_config = load_model(args, script_type=SCRIPT_TYPE)

    # Special configurations
    gen_config.update(model_config)

    if args.decoder and args.input == 'VDJdb':
        raise ValueError('No VDJdb option for decoder')

    inp_data = input_process(args.input, gen_config)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Processing model based on request
    if gen_config['cdr3_ex']:

        if args.decoder:
            data_cdr3 = inp_data.copy()
            data_cdr3.drop(columns='cdr3', inplace=True)

            if data_cdr3.isnull().values.any():
                raise ValueError('There is NA values in embeddings data')
        else:
            inp_data_cdr3 = inp_data.cdr3.to_frame()
            data_cdr3 = filter_input(inp_data_cdr3, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3')

        model_cdr3.load_state_dict(torch.load(gen_config['WEIGHTS_CDR3'],
                                              map_location=device,
                                              weights_only=True))
        model_cdr3.to(device)

        if args.decoder:
            cdr3_df, _ = model_cdr3.make_seq_from_embeddings(input_embds=data_cdr3,
                                                             device=device)
        else:
            cdr3_df = model_cdr3.make_embeddings_from_seq(input_data=data_cdr3,
                                                          device=device)

        saving_results(cdr3_df, output_path, args, 'cdr3')

    if gen_config['epitope_ex']:

        if args.decoder:
            data_epitope = inp_data.copy()
            data_epitope.drop(columns='antigen_epitope', inplace=True)

            if data_epitope.isnull().values.any():
                raise ValueError('There is NA values in embeddings data')
        else:
            inp_data_epitope = inp_data.antigen_epitope.to_frame()
            data_epitope = filter_input(inp_data_epitope, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope')

        model_epitope.load_state_dict(torch.load(gen_config['WEIGHTS_EPIOPE'],
                                                 map_location=device,
                                                 weights_only=True))
        model_epitope.to(device)

        if args.decoder:
            epitope_df, _ = model_epitope.make_seq_from_embeddings(input_embds=data_epitope,
                                                                   device=device)
        else:
            epitope_df = model_epitope.make_embeddings_from_seq(input_data=data_epitope,
                                                                device=device)

        saving_results(epitope_df, output_path, args, 'antigen_epitope')

    print("All files saved!")


if __name__ == '__main__':
    main()
