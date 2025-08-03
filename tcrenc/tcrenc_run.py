from pathlib import Path

import torch
from torchtune import config as torchtune_config

from utils.argparsers import run_argparser
from utils.basic import read_config, filter_input, set_device
from utils.run import input_process, saving_results

SCRIPT_TYPE = 'run'


def main():
    args = run_argparser()

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    # Loading model
    if args.embed_type == 'onehot':
        script_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

    elif args.embed_type == 'kidera':
        script_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=SCRIPT_TYPE)
        from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

    elif args.embed_type == 'atchley':
        script_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=SCRIPT_TYPE)
        # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    # Special configurations
    gen_config.update(script_config)
    loss_function = torchtune_config.instantiate(gen_config['LOSS_FUNCTION'])

    inp_data = input_process(args.input, gen_config)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Processing model based on request
    if gen_config['cdr3_ex'] is True:

        inp_data_cdr3 = inp_data.cdr3.to_frame()
        data_cdr3 = filter_input(inp_data_cdr3, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3')

        model_cdr3.load_state_dict(torch.load(gen_config['WEIGHTS_CDR3'],
                                              map_location=device,
                                              weights_only=True))
        model_cdr3.to(device)

        cdr3_dataloader = model_cdr3.input_data_process(inp_data=data_cdr3['cdr3'])

        cdr3_output = model_cdr3.model_process(input_dataloader=cdr3_dataloader,
                                               device=device,
                                               criterion=loss_function,
                                               process_type=SCRIPT_TYPE)

        cdr3_df = model_cdr3.embeddings_data_process(cdr3_output, data_cdr3['cdr3'])

        saving_results(cdr3_df, output_path, args.embed_type, data_cdr3['cdr3'].name)

    if gen_config['epitope_ex'] is True:

        inp_data_epitope = inp_data.antigen_epitope.to_frame()
        data_epitope = filter_input(inp_data_epitope, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope')

        model_epitope.load_state_dict(torch.load(gen_config['WEIGHTS_EPIOPE'],
                                                 map_location=device,
                                                 weights_only=True))
        model_epitope.to(device)

        epitope_dataloader = model_epitope.input_data_process(inp_data=data_epitope['antigen_epitope'])

        epitope_output = model_epitope.model_process(input_dataloader=epitope_dataloader,
                                                     device=device,
                                                     criterion=loss_function,
                                                     process_type=SCRIPT_TYPE)

        epitope_df = model_epitope.embeddings_data_process(epitope_output, data_epitope['antigen_epitope'])

        saving_results(epitope_df, output_path, args.embed_type, data_epitope['antigen_epitope'].name)

    print("All files saved!")


if __name__ == '__main__':
    main()
