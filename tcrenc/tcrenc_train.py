from pathlib import Path

from torchtune import config as torchtune_config
from sklearn.model_selection import train_test_split

from utils.argparsers import train_argparser
from utils.basic import read_config, filter_input, set_device
from utils.train import input_process, saving_results

SCRIPT_TYPE = 'train'


def main():
    args = train_argparser()

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

        model_cdr3.to(device)

        if args.split != 1:
            train_cdr3_set, val_cdr3_set = train_test_split(data_cdr3, test_size=1-args.split, random_state=42)

            train_cdr3_dataloader = model_cdr3.input_data_process(inp_data=train_cdr3_set['cdr3'])
            val_cdr3_dataloader = model_cdr3.input_data_process(inp_data=val_cdr3_set['cdr3'])

            model_cdr3.model_process(input_dataloader=train_cdr3_dataloader,
                                     device=device,
                                     criterion=loss_function,
                                     process_type=SCRIPT_TYPE,
                                     test_dataloader=val_cdr3_dataloader)
        else:
            cdr3_dataloader = model_cdr3.input_data_process(inp_data=data_cdr3['cdr3'])

            model_cdr3.model_process(input_dataloader=cdr3_dataloader,
                                     device=device,
                                     criterion=loss_function,
                                     process_type=SCRIPT_TYPE)

        if args.weights_save:
            saving_results(model_cdr3, output_path, args.embed_type, 'cdr3')

    if gen_config['epitope_ex'] is True:

        inp_data_epitope = inp_data.antigen_epitope.to_frame()
        data_epitope = filter_input(inp_data_epitope, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope')

        model_epitope.to(device)

        if args.split != 1:
            train_epitope_set, val_epitope_set = train_test_split(data_epitope, test_size=1-args.split, random_state=42)

            train_epitope_dataloader = model_epitope.input_data_process(inp_data=train_epitope_set['antigen_epitope'])
            val_epitope_dataloader = model_epitope.input_data_process(inp_data=val_epitope_set['antigen_epitope'])

            model_epitope.model_process(input_dataloader=train_epitope_dataloader,
                                        device=device,
                                        criterion=loss_function,
                                        process_type=SCRIPT_TYPE,
                                        test_dataloader=val_epitope_dataloader)
        else:
            epitope_dataloader = model_epitope.input_data_process(inp_data=data_epitope['antigen_epitope'])

            model_epitope.model_process(input_dataloader=epitope_dataloader,
                                        device=device,
                                        criterion=loss_function,
                                        process_type=SCRIPT_TYPE)

        if args.weights_save:
            saving_results(model_epitope, output_path, args.embed_type, 'epitope')

    print("Script end")


if __name__ == '__main__':
    main()
