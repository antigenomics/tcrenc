from pathlib import Path

from torchtune import config as torchtune_config
from sklearn.model_selection import train_test_split

from utils.argparsers import train_argparser
from utils.basic import read_config, filter_input, set_device, input_process
from models.models_list import load_model_for_train


SCRIPT_TYPE = 'train'


def main(args=None):
    args = train_argparser()

    args_check(args)

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    Model, model_config = load_model_for_train(args, script_type=SCRIPT_TYPE)

    gen_config.update(model_config)
    loss_function = torchtune_config.instantiate(gen_config['LOSS_FUNCTION'])

    inp_data = input_process(args.input, gen_config)

    if args.cdr:
        gen_config['cdr3_ex'] = True
        gen_config['epitope_ex'] = False

    elif args.epitope:
        gen_config['epitope_ex'] = True
        gen_config['cdr3_ex'] = False

    if not args.encoder_train and not args.decoder_train:
        full_autoencoder_train(args, inp_data, gen_config,
                               Model, device, loss_function, output_path)
    else:
        part_autoencoder_train(args, inp_data, gen_config,
                               Model, device, loss_function, output_path)


def full_autoencoder_train(args, inp_data, gen_config, Model, device, loss_function, output_path):

    if gen_config['cdr3_ex']:

        inp_data_cdr3 = inp_data.cdr3.to_frame()
        data_cdr3 = filter_input(inp_data_cdr3, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3', device=device)

        if args.split != 1:
            train_cdr3_set, test_cdr3_set = train_test_split(data_cdr3,
                                                             test_size=1-args.split,
                                                             random_state=42)

        else:
            train_cdr3_set = data_cdr3
            test_cdr3_set = None

        model_cdr3.model_train(train_data=train_cdr3_set,
                               criterion=loss_function,
                               input_train_seqs=train_cdr3_set['cdr3'],
                               test_data=test_cdr3_set)

        if args.weights_save:
            model_cdr3.save_model(output_path)

    if gen_config['epitope_ex']:

        inp_data_epitope = inp_data.antigen_epitope.to_frame()
        data_epitope = filter_input(inp_data_epitope, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope',
                              device=device)

        if args.split != 1:
            train_epitope_set, test_epitope_set = train_test_split(data_epitope,
                                                                   test_size=1-args.split,
                                                                   random_state=42)

        else:
            train_epitope_set = data_epitope
            test_epitope_set = None

        model_epitope.model_train(train_data=train_epitope_set,
                                  criterion=loss_function,
                                  input_train_seqs=train_epitope_set['antigen_epitope'],
                                  test_data=test_epitope_set)

        if args.weights_save:
            model_epitope.save_model(output_path)


def part_autoencoder_train(args, inp_data, gen_config, Model, device, loss_function, output_path):

    if gen_config['cdr3_ex'] and gen_config['epitope_ex']:
        raise ValueError('Only one sequence type for encoder/decoder training')

    if gen_config['cdr3_ex']:

        inp_data_filtered = filter_input(inp_data, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3', device=device)

        if args.split != 1:
            train_cdr3_set, test_cdr3_set = train_test_split(inp_data_filtered,
                                                             test_size=1-args.split,
                                                             random_state=42)
        else:
            train_cdr3_set = inp_data_filtered
            test_cdr3_set = None

        model_cdr3.model_train(train_data=train_cdr3_set,
                               criterion=loss_function,
                               input_train_seqs=train_cdr3_set['cdr3'],
                               test_data=test_cdr3_set)

        if args.weights_save:
            model_cdr3.save_model(output_path)

    if gen_config['epitope_ex']:

        inp_data_filtered = filter_input(inp_data, gen_config)

        model_epitope = Model(gen_config, seq_type='antigen_epitope',
                              device=device)

        if args.split != 1:
            train_epitope_set, test_epitope_set = train_test_split(inp_data_filtered,
                                                                   test_size=1-args.split,
                                                                   random_state=42)
        else:
            train_epitope_set = inp_data_filtered
            test_epitope_set = None

        model_epitope.model_train(train_data=train_epitope_set,
                                  criterion=loss_function,
                                  input_train_seqs=train_epitope_set['antigen_epitope'],
                                  test_data=test_epitope_set)

        if args.weights_save:
            model_epitope.save_model(output_path)


def args_check(args):

    if args.input != 'VDJdb' and args.cdr:
        raise ValueError('cdr option only for VDJdb')
    elif args.input != 'VDJdb' and args.epitope:
        raise ValueError('epitope option only for VDJdb')
    elif args.input == 'VDJdb' and args.epitope and args.cdr:
        raise ValueError('Do not use epitope and cdr flags for VDJdb option')

    if args.encoder_train and args.decoder_train:
        raise ValueError('Chose only one option what to train.')
    else:
        if args.input == 'VDJdb' and (args.encoder_train or args.decoder_train):
            raise ValueError('No VDJdb option for encoder/decoder training')


if __name__ == '__main__':
    main()
