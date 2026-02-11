from pathlib import Path

from torchtune import config as torchtune_config

from utils.argparsers import validate_argrapser
from utils.basic import read_config, filter_input, set_device, input_process
from utils.validate import make_report
from models.models_list import load_model_for_validate


SCRIPT_TYPE = 'validate'


def main(args=None):
    args = validate_argrapser()

    args_check(args)

    gen_config = read_config('./tcrenc/configs/config_general.yaml')

    device = set_device(gen_config['USE_GPU'])

    Model, model_config = load_model_for_validate(args, script_type=SCRIPT_TYPE)

    # Special configurations
    gen_config.update(model_config)
    loss_function = torchtune_config.instantiate(gen_config['LOSS_FUNCTION'])

    inp_data = input_process(args.input, gen_config)

    if args.cdr:
        gen_config['cdr3_ex'] = True
        gen_config['epitope_ex'] = False

    elif args.epitope:
        gen_config['epitope_ex'] = True
        gen_config['cdr3_ex'] = False

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Processing model based on request
    if gen_config['cdr3_ex']:

        inp_data_cdr3 = inp_data.cdr3.to_frame()
        data_cdr3 = filter_input(inp_data_cdr3, gen_config)

        model_cdr3 = Model(gen_config, seq_type='cdr3',
                           device=device)

        model_cdr3. weight_load()

        input_seqs, output_seqs, loss_value = model_cdr3.validation_on_seqs(
            input_data=data_cdr3,
            loss_function=loss_function,
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

        model_epitope = Model(gen_config, seq_type='antigen_epitope',
                              device=device)

        model_epitope. weight_load()

        input_seqs, output_seqs, loss_value = model_epitope.validation_on_seqs(
            input_data=data_epitope,
            loss_function=loss_function,
        )

        make_report(input_seqs,
                    output_seqs,
                    output_path,
                    embed_type=args.embed_type,
                    seq_type='antigen_epitope',
                    loss_value=loss_value
                    )

    print("All files saved!")


def args_check(args):

    if args.input != 'VDJdb' and args.cdr:
        raise ValueError('cdr option only for VDJdb')
    elif args.input != 'VDJdb' and args.epitope:
        raise ValueError('epitope option only for VDJdb')
    elif args.input == 'VDJdb' and args.epitope and args.cdr:
        raise ValueError('Do not use epitope and cdr flags for VDJdb option')


if __name__ == '__main__':
    main()
