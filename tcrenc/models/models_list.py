from tcrenc.utils.basic import read_config


def load_model_for_train(args, script_type):

    if script_type != 'train':
        raise ValueError('Wrong model loading function for this type of script')

    if not args.encoder_train and not args.decoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

        elif args.embed_type == 'atchley':
            model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
            # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    elif args.encoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.encoder_onehot import Encoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.encoder_kidera import Encoder_kidera as Model

        elif args.embed_type == 'atchley':
            model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
            # from models.autoencoder_atchley.encoder_atchley import Encoder_atchley as Model

    elif args.decoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.decoder_onehot import Decoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.decoder_kidera import Decoder_kidera as Model

        elif args.embed_type == 'atchley':
            model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
            # from models.autoencoder_atchley.decoder_atchley import Decoder_atchley as Model

    return Model, model_config


def load_model_for_validate(args, script_type):

    if script_type != 'validate':
        raise ValueError('Wrong model loading function for this type of script')

    if not args.decoder:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

        elif args.embed_type == 'atchley':
            model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
            # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model
    else:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.decoder_onehot import Decoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.decoder_kidera import Decoder_kidera as Model

        elif args.embed_type == 'atchley':
            model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
            # from models.autoencoder_atchley.decoder_atchley import Decoder_atchley as Model

    return Model, model_config


def load_model_for_run(args, script_type):

    if script_type != 'run':
        raise ValueError('Wrong model loading function for this type of script')

    if args.embed_type == 'onehot':
        model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
        from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

    elif args.embed_type == 'kidera':
        model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
        from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

    elif args.embed_type == 'atchley':
        model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        # from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    return Model, model_config
