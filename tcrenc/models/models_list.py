from tcrenc.utils.basic import read_config
"""
Model loader utilities for the tcrenc autoencoder package.

This module provides functions to select and load the appropriate model class
and its associated configuration based on the requested embedding type and
operation mode (train / validate / run). Each loader function returns a tuple
(ModelClass, model_config).

When adding a new model:
- Add the new embedding type string to AV_EMBD_TYPE.
- Add branches in required loader function (load_model_for_train,
  load_model_for_validate, load_model_for_run) to load the correct config file
  and import the corresponding model.

Notes:
- 'atchley' model will be available in future releases.
- The loader functions raise ValueError if an unsupported embed_type is given.
"""

AV_EMBD_TYPE = ["onehot", "kidera"]


def load_model_for_train(args, script_type):

    if args.embed_type not in AV_EMBD_TYPE:
        raise ValueError('Wrong model or embed type!')

    if not args.encoder_train and not args.decoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

        # elif args.embed_type == 'atchley':
        #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        #     from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    elif args.encoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.encoder_onehot import Encoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.encoder_kidera import Encoder_kidera as Model

        # elif args.embed_type == 'atchley':
        #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        #     from models.autoencoder_atchley.encoder_atchley import Encoder_atchley as Model

    elif args.decoder_train:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.decoder_onehot import Decoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.decoder_kidera import Decoder_kidera as Model

        # elif args.embed_type == 'atchley':
        #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        #     from models.autoencoder_atchley.decoder_atchley import Decoder_atchley as Model

    return Model, model_config


def load_model_for_validate(args, script_type):

    if args.embed_type not in AV_EMBD_TYPE:
        raise ValueError('Wrong model or embed type!')

    if not args.decoder:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

        # elif args.embed_type == 'atchley':
        #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        #     from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model
    else:
        if args.embed_type == 'onehot':
            model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
            from models.autoencoder_onehot.decoder_onehot import Decoder_onehot as Model

        elif args.embed_type == 'kidera':
            model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
            from models.autoencoder_kidera.decoder_kidera import Decoder_kidera as Model

        # elif args.embed_type == 'atchley':
        #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
        #     from models.autoencoder_atchley.decoder_atchley import Decoder_atchley as Model

    return Model, model_config


def load_model_for_run(args, script_type):

    if args.embed_type not in AV_EMBD_TYPE:
        raise ValueError('Wrong model or embed type!')

    if args.embed_type == 'onehot':
        model_config = read_config('./tcrenc/configs/config_onehot.yaml', script_type=script_type)
        from models.autoencoder_onehot.autoencoder_onehot import Autoencoder_onehot as Model

    elif args.embed_type == 'kidera':
        model_config = read_config('./tcrenc/configs/config_kidera.yaml', script_type=script_type)
        from models.autoencoder_kidera.autoencoder_kidera import Autoencoder_kidera as Model

    # elif args.embed_type == 'atchley':
    #     model_config = read_config('./tcrenc/configs/config_atchley.yaml', script_type=script_type)
    #     from models.autoencoder_atchley.autoencoder_atchley import Autoencoder_atchley as Model

    return Model, model_config
