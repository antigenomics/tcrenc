import argparse


def run_argparser():
    parser = argparse.ArgumentParser(description="Making embeddings from TCR or epitope sequences")
    parser.add_argument(
        "--input", type=str, required=True, help="VDJdb option or path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        required=True,
        choices=["onehot", "kidera"],
        help="Type of sequence representation: onehot or kidera factors",
    )
    parser.add_argument(
        "--decoder",
        action="store_true",
        help="Option to use decoder only (default: false)",
    )
    parser.add_argument(
        "--cdr",
        action="store_true",
        help="Option to specify decoder seq type (default: false)",
    )
    parser.add_argument(
        "--epitope",
        action="store_true",
        help="Option to specify decoder seq type (default: false)",
    )

    return parser.parse_args()


def train_argparser():
    parser = argparse.ArgumentParser(description="Train models for TCR or epitope sequences")
    parser.add_argument(
        "--input", type=str, required=True, help="VDJdb option or path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory for weights saving"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        required=True,
        choices=["onehot", "kidera", "atchley"],
        help="Type of sequence representation: onehot or kidera factors or atchley factors",
    )
    parser.add_argument(
        "--weights_save",
        action="store_true",
        help="Option to save weights or not (default: false)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=1,
        help="Option to split data to train/test sets",
    )
    parser.add_argument(
        "--encoder_train",
        action="store_true",
        help="Option to train only encoder (default: false)",
    )
    parser.add_argument(
        "--decoder_train",
        action="store_true",
        help="Option to train only decoder (default: false)",
    )
    parser.add_argument(
        "--cdr",
        action="store_true",
        help="Option to specify VDJdb seq type",
    )
    parser.add_argument(
        "--epitope",
        action="store_true",
        help="Option to specify VDJdb seq type",
    )

    return parser.parse_args()


def validate_argrapser():
    parser = argparse.ArgumentParser(description="Check model on input TCR or epitope sequences")
    parser.add_argument(
        "--input", type=str, required=True, help="VDJdb option or path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        required=True,
        choices=["onehot", "kidera"],
        help="Type of sequence representation: onehot or kidera factors",
    )
    parser.add_argument(
        "--decoder",
        action="store_true",
        help="Option to use decoder only (default: false)",
    )
    parser.add_argument(
        "--cdr",
        action="store_true",
        help="Option to specify VDJdb seq type",
    )
    parser.add_argument(
        "--epitope",
        action="store_true",
        help="Option to specify VDJdb seq type",
    )

    return parser.parse_args()
