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
        "--residual_block",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use residual block for kidera factors representation (default: false)",
    )

    return parser.parse_args()


def train_argparser():
    pass


def validate_argrapser():
    pass
