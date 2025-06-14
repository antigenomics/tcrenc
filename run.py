import argparse
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Process embeddings and save results")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        required=True,
        choices=["onehot", "kidera"],
        help="Type of embedding: onehot or kidera",
    )
    parser.add_argument(
        "--residual_block",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use residual block (default: false)",
    )

    args = parser.parse_args()

    # Checking
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist")

    # Making directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Definition of script
    if args.embed_type == "onehot":
        script_name = "./code/code_onehot/process_onehot.py"
    else:
        script_name = "./code/code_kidera/process_kidera.py"

    # Residual block
    residual_block = args.residual_block.lower() == "true"
    if script_name == "./code/code_kidera/process_kidera.py":
        command = [
            "python",
            script_name,
            "--input",
            args.input,
            "--output",
            args.output,
            "--residual_block",
            str(residual_block),
        ]
    else:
        command = [
            "python",
            script_name,
            "--input",
            args.input,
            "--output",
            args.output,
        ]
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
