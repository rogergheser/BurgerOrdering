import argparse
from argparse import Namespace

from utils import MODELS, TEMPLATES, generate, load_model, set_hf_home


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m query_model",
        description="Query a specific model with a given input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=list(MODELS.keys()),
        help="The model to query.",
    )
    parser.add_argument(
        "input",
        metavar="INPUT_TEXT",
        type=str,
        help="The input to query the model with.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="The system prompt to use for the model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["f32", "bf16"],
        default="f32",
        help="The data type to use for the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum sequence length to use for the model.",
    )
    parser.add_argument(
        "--return-full",
        action="store_true",
        help="Return the full output.",
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=".env",
        help="The path to the .env file.",
    )

    parsed_args = parser.parse_args()
    parsed_args.chat_template = TEMPLATES[parsed_args.model_name]
    parsed_args.model_name = MODELS[parsed_args.model_name]

    return parsed_args


def main():
    args = get_args()
    set_hf_home(args.dotenv_path)
    model, tokenizer = load_model(args)
    input_text = args.chat_template.format(args.system_prompt, args.input)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    reply = generate(model, inputs, tokenizer, args)
    print(reply)


if __name__ == "__main__":
    main()