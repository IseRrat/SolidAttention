#!/usr/bin/env python3
"""Create a fixed-token replay configuration from an existing reference result."""
import argparse
import json
from pathlib import Path


def prepare(reference, template, output):
    reference_path = Path(reference).resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    template_path = Path(template).resolve()
    mode = json.loads(template_path.read_text(encoding="utf-8"))
    config = json.loads((reference_path.parent / "config.json").read_text(encoding="utf-8"))
    ids = reference["generated_token_ids"]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("reference must contain generated token IDs")
    inputs = reference["decode_input_token_ids"]
    if len(inputs) + 1 != len(ids) or any(type(token) is not int or token < 0 for token in inputs):
        raise ValueError("reference decode inputs differ from recorded output length")
    config["prompt"] = str((reference_path.parent / "prompt.txt").resolve(strict=True))
    config.pop("attention", None)
    if "attention" in mode:
        config["attention"] = mode["attention"]
        if "session_file" in config["attention"]:
            config["attention"]["session_file"] = str((template_path.parent / config["attention"]["session_file"]).resolve())
        if "selection_replay" in config["attention"]:
            config["attention"]["selection_replay"] = str((template_path.parent / config["attention"]["selection_replay"]).resolve())
    config["output"] = str(Path(output).resolve())
    config["max_new_tokens"] = len(ids)
    config["replay_token_ids"] = inputs
    if "prompt_token_ids" in reference:
        config["prompt_token_ids"] = reference["prompt_token_ids"]
    config["dump_logits"] = True
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = prepare(args.reference, args.template, args.output)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
