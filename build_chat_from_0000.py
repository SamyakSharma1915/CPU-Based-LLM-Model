import argparse
import json
from pathlib import Path


def normalize_text(text):
    return " ".join(str(text).split())


def format_prompt(turns):
    parts = []
    for role, value in turns:
        tag = "<|user|>" if role == "human" else "<|assistant|>"
        parts.append(f"{tag} {normalize_text(value)}")
    parts.append("<|assistant|>")
    return " ".join(parts)


def build_examples(items, max_examples=None):
    examples = []
    for item in items:
        convo = item.get("conversations", [])
        history = []
        for turn in convo:
            role = turn.get("from")
            value = normalize_text(turn.get("value", ""))
            if not value:
                continue

            if role == "gpt" and history:
                prompt = format_prompt(history)
                examples.append(
                    {
                        "title": item.get("id", "conversation"),
                        "prompt": prompt,
                        "response": value,
                    }
                )
                if max_examples is not None and len(examples) >= max_examples:
                    return examples

            history.append((role, value))

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert 0000.json conversations to fine_tune.jsonl")
    parser.add_argument("--input", default="0000.json")
    parser.add_argument("--output", default="data/fine_tune.jsonl")
    parser.add_argument("--max_examples", type=int, default=20000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    items = json.loads(input_path.read_text(encoding="utf-8", errors="replace"))
    examples = build_examples(items, max_examples=args.max_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
