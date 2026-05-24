import argparse
import json
from pathlib import Path


TITLE_PROMPT_TEMPLATES = [
    "What is {title}?",
    "Tell me about {title}.",
    "Explain {title} in simple words.",
    "Give me an overview of {title}.",
    "What should I know about {title}?",
    "Can you describe {title}?",
]

DETAIL_PROMPT_TEMPLATES = [
    "Explain this topic clearly: {title}.",
    "Teach me about {title} like I am a beginner.",
    "Summarize {title}.",
    "Share some important facts about {title}.",
]


def split_blocks(text):
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def normalize_text(text):
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def looks_like_title(block):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    title = lines[0]
    if len(title) < 2 or len(title) > 100:
        return False
    if title.endswith((".", "!", "?", ":", ";", ",")):
        return False
    return True


def clean_response(text):
    return normalize_text(text).replace("\uFFFD", " ").strip()


def build_examples(corpus_text, max_examples=None):
    blocks = split_blocks(corpus_text)
    current_title = None
    examples = []
    seen = set()

    for block in blocks:
        if looks_like_title(block):
            current_title = normalize_text(block)
            continue

        if not current_title:
            continue

        paragraph = clean_response(block)
        if len(paragraph) < 80:
            continue

        templates = TITLE_PROMPT_TEMPLATES + DETAIL_PROMPT_TEMPLATES
        for template in templates:
            prompt = f"<|user|> {template.format(title=current_title)} <|assistant|>"
            key = (prompt, paragraph)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                {
                    "title": current_title,
                    "prompt": prompt,
                    "response": paragraph,
                }
            )

            if max_examples is not None and len(examples) >= max_examples:
                return examples

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build newfinetune.jsonl from a wiki-style corpus file."
    )
    parser.add_argument("--corpus", default="NewCORPUS.txt")
    parser.add_argument("--output", default="newfinetune.jsonl")
    parser.add_argument("--max_examples", type=int, default=50000)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    examples = build_examples(text, max_examples=args.max_examples)

    with output_path.open("w", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
