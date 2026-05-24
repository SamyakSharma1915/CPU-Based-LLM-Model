import argparse
import json
from pathlib import Path


TITLE_PROMPT_TEMPLATES = [
    "Tell me about {title}.",
    "Explain {title}.",
    "What is {title}?",
    "Give me information about {title}.",
    "Can you describe {title}?",
    "Write a short note on {title}.",
    "Summarize {title}.",
    "What should I know about {title}?",
    "Teach me about {title} in simple words.",
    "Give me a quick explanation of {title}.",
    "What are some facts about {title}?",
    "Help me understand {title}.",
    "Could you explain {title} clearly?",
    "Give an overview of {title}.",
    "Who is this {title}?"
]

PARAGRAPH_PROMPT_TEMPLATES = [
    "Answer this prompt about {title}: {hint}",
    "If I ask about {title}, what would you say?",
    "Give a concise explanation of {title}.",
    "Explain this topic to a beginner: {title}.",
    "What is important about {title}?",
    "Can you answer a question on {title}?",
]

GENERIC_PROMPT_TEMPLATES = [
    "Explain this clearly.",
    "Teach me about this topic.",
    "Summarize this in simple words.",
    "What does this mean?",
    "Can you explain this to a beginner?",
    "Give me a short explanation.",
]

CHAT_PROMPT_TEMPLATES = [
    "{user}",
    "Please answer this: {user}",
    "Can you help with this question: {user}",
    "I want to know: {user}",
    "Explain this simply: {user}",
    "Give me a short answer: {user}",
    "Answer clearly: {user}",
    "Could you respond to this: {user}",
    "Help me understand this: {user}",
    "What would you say to this: {user}",
    "Respond in simple words: {user}",
    "Give a helpful reply to: {user}",
    "As a helpful assistant, answer: {user}",
    "Please explain: {user}",
    "Provide a beginner-friendly answer: {user}",
    "Answer this question directly: {user}",
]

SKIP_TITLES = {
    "Simple Knowledge Corpus",
    "Conversation Practice",
    "Study Notes Collection",
    "Comparison Notes",
    "Helpful Assistant Style",
    "Expanded Knowledge Corpus",
}


def looks_like_title(text):
    if "\n" in text:
        return False
    text = text.strip()
    if not text:
        return False
    if len(text) > 80:
        return False
    if text.endswith((".", "!", "?", ":", ";")):
        return False
    return True


def split_blocks(text):
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def first_sentence(text, limit=90):
    sentence = text.split(". ", 1)[0].strip()
    if len(sentence) > limit:
        sentence = sentence[: limit - 3].rstrip() + "..."
    return sentence.rstrip(".!?")


def normalize_paragraph(block):
    return " ".join(line.strip() for line in block.splitlines() if line.strip())


def should_skip_block(title, paragraph):
    if title in SKIP_TITLES:
        return True
    lower = paragraph.lower()
    if lower.startswith("comparison:"):
        return True
    if lower.startswith("study notes:"):
        return True
    if "helpful assistant" in lower and "user" not in lower and "assistant:" not in lower:
        return True
    if paragraph.startswith("Definition: ") or paragraph.startswith("Important point: "):
        return True
    return False


def infer_title(paragraph):
    if paragraph.startswith("User:") and "Assistant:" in paragraph:
        user_part = paragraph.split("Assistant:", 1)[0].replace("User:", "").strip()
        cleaned = user_part.strip(" ?!.,")
        if cleaned:
            return cleaned.title()
        return "Conversation"

    first_sentence = paragraph.split(". ", 1)[0].strip()
    if not first_sentence:
        return "General Topic"

    lower = first_sentence.lower()
    patterns = [
        ("an artificial intelligence system is", "Artificial Intelligence System"),
        ("artificial intelligence can", "Artificial Intelligence"),
        ("artificial intelligence systems are", "Artificial Intelligence Systems"),
        ("earth is", "Earth"),
        ("the sun is", "The Sun"),
        ("the moon is", "The Moon"),
        ("language is", "Language"),
        ("a computer program is", "Computer Program"),
        ("human beings use", "Human Reasoning"),
        ("education is", "Education"),
        ("science is", "Science"),
        ("technology refers to", "Technology"),
        ("the internet is", "The Internet"),
        ("good communication involves", "Good Communication"),
        ("problem-solving is", "Problem-Solving"),
        ("learning requires", "Learning"),
    ]
    for prefix, title in patterns:
        if lower.startswith(prefix):
            return title

    words = [word.strip(",;:()[]{}") for word in first_sentence.split()]
    words = [word for word in words if word]
    if not words:
        return "General Topic"

    title_words = words[:4]
    return " ".join(word.capitalize() for word in title_words)


def build_examples(corpus_text, max_examples=None):
    blocks = split_blocks(corpus_text)
    current_title = None
    examples = []

    for block in blocks:
        if looks_like_title(block):
            current_title = block
            continue

        paragraph = normalize_paragraph(block)
        if len(paragraph) < 40:
            continue
        if should_skip_block(current_title, paragraph):
            continue

        if paragraph.startswith("User:") and "Assistant:" in paragraph:
            user_text, assistant_text = paragraph.split("Assistant:", 1)
            user_text = user_text.replace("User:", "").strip()
            assistant_text = assistant_text.strip()
            title = infer_title(paragraph)
            for template in CHAT_PROMPT_TEMPLATES:
                prompt_text = template.format(user=user_text)
                prompt = f"<|user|> {prompt_text} <|assistant|>"
                examples.append({"title": title, "prompt": prompt, "response": assistant_text})

                if max_examples is not None and len(examples) >= max_examples:
                    return examples
            continue

        if not current_title:
            generic_template = GENERIC_PROMPT_TEMPLATES[len(examples) % len(GENERIC_PROMPT_TEMPLATES)]
            generic_prompt = f"<|user|> {generic_template} <|assistant|>"
            fallback_title = infer_title(paragraph)
            examples.append({"title": fallback_title, "prompt": generic_prompt, "response": paragraph})

            if max_examples is not None and len(examples) >= max_examples:
                return examples

            hint = first_sentence(paragraph)
            para_template = PARAGRAPH_PROMPT_TEMPLATES[len(examples) % len(PARAGRAPH_PROMPT_TEMPLATES)]
            para_prompt = f"<|user|> {para_template.format(title=fallback_title, hint=hint)} <|assistant|>"
            examples.append({"title": fallback_title, "prompt": para_prompt, "response": paragraph})

            if max_examples is not None and len(examples) >= max_examples:
                return examples
            continue

        title_template = TITLE_PROMPT_TEMPLATES[len(examples) % len(TITLE_PROMPT_TEMPLATES)]
        title_prompt = f"<|user|> {title_template.format(title=current_title)} <|assistant|>"
        examples.append({"title": current_title, "prompt": title_prompt, "response": paragraph})

        if max_examples is not None and len(examples) >= max_examples:
            return examples

        hint = first_sentence(paragraph)
        para_template = PARAGRAPH_PROMPT_TEMPLATES[len(examples) % len(PARAGRAPH_PROMPT_TEMPLATES)]
        para_prompt = f"<|user|> {para_template.format(title=current_title, hint=hint)} <|assistant|>"
        examples.append({"title": current_title, "prompt": para_prompt, "response": paragraph})

        if max_examples is not None and len(examples) >= max_examples:
            return examples

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build fine_tune.jsonl from corpus.txt")
    parser.add_argument("--corpus", default="data/corpus.txt")
    parser.add_argument("--output", default="data/fine_tune.jsonl")
    parser.add_argument("--max_examples", type=int, default=800)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    examples = build_examples(text, max_examples=args.max_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
