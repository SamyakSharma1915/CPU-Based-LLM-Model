import argparse
from pathlib import Path


TOPICS = [
    {
        "title": "Artificial Intelligence",
        "summary": "Artificial intelligence is a branch of computer science focused on creating systems that can process information, recognize patterns, and support useful tasks.",
        "details": [
            "AI can help with summarization, search, classification, tutoring, and automation.",
            "Modern AI models learn from examples instead of being told every rule by hand.",
            "AI works best as a tool that extends human ability rather than replacing human judgment.",
        ],
    },
    {
        "title": "Earth",
        "summary": "Earth is the planet where humans and many other forms of life live.",
        "details": [
            "It has air, water, land, and ecosystems that support living organisms.",
            "Earth rotates to create day and night and revolves around the Sun to create the yearly cycle.",
            "Its atmosphere, water, and moderate temperatures make it unique among known planets.",
        ],
    },
    {
        "title": "The Sun",
        "summary": "The Sun is the star at the center of the solar system and provides light and heat to Earth.",
        "details": [
            "Sunlight supports plant growth and many natural processes.",
            "Without the Sun, Earth would be dark, cold, and unable to support life as it does now.",
            "The Sun is essential to climate, seasons, and energy cycles on Earth.",
        ],
    },
    {
        "title": "The Moon",
        "summary": "The Moon is Earth's natural satellite and moves around Earth in a steady orbit.",
        "details": [
            "It reflects sunlight and appears bright in the night sky.",
            "The Moon influences ocean tides through gravitational pull.",
            "It also helps stabilize Earth's rotational behavior over time.",
        ],
    },
    {
        "title": "Programming",
        "summary": "Programming is the process of writing instructions that tell a computer how to perform tasks.",
        "details": [
            "Programmers use languages such as Python, JavaScript, Java, C++, and Go.",
            "Software can automate repetitive work, control systems, and process large amounts of information.",
            "Good programming combines logic, structure, readability, and testing.",
        ],
    },
    {
        "title": "Python",
        "summary": "Python is a popular programming language known for clear syntax and strong library support.",
        "details": [
            "People use Python for automation, web development, data analysis, and machine learning.",
            "Its readable style makes it a common choice for beginners.",
            "It is also powerful enough for professional tools and production systems.",
        ],
    },
    {
        "title": "The Internet",
        "summary": "The internet is a global network that connects computers, services, and people around the world.",
        "details": [
            "It enables websites, messaging, streaming, online learning, and many modern applications.",
            "People use the internet to communicate, search for information, and access digital services.",
            "Reliable internet access affects education, business, and everyday life.",
        ],
    },
    {
        "title": "Science",
        "summary": "Science is a method of understanding the natural world through observation, testing, and explanation.",
        "details": [
            "Fields such as physics, chemistry, and biology explain different parts of the world.",
            "Scientific ideas are tested carefully so people can compare evidence and improve understanding.",
            "Science helps drive progress in medicine, engineering, agriculture, and technology.",
        ],
    },
    {
        "title": "Education",
        "summary": "Education helps people build knowledge, skills, and judgment over time.",
        "details": [
            "Students learn through reading, listening, practice, experimentation, and discussion.",
            "Education supports personal growth, work preparation, and critical thinking.",
            "Strong education encourages curiosity, clarity, and long-term learning habits.",
        ],
    },
    {
        "title": "Language",
        "summary": "Language is a system people use to share ideas, emotions, and information.",
        "details": [
            "People communicate through speaking, writing, reading, and signing.",
            "Language helps preserve culture and pass knowledge between generations.",
            "Clear language is important in teaching, teamwork, documentation, and everyday conversation.",
        ],
    },
    {
        "title": "Communication",
        "summary": "Good communication means expressing ideas clearly and understanding others carefully.",
        "details": [
            "It involves listening, asking useful questions, and choosing clear words.",
            "Strong communication helps in work, friendships, study, and leadership.",
            "Misunderstandings are reduced when people explain directly and respond respectfully.",
        ],
    },
    {
        "title": "Problem Solving",
        "summary": "Problem solving is the process of understanding a challenge and finding a workable answer.",
        "details": [
            "A useful process includes defining the problem, exploring options, testing ideas, and reviewing results.",
            "Large problems become easier when they are split into smaller steps.",
            "Problem solving matters in daily life, engineering, research, and software development.",
        ],
    },
    {
        "title": "Machine Learning",
        "summary": "Machine learning is a branch of AI in which models learn useful patterns from data.",
        "details": [
            "Instead of hard-coding every rule, a model can learn from many examples.",
            "Machine learning is used for recommendations, recognition, prediction, and ranking systems.",
            "Model quality depends on data quality, evaluation, and training design.",
        ],
    },
    {
        "title": "Software Testing",
        "summary": "Software testing checks whether a program behaves correctly under expected and unexpected conditions.",
        "details": [
            "Tests can verify small units, larger workflows, and edge cases.",
            "Good testing reduces regressions and improves confidence in code changes.",
            "Automated tests are especially helpful in larger or rapidly changing projects.",
        ],
    },
    {
        "title": "Databases",
        "summary": "A database is a system used to store, organize, and retrieve information efficiently.",
        "details": [
            "Applications use databases to manage users, products, messages, logs, and many other records.",
            "Common database systems include SQLite, PostgreSQL, MySQL, and MongoDB.",
            "Schema design, indexing, and query patterns all affect performance and reliability.",
        ],
    },
    {
        "title": "History",
        "summary": "History is the study of past events, people, and societies.",
        "details": [
            "It helps people understand how communities, ideas, and institutions changed over time.",
            "Historical study depends on sources, context, and careful interpretation.",
            "Learning history can improve perspective and decision-making in the present.",
        ],
    },
    {
        "title": "Mathematics",
        "summary": "Mathematics is the study of numbers, patterns, structures, and relationships.",
        "details": [
            "It supports science, engineering, finance, computing, and many daily tasks.",
            "Mathematics helps people reason precisely and solve structured problems.",
            "Topics include arithmetic, algebra, geometry, calculus, and statistics.",
        ],
    },
    {
        "title": "Reading",
        "summary": "Reading is the process of understanding written language and extracting meaning from text.",
        "details": [
            "Reading builds vocabulary, knowledge, and concentration over time.",
            "People read for learning, work, entertainment, and reflection.",
            "Strong reading habits make study and communication easier.",
        ],
    },
    {
        "title": "Writing",
        "summary": "Writing is the process of expressing ideas in clear, structured text.",
        "details": [
            "Good writing depends on clarity, organization, and audience awareness.",
            "People write emails, essays, notes, reports, code comments, and documentation.",
            "Practice improves writing more than speed alone.",
        ],
    },
    {
        "title": "Learning",
        "summary": "Learning is a gradual process of building understanding through practice, reflection, and feedback.",
        "details": [
            "People often learn best by asking questions and applying ideas in real situations.",
            "Mistakes are a natural part of learning and often reveal what needs more attention.",
            "Consistent effort matters more than brief bursts of activity.",
        ],
    },
]

EXPLANATION_VARIANTS = [
    "{summary}",
    "{summary} {detail1}",
    "{summary} {detail1} {detail2}",
    "{summary} {detail1} {detail2} {detail3}",
    "In simple words, {summary.lower} {detail1}",
    "A beginner-friendly explanation is this: {summary} {detail2}",
    "A clear explanation of {title} starts with the main idea: {summary} Then it helps to remember that {detail2}",
]

QUESTION_TEMPLATES = [
    "What is {title}?",
    "Explain {title}.",
    "Tell me about {title}.",
    "Give me a simple explanation of {title}.",
    "Why is {title} important?",
    "Help me understand {title}.",
    "Can you teach me about {title}?",
    "Summarize {title} in simple words.",
    "What should I know about {title}?",
    "How would you explain {title} to a beginner?",
    "Give me key facts about {title}.",
    "Write a short note on {title}.",
]

ASSISTANT_STYLE_LINES = [
    "A helpful assistant answers the user's question before adding extra detail.",
    "A clear assistant avoids repeating the prompt and instead provides useful information.",
    "Short questions often deserve direct answers followed by one or two supporting sentences.",
    "A study assistant should prefer simple words when the user is learning a basic topic.",
    "A reliable assistant avoids unnecessary repetition and tries to stay on topic.",
    "When a user asks for an explanation, the assistant should explain the concept, not just repeat keywords.",
]


def render(text, topic):
    mapping = {
        "title": topic["title"],
        "summary": topic["summary"],
        "detail1": topic["details"][0],
        "detail2": topic["details"][1],
        "detail3": topic["details"][2],
        "summary.lower": topic["summary"][0].lower() + topic["summary"][1:],
    }
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_topic_sections(topic):
    blocks = [topic["title"], "", topic["summary"]]
    blocks.extend(topic["details"])
    blocks.append("")

    for variant in EXPLANATION_VARIANTS:
        blocks.append(render(variant, topic))
        blocks.append("")

    blocks.append(f"Study Notes: {topic['title']}")
    blocks.append("")
    blocks.append(f"Definition: {topic['summary']}")
    blocks.append(f"Important point: {topic['details'][0]}")
    blocks.append(f"Important point: {topic['details'][1]}")
    blocks.append(f"Important point: {topic['details'][2]}")
    blocks.append("")
    return "\n".join(blocks)


def build_dialogues(topic):
    answer_short = f"{topic['summary']} {topic['details'][0]}"
    answer_long = f"{topic['summary']} {topic['details'][0]} {topic['details'][1]} {topic['details'][2]}"
    blocks = []
    for idx, prompt in enumerate(QUESTION_TEMPLATES):
        user = prompt.format(title=topic["title"])
        answer = answer_short if idx % 2 == 0 else answer_long
        blocks.append(f"User: {user}\nAssistant: {answer}")
    return "\n\n".join(blocks)


def build_compare_blocks(topics):
    blocks = []
    for i in range(len(topics) - 1):
        a = topics[i]
        b = topics[i + 1]
        blocks.append(
            "\n".join(
                [
                    f"Comparison: {a['title']} and {b['title']}",
                    "",
                    f"{a['title']}: {a['summary']}",
                    f"{b['title']}: {b['summary']}",
                    f"Both topics can be understood clearly when their main purpose and important details are explained step by step.",
                    "",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_corpus():
    sections = ["Expanded Knowledge Corpus", ""]

    for topic in TOPICS:
        sections.append(build_topic_sections(topic))

    sections.append("Conversation Practice")
    sections.append("")
    for topic in TOPICS:
        sections.append(build_dialogues(topic))
        sections.append("")

    sections.append("Comparison Notes")
    sections.append("")
    sections.append(build_compare_blocks(TOPICS))
    sections.append("")

    sections.append("Helpful Assistant Style")
    sections.append("")
    sections.extend(ASSISTANT_STYLE_LINES)
    sections.append("")

    return "\n".join(sections).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build an expanded clean corpus for pretraining")
    parser.add_argument("--output", default="data/corpus.txt")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    output_path.write_text(corpus, encoding="utf-8")
    print(f"Wrote corpus to {output_path} ({len(corpus)} chars)")


if __name__ == "__main__":
    main()
