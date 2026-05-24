import argparse
from .generate import load_model, sample

def chat_loop(ckpt, temp, top_k, top_p, json_mode, max_new_tokens):
    print(f"Loading model from: {ckpt}")

    model, tokenizer = load_model(ckpt, device="cpu")
    model.eval()

    print("TinyGPT chat started (ctrl+c or ctrl+d to exit).")

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue

            prompt = f"<|user|> {user} <|assistant|>"

            out = sample(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                json_mode=json_mode,
            )

            if out.startswith(prompt):
                out = out[len(prompt):]

            print("Bot:", out.strip())

    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="fine_tuned.pt", help="Path to model checkpoint")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max", type=int, default=256)

    args = parser.parse_args()

    chat_loop(args.ckpt, args.temp, args.top_k, args.top_p, args.json, args.max)
