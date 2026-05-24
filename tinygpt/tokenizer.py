# tinygpt/tokenizer.py
import os
import sentencepiece as spm

def train_tokenizer(
    corpus_path="data/corpus.txt",
    model_prefix="tokenizer/spm",
    vocab_size=8000,
):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols="<|system|>,<|user|>,<|assistant|>",
        character_coverage=1.0,
    )
    return f"{model_prefix}.model"

class Tokenizer:
    def __init__(self, model_path="tokenizer/spm.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def encode_chat(self, prompt, response):
        prompt_ids = self.encode(prompt, add_bos=True, add_eos=True)
        response_ids = self.encode(response, add_bos=False, add_eos=True)
        input_ids = prompt_ids + response_ids
        target_ids = list(input_ids)
        for i in range(len(prompt_ids)):
            target_ids[i] = -1
        return input_ids, target_ids

if __name__ == "__main__":
    import argparse
    from .config import ModelConfig

    parser = argparse.ArgumentParser(description="Train tokenizer and write spm.model")
    parser.add_argument("--corpus", default="data/corpus.txt")
    parser.add_argument("--model_prefix", default="tokenizer/spm")
    parser.add_argument("--vocab_size", type=int, default=ModelConfig.vocab_size)
    args = parser.parse_args()
    train_tokenizer(args.corpus, args.model_prefix, args.vocab_size)
    print(f"Saved tokenizer to {args.model_prefix}.model")
