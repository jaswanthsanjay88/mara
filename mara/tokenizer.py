from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 8192


def train_tokenizer(texts, out_path: str) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(out_path)
    return load_tokenizer(out_path)


def load_tokenizer(path: str) -> PreTrainedTokenizerFast:
    hf = PreTrainedTokenizerFast(
        tokenizer_file=path,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    return hf
