"""Byte-level BPE tokenizer trained from scratch on the filtered corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "</system>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<plan>",
    "</plan>",
    "<code>",
    "</code>",
    "<tests>",
    "</tests>",
    "<answer>",
    "</answer>",
]

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SYSTEM_TOKEN = "<system>"
SYSTEM_END_TOKEN = "</system>"
USER_TOKEN = "<user>"
USER_END_TOKEN = "</user>"
ASSISTANT_TOKEN = "<assistant>"
ASSISTANT_END_TOKEN = "</assistant>"
PLAN_TOKEN = "<plan>"
PLAN_END_TOKEN = "</plan>"
CODE_TOKEN = "<code>"
CODE_END_TOKEN = "</code>"
TESTS_TOKEN = "<tests>"
TESTS_END_TOKEN = "</tests>"
ANSWER_TOKEN = "<answer>"
ANSWER_END_TOKEN = "</answer>"

DEFAULT_SYSTEM_PROMPT = (
    "You are CodePlan-LM, a private coding and planning assistant. "
    "Answer only in German or English. Focus on software engineering, code, "
    "debugging, architecture, tests, and project planning. If useful, structure "
    "the answer with <plan>, <code>, <tests>, and <answer> sections. "
    "Do not claim to be a large pretrained model."
)


class CodePlanTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = tokenizer.token_to_id(UNK_TOKEN)
        self.bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
        self.assistant_end_id = tokenizer.token_to_id(ASSISTANT_END_TOKEN)
        if self.pad_id is None or self.eos_id is None:
            raise ValueError("Tokenizer is missing required special tokens")

    @classmethod
    def load(cls, path: str | Path) -> "CodePlanTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        """Encode many texts at once; the backend parallelises across cores."""
        encodings = self.tokenizer.encode_batch(texts)
        results = []
        for enc in encodings:
            ids = list(enc.ids)
            if add_eos:
                ids.append(self.eos_id)
            results.append(ids)
        return results

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def format_chat(
        self,
        user: str,
        system: str = DEFAULT_SYSTEM_PROMPT,
        assistant_prefix: str = "",
    ) -> str:
        return (
            f"{BOS_TOKEN}{SYSTEM_TOKEN}{system}{SYSTEM_END_TOKEN}"
            f"{USER_TOKEN}{user}{USER_END_TOKEN}"
            f"{ASSISTANT_TOKEN}{assistant_prefix}"
        )

    def format_sft(self, system: str, user: str, assistant: str) -> tuple[list[int], list[int]]:
        """Return (input_ids, labels). Labels are -100 outside assistant content."""
        prompt = (
            f"{BOS_TOKEN}{SYSTEM_TOKEN}{system}{SYSTEM_END_TOKEN}"
            f"{USER_TOKEN}{user}{USER_END_TOKEN}{ASSISTANT_TOKEN}"
        )
        prompt_ids = self.encode(prompt)
        assistant_ids = self.encode(assistant, add_eos=True)
        input_ids = prompt_ids + assistant_ids
        labels = [-100] * len(prompt_ids) + assistant_ids
        return input_ids, labels


MIN_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)  # byte-level alphabet + special tokens


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    out_path: str | Path = "artifacts/tokenizer.json",
) -> CodePlanTokenizer:
    """Train a byte-level BPE tokenizer.

    ``vocab_size`` must be at least :data:`MIN_VOCAB_SIZE` (256 byte tokens plus
    the 18 special tokens), otherwise the trainer cannot honour the request.
    """
    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size={vocab_size} is impossible for byte-level BPE: "
            f"at least {MIN_VOCAB_SIZE} tokens are required "
            f"(256 bytes + {len(SPECIAL_TOKENS)} special tokens)"
        )
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    wrapped = CodePlanTokenizer(tokenizer)
    wrapped.save(out_path)
    return wrapped
