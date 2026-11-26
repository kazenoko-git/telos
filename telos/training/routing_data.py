"""Small prompt/completion corpora for the matched routing experiment."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ANSWER = 24
PROMPT = 5
VOCAB = 40


@dataclass(frozen=True)
class Sequences:
    ids: np.ndarray
    prompts: np.ndarray
    lengths: np.ndarray

    def __len__(self):
        return len(self.ids)

    def examples(self, indices):
        for index in indices:
            prompt = self.prompts[index]
            yield (self.ids[index, :prompt].tolist(), self.ids[index, prompt:self.lengths[index]].tolist())


@dataclass(frozen=True)
class RoutingData:
    train: Sequences
    heldout: Sequences
    benchmark_indices: np.ndarray
    vocab_size: int
    tokenizer: Tokenizer | None = None
    eos_id: int | None = None


def _packed(rows, width, pad):
    ids = np.full((len(rows), width), pad, np.int32)
    prompts = np.empty(len(rows), np.int32)
    lengths = np.empty(len(rows), np.int32)
    for i, (prefix, answer) in enumerate(rows):
        sequence = prefix + answer
        ids[i, :len(sequence)] = sequence
        prompts[i] = len(prefix)
        lengths[i] = len(sequence)
    return Sequences(ids, prompts, lengths)


def synthetic_corpus():
    rng = np.random.default_rng(2026)
    train, heldout = [], []
    for task in range(4):
        table = rng.permutation(16)
        for seed in range(16):
            for offset in range(16):
                prefix, answer = [37, task+1, seed+5, offset+5, 38], []
                value = (seed + offset) % 16
                for _ in range(ANSWER):
                    answer.append(int(value)+21)
                    value = table[value]
                target = heldout if (seed + 3*offset + task) % 5 == 0 else train
                target.append((prefix, answer))
    pick = np.random.default_rng(2027)
    domains = [[i for i, (prefix, _) in enumerate(heldout) if prefix[1] == task+1] for task in range(4)]
    indices = np.concatenate([pick.choice(rows, 8, replace=False) for rows in domains])
    return RoutingData(_packed(train, PROMPT+ANSWER, 0), _packed(heldout, PROMPT+ANSWER, 0), indices, VOCAB)


def instruction_corpus(path, *, width=96, train_limit=4096, test_limit=512):
    """Read instruction, response and optional context from JSONL."""
    partitions = {False: [], True: []}
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = row["instruction"].strip()
            response = row["response"].strip()
            context = (row.get("context") or "").strip()
            key = " ".join(instruction.casefold().split())
            identity = (key, " ".join(context.casefold().split()))
            if not key or not response or identity in seen:
                continue
            seen.add(identity)
            # Group repeated instructions across contexts into the same split.
            heldout = hashlib.sha256(key.encode()).digest()[0] < 28
            prompt = instruction + ("\nContext: " + context if context else "")
            partitions[heldout].append((prompt, response))
    if not all(partitions.values()):
        raise ValueError("Instruction dataset needs examples in both deterministic splits")

    rng = np.random.default_rng(2026)
    for rows in partitions.values():
        rng.shuffle(rows)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    specials = ("[PAD]", "[BOS]", "[ANSWER]", "[EOS]")
    tokenizer.train_from_iterator(
        (text for row in partitions[False][:8192] for text in row),
        trainers.BpeTrainer(vocab_size=512, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=list(specials), show_progress=False),
    )
    pad, bos, answer_marker, eos = (tokenizer.token_to_id(token) for token in specials)

    def encode(rows, limit):
        chosen = []
        for prompt, response in rows:
            prefix = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids, answer_marker]
            answer = [*tokenizer.encode(response, add_special_tokens=False).ids, eos]
            if len(prefix)+len(answer) <= width:
                chosen.append((prefix, answer))
                if len(chosen) == limit:
                    break
        if not chosen:
            raise ValueError(f"No complete instruction examples fit {width} tokens in one split")
        return _packed(chosen, width, pad)

    train = encode(partitions[False], train_limit)
    heldout = encode(partitions[True], test_limit)
    indices = np.random.default_rng(2027).choice(len(heldout), min(len(heldout), 32), replace=False)
    return RoutingData(train, heldout, indices, tokenizer.get_vocab_size(), tokenizer, eos)
