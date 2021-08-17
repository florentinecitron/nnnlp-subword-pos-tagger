import copy
import csv
import sys
csv.field_size_limit(sys.maxsize)
import itertools as it
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import os.path
from typing import Optional

from polyglot.mapping import Embedding
import torch


def iter_file(f):
    headers = [
        "idx",
        "form",
        "lemma",
        "udpos",
        "xpos",
        "ignore1",
        "ignore2",
        "ignore3",
        "ignore4",
        "ignore6",
    ]

    reader = csv.DictReader(f, delimiter="\t", fieldnames=headers, quoting=csv.QUOTE_NONE)

    s = None
    for line in reader:
        if line["idx"][0] == "#":
            continue
        if line["idx"] == "1":
            if s:
                yield s
            s = [line]
        else:
            s.append(line)
    yield s


class Vocab:
    def __init__(self, add_special=[]):
        self.w2i = defaultdict(it.count(0).__next__)
        self.i2w = {}
        self.counts = Counter()
        for el in add_special:
            self.add(el)

    def add(self, w):
        i = self.w2i[w]
        self.i2w[i] = w
        self.counts[w] += 1

    def is_singleton(self, widx):
        return self.counts[self.i2w[widx]] <= 1

    def add_random_unk(self, example, p):
        rval = copy.deepcopy(example)
        rval["tokens"] = [
            #self.w2i["<unk>"] if self.is_singleton(t_idx) and random.random() < p else t_idx
            self.w2i["<unk>"] if self.drop(t_idx, 0.25) else t_idx
            for t_idx
            in example["tokens"]
        ]
        rval["unk_indices"] = [idx == self.w2i["<unk>"] for idx in rval["tokens"]]
        return rval

    def drop(self, widx, dropout_rate):
        count = self.counts[self.i2w[widx]]
        return random.random() > (count/(dropout_rate+count))

    def get(self, w):
        if w in self.w2i:
            return self.w2i[w]
        elif isinstance(w, str) and w.lower() in self.w2i:
            return self.w2i[w.lower()]
        return self.w2i["<unk>"]

    def __len__(self):
        return len(self.w2i)

    def convert_embedding(self, embedding):
        embedding_word_id = embedding.vocabulary.word_id

        for w in set(embedding.vocabulary.word_id) - set(self.w2i.keys()):
            self.add(w)
        vocab_embedding_indices = [
            (idx, embedding_word_id[word if word in embedding_word_id else word.lower()])
            for word, idx in self.w2i.items()
            if word in embedding_word_id or word.lower() in embedding_word_id
        ]
        vocab_idxs, embeddings_indices = zip(*vocab_embedding_indices)
        rval = torch.zeros((len(self), embedding.vectors.shape[1]))
        rval[vocab_idxs, :] = torch.tensor(
            embedding.vectors[list(embeddings_indices), :],
            dtype=torch.float
        )
        return rval


lang_to_ud_lang = {
    "de": "UD_German",
    "en": "UD_English",
    "id": "UD_Indonesian",
    "fi": "UD_Finnish"
}


@dataclass
class Data:
    v_char: Vocab
    v_word: Vocab
    v_bytes: Vocab
    v_label: Vocab
    v_freq_bin: Vocab
    embeddings: Optional[torch.tensor]
    train: list
    val: list
    test: list


@dataclass
class DataRef:
    lang: str
    ud_lang_id: str
    ud_lang: str
    base_path: str
    base_embedding_path: str

    @classmethod
    def from_lang_id(
        cls,
        lang,
        base_path="data",
        base_embeddings_path=os.path.expanduser("~/polyglot_data/embeddings2/"),
    ) -> "DataRef":
        ud_lang_id = lang_to_ud_lang[lang]
        to_search = Path(base_path) / "universal-dependencies-1.2" / ud_lang_id
        ud_lang = next(to_search.glob("*.conllu")).name.rsplit(".", 1)[0].split("-")[0]
        return cls(lang, ud_lang_id, ud_lang, base_path, base_embeddings_path)

    def train_file(self):
        return f"{self.base_path}/universal-dependencies-1.2/{self.ud_lang_id}/{self.ud_lang}-ud-train.conllu"

    def dev_file(self):
        return f"{self.base_path}/universal-dependencies-1.2/{self.ud_lang_id}/{self.ud_lang}-ud-dev.conllu"

    def test_file(self):
        return f"{self.base_path}/universal-dependencies-1.2/{self.ud_lang_id}/{self.ud_lang}-ud-test.conllu"

    def get_freq_bins(self):
        with open(self.train_file()) as i:
            tokens = it.chain.from_iterable(
                [t["form"] for t in s] for s in iter_file(i)
            )
            c = Counter(tokens)
        return {t: int(math.log(v)) for t, v in c.items()}

    def get_embeddings(self):
        return Embedding.load(
            f"{self.base_embedding_path}/{self.lang}/embeddings_pkl.tar.bz2"
        )

    def get_vocab(self):
        freq_bins = self.get_freq_bins()
        v_word = Vocab(add_special=["<unk>"])
        # padding index will be 0
        v_char = Vocab(add_special=["<pad>", "<s>", "</s>", "<unk>"])
        v_bytes = Vocab(add_special=["<pad>", "<s>", "</s>", "<unk>"])
        v_label = Vocab()
        v_freq_bin = Vocab()

        for v in freq_bins.values():
            v_freq_bin.add(v)

        with open(self.train_file()) as i:
            for s in iter_file(i):
                for t in s:
                    v_word.add(t["form"])
                    for c in t["form"]:
                        v_char.add(c)
                    for c in t["form"].encode("utf-8"):
                        v_bytes.add(c)
                    v_label.add(t["udpos"])
        return v_char, v_word, v_bytes, v_label, freq_bins, v_freq_bin

    def get_data(self, with_embeddings=False) -> Data:
        v_char, v_word, v_bytes, v_label, w_freq_bins, v_freq_bin = self.get_vocab()
        splits = []
        for f, is_train in zip(
            [self.train_file(), self.dev_file(), self.test_file()], [True, False, False]
        ):
            with open(f) as i:
                split = []
                for s in iter_file(i):
                    tokens_indices = (
                        [v_word.get(t["form"]) for t in s]
                    )
                    char_indices = (
                        [
                            [v_char.get("<s>")]
                            + [v_char.get(c) for c in t["form"]]
                            + [v_char.get("</s>")]
                            for t in s
                        ]
                    )
                    bytes_indices = (
                        [
                            [v_bytes.get("<s>")]
                            + [v_bytes.get(c) for c in t["form"].encode("utf-8")]
                            + [v_bytes.get("</s>")]
                            for t in s
                        ]
                    )
                    label_indices = (
                        [v_label.get(t["udpos"]) for t in s]
                    )

                    example = {
                        "padded_forms": (
                            [t["form"] for t in s]
                        ),
                        "tokens": tokens_indices,
                        "chars": char_indices,
                        "bytes": bytes_indices,
                        "labels": label_indices,
                    }
                    if is_train:
                        freq_bins = (
                            [v_freq_bin.get(w_freq_bins[t["form"]]) for t in s]
                        )
                        example["freq_bins"] = freq_bins
                    else:
                        example["unk_indices"] = [idx == v_word.w2i["<unk>"] for idx in example["tokens"]]

                    split.append(example)
            splits.append(split)

        embeddings = v_word.convert_embedding(self.get_embeddings()) if with_embeddings else None
        return Data(v_char, v_word, v_bytes, v_label, v_freq_bin, embeddings, *splits)