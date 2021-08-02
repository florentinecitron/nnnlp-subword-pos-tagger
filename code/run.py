import random

import fire
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import Data
from model import PosTagger
from util import RollingAverage


def train(lang_id: str, n_epochs: int = 2):
    d = Data.from_lang_id(lang_id, base_path="data")
    (v_char, v_word, v_bytes, v_label, v_freq_bins), transition_matrix, (train, val, test) = d.get_data()

    net = self = PosTagger(
        n_subtoken_embeddings=len(v_char),
        subtoken_embedding_dim=100,
        n_byte_embeddings=len(v_bytes),
        byte_embedding_dim=100,
        n_word_embeddings=len(v_word),
        word_embedding_dim=128,
        hidden_size=100,
        n_out=len(v_label),
        n_freq_bins=len(v_freq_bins),
        noise_sd=.2,
        use_word=True,
        use_char=False,
        use_byte=True
    )

    # X = example = {
    #    'tokens': [3, 4, 5, 6, 7, 8, 9],
    #    'chars': [[3, 4, 5, 6], [7, 8, 9], [10, 11], [12, 5, 6, 9, 13], [7, 8, 14, 13], [13, 7, 5, 4, 15], [16]],
    #    'labels': [0, 1, 2, 3, 1, 4, 5]
    # }

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion_label = nn.CrossEntropyLoss()
    criterion_freq_bin = nn.CrossEntropyLoss()

    pbar_epoch = tqdm(total=n_epochs, position=0, desc="Epochs")

    with pbar_epoch:

        pbar_example = tqdm(total=len(train), position=1, desc="Training examples")
        with pbar_example:
            for epoch in range(n_epochs):
                random.shuffle(train)
                net.train()
                loss_average = RollingAverage.make(100)

                for ex in train:
                    optimizer.zero_grad()
                    outputs_label, outputs_freq_bin = net(ex, return_freq_bins=True)
                    loss = torch.tensor(0.0)
                    for idx, (label, freq_bin) in enumerate(zip(ex["labels"], ex["freq_bins"])):
                        loss_label = criterion_label(
                            outputs_label[:, idx, :].squeeze()[None, :],
                            torch.tensor([label], dtype=torch.long),
                        )
                        loss_freq_bin = criterion_freq_bin(
                            outputs_freq_bin[:, idx, :].squeeze()[None, :],
                            torch.tensor([freq_bin], dtype=torch.long),
                        )
                        loss += loss_label + loss_freq_bin
                    loss_average.add_stat(loss.item())
                    loss.backward()
                    optimizer.step()
                    pbar_example.update(1)
                    pbar_example.set_description(
                        f"Training examples (loss: {round(loss_average.get_val(), 2)})"
                    )
                pbar_example.reset()

                net.eval()
                acc, total = 0, 0
                for ex in val:
                    outputs = net(ex, return_freq_bins=False)
                    preds = outputs.argmax(axis=2).squeeze().tolist()
                    if not isinstance(preds, list):
                        preds = [preds]
                    for pred, correct in zip(preds, ex["labels"]):
                        total += 1
                        acc += int(pred == correct)
                accuracy = acc / total
                pbar_epoch.set_description(f"Epochs (accuracy: {round(accuracy, 4)})")
                pbar_epoch.update(1)

def eval(
    lang_id,
):
    pass


if __name__ == "__main__":
    fire.Fire()
