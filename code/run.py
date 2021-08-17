import copy
import csv
import itertools as it
import random
import sys
from collections import Counter
from typing import Dict, List, Optional

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import DataRef
from model import PosTagger
from util import RollingAverage


def train(
    lang_id: str,
    n_epochs: int = 20,
    patience: Optional[int] = None,
    use_pretrained_embeddings: bool = False,
    use_word: bool = True,
    use_char: bool = False,
    use_byte: bool = True,
    use_freq_bin: bool = True,
    unknown_probability: float = 0.2,
    show_progress_bar: bool = False,
    train_use_n_examples: Optional[int] = None
):
    d = DataRef.from_lang_id(lang_id, base_path="data")
    data = d.get_data(with_embeddings=use_pretrained_embeddings)
    net = PosTagger(
        n_subtoken_embeddings=len(data.v_char),
        subtoken_embedding_dim=100,
        n_byte_embeddings=len(data.v_bytes),
        byte_embedding_dim=100,
        n_word_embeddings=len(data.v_word),
        word_embedding_dim=64, # accidentally used 64 as dimension
        hidden_size=100,
        n_out=len(data.v_label),
        n_freq_bins=len(data.v_freq_bin),
        noise_sd=0.2,
        use_word=use_word,
        use_char=use_char,
        use_byte=use_byte,
        pretrained_embedding=data.embeddings if use_pretrained_embeddings else None,
    )

    # X = example = {
    #    'tokens': [3, 4, 5, 6, 7, 8, 9],
    #    'chars': [[3, 4, 5, 6], [7, 8, 9], [10, 11], [12, 5, 6, 9, 13], [7, 8, 14, 13], [13, 7, 5, 4, 15], [16]],
    #    'labels': [0, 1, 2, 3, 1, 4, 5]
    # }

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion_label = nn.CrossEntropyLoss()
    criterion_freq_bin = nn.CrossEntropyLoss()

    pbar_epoch = tqdm(
        total=n_epochs, desc="Epochs", leave=False, disable=not show_progress_bar
    )

    best_model, best_val_acc, best_model_seen = None, float("-inf"), 0
    train_data = list(it.islice(data.train, 0, train_use_n_examples))
    pbar_example = tqdm(
        total=len(train_data),
        desc="Training examples",
        leave=False,
        disable=not show_progress_bar,
    )



    for _ in range(n_epochs):
        random.shuffle(train_data)
        # net = copy.deepcopy(best_model)
        #net.train()
        loss_average = RollingAverage.make(1000)
        for ex in train_data: #it.islice(data.train, 0, 200):
            optimizer.zero_grad()
            #ex_with_unk = ex
            ex_with_unk = data.v_word.add_random_unk(ex, unknown_probability)
            if use_freq_bin:
                outputs_label, outputs_freq_bin = net(
                    ex_with_unk, return_freq_bins=True, train=True
                )
            else:
                outputs_label = net(ex_with_unk, return_freq_bins=False, train=True)

            #print(outputs_label.tolist())
            #print("correct", ex["labels"])
            #preds = outputs_label.argmax(dim=2).squeeze().tolist()
            #print("pred", preds)

            loss = None
            for idx, (label, freq_bin) in enumerate(zip(ex["labels"], ex["freq_bins"])):
                loss_label = criterion_label(
                    outputs_label[idx, :],
                    torch.tensor([label], dtype=torch.long),
                )
                #print(loss_label.item())
                if loss is None:
                    loss = loss_label
                else:
                    loss += loss_label
                if use_freq_bin:
                    loss_freq_bin = criterion_freq_bin(
                        outputs_freq_bin[idx, :],
                        torch.tensor([freq_bin], dtype=torch.long),
                    )
                    loss += loss_freq_bin
            loss_average.add_stat(loss.item())
            loss.backward()
            optimizer.step()
            pbar_example.update(1)
            pbar_example.set_description(
                f"Training examples (loss: {round(loss_average.get_val(), 2)})"
            )

        #net.eval()
        with torch.no_grad():
            acc, total = 0, 0
            n_correct_oov, total_oov = 0, 0
            #incorrect_counts = Counter()

            for ex in data.val:
                outputs = net(ex, return_freq_bins=False, train=False)
                preds = outputs.argmax(dim=2).squeeze().tolist()
                #print(preds)
                if not isinstance(preds, list):
                    preds = [preds]
                for pred, correct, form in list(
                    zip(preds, ex["labels"], ex["padded_forms"])
                ):
                    total += 1
                    is_correct = int(pred == correct)
                    acc += is_correct
                    #incorrect_counts[(pred, correct)] += 1
                    if form is not None and form not in data.v_word.w2i:
                        total_oov += 1
                        n_correct_oov += is_correct

            accuracy = acc / total
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                best_model = copy.deepcopy(net)
                best_model_seen = 0
            else:
                best_model_seen += 1
            accuracy_oov = n_correct_oov / total_oov

            if patience is not None and best_model_seen > patience:
                break

        pbar_epoch.set_description(
            f"Epochs (accuracy: {round(accuracy, 4)}, accuracy_oov: {round(accuracy_oov, 4)})"
        )
        pbar_epoch.update(1)
        pbar_example.reset()

    return data, best_model


def evaluate(data, net):
    net.eval()
    n_correct, total = 0, 0
    n_correct_oov, total_oov = 0, 0
    for ex in data.test:
        outputs = net(ex, return_freq_bins=False, train=False)
        preds = outputs.argmax(dim=2).squeeze().tolist()
        if not isinstance(preds, list):
            preds = [preds]
        for pred_label, correct_label, form in list(
            zip(preds, ex["labels"], ex["padded_forms"])
        ):
            total += 1
            is_correct = int(pred_label == correct_label)
            n_correct += is_correct
            if form is not None and form not in data.v_word.w2i:
                total_oov += 1
                n_correct_oov += is_correct
    accuracy = n_correct / total
    accuracy_oov = n_correct_oov / total_oov
    return {"accuracy": accuracy, "accuracy_oov": accuracy_oov}


def _run_experiment(experiments: dict, langs: List[str], out_file: str):
    show_progress_bar = True

    pbar_lang = tqdm(total=len(langs), desc="Languages", leave=True, disable=not show_progress_bar)

    with open(out_file, "a") as o:
        writer = csv.DictWriter(
            o, fieldnames=["lang", "experiment", "accuracy", "accuracy_oov"]
        )

        try:
            for lang in langs:
                pbar_lang.set_description(f"Language {lang}")

                pbar_experiment = tqdm(
                    total=len(experiments),
                    desc="Experiments",
                    leave=False,
                    disable=not show_progress_bar,
                )

                for experiment_name, experiment_params in experiments.items():

                    pbar_experiment.set_description(f"Experiment {experiment_name}")

                    data, net = train(lang_id=lang, show_progress_bar=show_progress_bar, **experiment_params)
                    results = evaluate(data, net)

                    writer.writerow(
                        {"lang": lang, "experiment": experiment_name, **results}
                    )
                    o.flush()

                    pbar_experiment.update(1)

                pbar_lang.update(1)

        except KeyboardInterrupt:
            pass


def run_experiment(out_file: str):
    langs = ["en", "id", "de", "fi"]

    experiments = {
        "w": {
            "use_pretrained_embeddings": False,
            "use_word": True,
            "use_char": False,
            "use_byte": False,
            "use_freq_bin": False,
        },
        "c": {
            "use_pretrained_embeddings": False,
            "use_word": False,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False,
        },
        "c+b": {
            "use_pretrained_embeddings": False,
            "use_word": False,
            "use_char": True,
            "use_byte": True,
            "use_freq_bin": False,
        },
        "w+c": {
            "use_pretrained_embeddings": False,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False,
        },
        "w+c+POLYGLOT": {
            "use_pretrained_embeddings": True,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False,
        },
        "w+c+POLYGLOT+FreqBin": {
            "use_pretrained_embeddings": True,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": True,
        },
    }
    _run_experiment(experiments, langs, out_file)


def run_experiments_data_size(out_file: str):
    langs = ["de", "en", "fi", "id"]
    experiments = {
        f"n_{n}": {
            "use_pretrained_embeddings": True,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False,
            "train_use_n_examples": n,
            "n_epochs": 10
        } for n in [100, 200, 400, 800, 1600]
    }
    _run_experiment(experiments, langs, out_file)


if __name__ == "__main__":
    fire.Fire()
