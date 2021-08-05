import copy
import csv
import random

import fire
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from data import DataRef
from model import PosTagger
from util import RollingAverage


def train(
    lang_id: str,
    n_epochs: int = 20,
    use_pretrained_embeddings: bool = False,
    use_word: bool = True,
    use_char: bool = False,
    use_byte: bool = True,
    use_freq_bin: bool = True,
    pbar_start_pos: int = 0
):
    d = DataRef.from_lang_id(lang_id, base_path="data")
    data = d.get_data()

    net = PosTagger(
        n_subtoken_embeddings=len(data.v_char),
        subtoken_embedding_dim=100,
        n_byte_embeddings=len(data.v_bytes),
        byte_embedding_dim=100,
        n_word_embeddings=len(data.v_word),
        word_embedding_dim=128,
        hidden_size=100,
        n_out=len(data.v_label),
        n_freq_bins=len(data.v_freq_bin),
        noise_sd=.2,
        use_word=use_word,
        use_char=use_char,
        use_byte=use_byte,
        pretrained_embedding=data.embeddings if use_pretrained_embeddings else None
    )

    # X = example = {
    #    'tokens': [3, 4, 5, 6, 7, 8, 9],
    #    'chars': [[3, 4, 5, 6], [7, 8, 9], [10, 11], [12, 5, 6, 9, 13], [7, 8, 14, 13], [13, 7, 5, 4, 15], [16]],
    #    'labels': [0, 1, 2, 3, 1, 4, 5]
    # }

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion_label = nn.CrossEntropyLoss()
    criterion_freq_bin = nn.CrossEntropyLoss()

    pbar_epoch = tqdm(total=n_epochs, desc="Epochs", leave=False)

    best_model, best_val_acc = None, float("-inf")

    pbar_example = tqdm(total=len(data.train), desc="Training examples", leave=False)
    for epoch in range(n_epochs):
        random.shuffle(data.train)
        net.train()
        loss_average = RollingAverage.make(100)

        for ex in data.train:
            optimizer.zero_grad()
            outputs_label, outputs_freq_bin = net(ex, return_freq_bins=True)
            loss = torch.tensor(0.0)
            for idx, (label, freq_bin) in enumerate(zip(ex["labels"], ex["freq_bins"])):
                loss_label = criterion_label(
                    outputs_label[:, idx, :].squeeze()[None, :],
                    torch.tensor([label], dtype=torch.long),
                )
                loss += loss_label
                if use_freq_bin:
                    loss_freq_bin = criterion_freq_bin(
                        outputs_freq_bin[:, idx, :].squeeze()[None, :],
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

        net.eval()
        acc, total = 0, 0
        for ex in data.val:
            outputs = net(ex, return_freq_bins=False)
            preds = outputs.argmax(axis=2).squeeze().tolist()
            if not isinstance(preds, list):
                preds = [preds]
            for pred, correct in zip(preds, ex["labels"]):
                total += 1
                acc += int(pred == correct)
        accuracy = acc / total
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            best_model = copy.deepcopy(net)

        pbar_epoch.set_description(f"Epochs (accuracy: {round(accuracy, 4)})")
        pbar_epoch.update(1)

    return data, best_model


def evaluate(data, net):
    net.eval()
    n_correct, total = 0, 0
    n_correct_oov, total_oov = 0, 0
    for ex in data.test:
        outputs = net(ex, return_freq_bins=False)
        preds = outputs.argmax(axis=2).squeeze().tolist()
        if not isinstance(preds, list):
            preds = [preds]
        for pred_label, correct_label, form in zip(preds, ex["labels"], ex["padded_forms"]):
            total += 1
            is_correct = int(pred_label == correct_label)
            n_correct += is_correct
            if form is not None and form not in data.v_word.w2i:
                total_oov += 1
                n_correct_oov += is_correct
    accuracy = n_correct / total
    accuracy_oov = n_correct_oov / total_oov
    return {"accuracy": accuracy, "accuracy_oov": accuracy_oov}


def run_experiment(out_file: str):
    langs = ["en", "de", "id"]

    experiments = {
        "w": {
            "use_pretrained_embeddings": False,
            "use_word": True,
            "use_char": False,
            "use_byte": False,
            "use_freq_bin": False
        },
        "c": {
            "use_pretrained_embeddings": False,
            "use_word": False,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False
        },
        "c+b": {
            "use_pretrained_embeddings": False,
            "use_word": False,
            "use_char": True,
            "use_byte": True,
            "use_freq_bin": False
        },
        "w+c": {
            "use_pretrained_embeddings": False,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False
        },
        "w+c+POLYGLOT": {
            "use_pretrained_embeddings": True,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": False
        },
        "w+c+POLYGLOT+FreqBin": {
            "use_pretrained_embeddings": True,
            "use_word": True,
            "use_char": True,
            "use_byte": False,
            "use_freq_bin": True
        },
    }

    pbar_lang = tqdm(total=3, desc="Languages", leave=True)

    with open(out_file, "w") as o:
        writer = csv.DictWriter(o, fieldnames=["lang", "experiment", "accuracy", "accuracy_oov"])

        for lang in langs:
            pbar_lang.set_description(f"Language {lang}")

            pbar_experiment = tqdm(total=3, desc="Experiments", leave=False)

            for experiment_name, experiment_params in experiments.items():

                pbar_experiment.set_description(f"Experiment {experiment_name}")

                data, net = train(lang_id=lang, pbar_start_pos=2, **experiment_params)
                results = evaluate(data, net)

                writer.writerow({"lang": lang, "experiment": experiment_name, **results})

                pbar_experiment.update(1)

            pbar_lang.update(1)

if __name__ == "__main__":
    fire.Fire()