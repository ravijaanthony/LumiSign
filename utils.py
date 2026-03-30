import torch
import random
import os
import numpy as np
import json
import re


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path):
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file


def load_label_map(dataset):
    dataset = (dataset or "").strip()
    if not dataset:
        raise ValueError("dataset must be a non-empty string")

    dataset_lower = dataset.lower()
    dataset_slug = re.sub(r"[^a-z0-9]+", "_", dataset_lower).strip("_")
    dataset_compact = dataset_slug.replace("_", "")

    candidate_names = [
        f"label_map_{dataset}.json",
        f"label_map_{dataset_lower}.json",
        f"label_map_{dataset_slug}.json",
        f"label_map_{dataset_compact}.json",
    ]

    # Support common aliases used for this project's ISL split dataset.
    isl_aliases = {"isl_split_dataset", "islsplit", "islsplitdataset", "isl_split"}
    if dataset_slug in isl_aliases or dataset_compact in isl_aliases:
        candidate_names.extend(
            [
                "label_map_isl_split_dataset.json",
                "label_map_islsplit.json",
                "label_map_dataset.json",
                "label_map_Dataset.json",
            ]
        )

    seen = set()
    candidates = []
    for name in candidate_names:
        if name and name not in seen:
            seen.add(name)
            candidates.append(os.path.join("label_maps", name))

    for file_path in candidates:
        if os.path.isfile(file_path):
            return load_json(file_path)

    searched = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"No label map found for dataset '{dataset}'. Tried:\n  - {searched}"
    )


def get_experiment_name(args):
    exp_name = ""
    if args.use_cnn:
        exp_name += "cnn_"
    if args.use_augs:
        exp_name += "augs_"
    exp_name += args.model
    if args.model == "transformer":
        exp_name += f"_{args.transformer_size}"
    return exp_name


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=5, mode="min", delta=0.0):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, model_path, epoch_score, model, optimizer, scheduler=None):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, optimizer, scheduler, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else scheduler,
                    "score": epoch_score,
                },
                model_path,
            )
        self.val_score = epoch_score
