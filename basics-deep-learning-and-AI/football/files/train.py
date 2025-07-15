import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import custom_dataset, dataset_load, models, visualization, train_utils


ROOT_DIR = visualization.ROOT_DIR


def run():
    reset_cache = "--reset-cache" in sys.argv
    batch_size = 256

    train_dataset, val_dataset, test_dataset = custom_dataset.get_train_val_test_datasets(
        val_size=0.15, reset_cache=reset_cache
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_dataset.collate_fn,
    )

    info = dataset_load.get_dataset_info()
    model = models.AdvancedFootballModel(
        info["num_teams"],
        info["num_tournaments"],
        info["num_cities"],
        info["num_countries"],
        info["num_scorers"],
    )

    weights_folder = os.path.join(ROOT_DIR, "weights")

    result = train_utils.train_model(
        model,
        train_loader,
        val_loader,
        end_epoch=50,
        patience=10,
        lr=1e-3,
        save_folder=weights_folder,
        delete_old_weights=True,
    )

    visualization.show_table(result, model)
    visualization.plot(result)

    # Загрузить лучший вес
    state = torch.load(os.path.join(weights_folder, f"epoch_{result.best_epoch}.pt"))
    model.load_state_dict(state["model"])
    visualization.show_metrics(
        model, test_loader, save_folder=os.path.join(ROOT_DIR, "plots")
    )  # Предикт и метрики


if __name__ == "__main__":
    run()
