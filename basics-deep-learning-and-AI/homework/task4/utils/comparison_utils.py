import os
import torch

from ...task3.utils.experiment_utils import train_model
from ...task3.utils.visualization_utils import count_parameters, plot_results


def compare_models(
    models: dict[str, torch.nn.Module],
    train_loader,
    test_loader,
    models_save_folder=None,
    graphic_save_path=None,
    epochs=10,
    gradient_analysis=False,
    device="cpu",
) -> dict[str, dict[str, any]]:
    """Проводит сравнение нейронных сетей"""
    results = {}

    for name, model in models.items():
        print(f"\nТренировка модели: {name}")
        models[name] = model.to(device)

        history = train_model(
            model,
            train_loader,
            test_loader,
            epochs=epochs,
            save_folder=os.path.join(models_save_folder, name) if models_save_folder else None,
            gradient_analysis=gradient_analysis,
            device=device,
        )
        history["count_params"] = count_parameters(model)
        results[name] = history

    plot_results(results, save_path=graphic_save_path)  # Вывод графиков
    return results


def calculate_rf(config):
    """Считает рецептивное поле"""
    rf = 1
    for layer in config:
        k = layer["kernel"]
        rf = rf + (k - 1)
    return rf
