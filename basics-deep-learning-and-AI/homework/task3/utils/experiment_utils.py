import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def save_model(model, path, **kwargs):
    """Сохраняет модель"""
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(path):
        os.makedirs(dir_path, exist_ok=True)

    state = {
        "model": model.state_dict(),
    }
    state.update(kwargs)
    torch.save(state, path)
    print("Состояние модели сохранено в файл: ", path)


def load_model(model, path) -> dict:
    """Загружает модель"""
    state = torch.load(path)
    model.load_state_dict(state["model"])
    del state["model"]
    print("Состояние загружено из файла: ", path)
    return state


def run_epoch(model, data_loader, criterion, optimizer=None, device="cpu", is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(
        tqdm(data_loader, desc="Training" if not is_test else "Testing")
    ):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=5,
    lr=0.001,
    device="cpu",
    patience=10,
    weight_decay=0,
    save_folder=None,
    gradient_analysis=False,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Для анализа градиентов
    best_loss = float("inf")
    patience_counter = 0

    history = {
        "train_losses": [],
        "train_accs": [],
        "test_losses": [],
        "test_accs": [],
        "train_time": 0,
        "inference_time": float("inf"),
    }

    # Для анализа градиентов
    if gradient_analysis:
        history["gradients"] = (
            {name: [] for name, param in model.named_parameters() if param.requires_grad},
        )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        start_epoch_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_test=False
        )

        # Инференс
        inference_start_time = time.time()
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        history["inference_time"] = min(
            time.time() - inference_start_time, history["inference_time"]
        )

        # Запись градиентов
        if gradient_analysis:
            for param_name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    history["gradients"][param_name].append(param.grad.abs().mean().item())

        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)
        history["test_losses"].append(test_loss)
        history["test_accs"].append(test_acc)
        history["train_time"] += time.time() - start_epoch_time

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("-" * 100)

        if save_folder:  # Сохранение модели
            save_model(model, os.path.join(save_folder, f"epoch_{epoch + 1}.pt"), **history)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return history
