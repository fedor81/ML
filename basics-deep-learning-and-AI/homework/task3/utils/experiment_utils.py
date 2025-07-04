import time

import torch
import torch.nn as nn
from tqdm import tqdm


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
    model, train_loader, test_loader, epochs=10, lr=0.001, device="cpu", patience=10, weight_decay=0
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_test=False
        )
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("-" * 100)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    end_time = time.time()

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "train_time": end_time - start_time,
    }
