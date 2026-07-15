import numpy as np
import torch


def compute_class_weights_tensor(class_counts: dict, class_order: list) -> torch.Tensor:
    counts = np.array([class_counts.get(c, 0) for c in class_order], dtype=np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (len(class_order) * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for pre, post, labels in dataloader:
        pre, post, labels = pre.to(device), post.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pre, post)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pre.size(0)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pre, post, labels in dataloader:
            pre, post, labels = pre.to(device), post.to(device), labels.to(device)
            outputs = model(pre, post)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * pre.size(0)
    return total_loss / len(dataloader.dataset)
