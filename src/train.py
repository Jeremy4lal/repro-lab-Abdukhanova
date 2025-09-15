import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# === подключение seed_utils ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seed_utils import set_seed  # noqa: E402


# === генерация синтетических данных ===
def make_dataset(seed, n_samples=1000):
    """Генерирует два гауссовых класса по параметрам варианта."""
    set_seed(seed)

    mu0 = np.array([-1.0, -0.5])
    mu1 = np.array([+1.0, +0.7])
    sigma = 0.9

    x0 = np.random.normal(loc=mu0, scale=sigma, size=(n_samples, 2))
    x1 = np.random.normal(loc=mu1, scale=sigma, size=(n_samples, 2))

    X = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


# === модель ===
class SimpleNet(nn.Module):
    """Простая модель: Linear -> ReLU -> Linear."""

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def compute_sha256(path):
    """Вычисление SHA256 от файла модели."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)

    # === данные ===

    dataset = make_dataset(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,  # type: ignore[arg-type]
        num_workers=0,
    )

    # === модель ===
    model = SimpleNet(hidden_dim=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
    )

    # === обучение ===
    for epoch in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    final_loss = loss.item()

    # === сохранение модели ===
    os.makedirs("runs", exist_ok=True)
    model_path = f"runs/model_seed{args.seed}.pt"
    torch.save(model.state_dict(), model_path)

    sha256 = compute_sha256(model_path)

    # === сохранение протокола ===
    run_info = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_loss": final_loss,
        "model_path": model_path,
        "model_sha256": sha256,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }

    run_path = f"runs/run_seed{args.seed}.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"Training finished. Final loss: {final_loss:.8f}")
    print(f"Model saved to {model_path}, SHA256: {sha256}")


if __name__ == "__main__":
    main()
