"""Train the MNIST target: a 784-256-256-10 ReLU MLP, Adam, 12 epochs. Writes ~/mpd-data/mnist/mlp.pt and prints test
accuracy.

  ~/mpd-data/venv/bin/python experiments/oracle/mnist/train_mlp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, MLP, load_mnist  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    xtr, ytr, xte, yte = load_mnist()
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs, bs = 12, 128
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(xtr) // bs))
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        model.train()
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i : i + bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xte).argmax(-1) == yte).float().mean().item()
        print(f"epoch {ep} loss {loss.item():.4f} test acc {acc:.4f}", flush=True)
    with torch.no_grad():
        tr_acc = (model(xtr).argmax(-1) == ytr).float().mean().item()
        te_acc = (model(xte).argmax(-1) == yte).float().mean().item()
    torch.save(model.state_dict(), DATA / "mlp.pt")
    (DATA / "mlp.json").write_text(json.dumps({"train_acc": tr_acc, "test_acc": te_acc, "epochs": epochs}))
    print(json.dumps({"train_acc": tr_acc, "test_acc": te_acc}))


if __name__ == "__main__":
    main()
