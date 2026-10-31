"""
Live inference helper for the trained topology-aware GNN.
Shadow mode only: this module predicts next observed utilization and does not route traffic.
"""
from collections import deque
from datetime import datetime
import csv
import os

import joblib
import numpy as np
import torch
from torch import nn


class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_fc = nn.Linear(in_dim, out_dim)
        self.neighbor_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return torch.relu(self.self_fc(x) + self.neighbor_fc(torch.matmul(adj, x)))


class GraphForecaster(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.g1 = GraphLayer(n_features, hidden)
        self.g2 = GraphLayer(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, max(hidden // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 4), 1),
        )

    def forward(self, x, adj):
        return self.head(self.g2(self.g1(x, adj), adj)).squeeze(-1)


class LiveGNNPredictor:
    """
    update_switch(switch, rx_mbps, tx_mbps, utilization_percent, timestamp=None)
    returns predictions only when a complete aligned snapshot is available.
    Feature order matches train_gnn_traffic_forecaster.py:
    rx_lag1, tx_lag1, util_lag1, util_lag2, util_lag3,
    util_mean3, util_delta1, rx_delta1, tx_delta1.
    """
    FEATURE_NAMES = [
        "rx_lag1", "tx_lag1", "util_lag1", "util_lag2", "util_lag3",
        "util_mean3", "util_delta1", "rx_delta1", "tx_delta1",
    ]

    def __init__(self, model_path, csv_path, max_snapshot_skew_seconds=3.0):
        self.model_path = os.path.expanduser(model_path)
        self.csv_path = os.path.expanduser(csv_path)
        self.max_snapshot_skew_seconds = float(max_snapshot_skew_seconds)
        payload = joblib.load(self.model_path)
        self.switches = [str(s) for s in payload["switches"]]
        self.feature_names = list(payload["feature_names"])
        if self.feature_names != self.FEATURE_NAMES:
            raise ValueError(
                f"GNN feature order mismatch: artifact={self.feature_names}, "
                f"expected={self.FEATURE_NAMES}"
            )
        self.mean = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(payload["feature_std"], dtype=np.float32).reshape(-1)
        self.adj_np = np.asarray(payload["adjacency"], dtype=np.float32)
        if self.mean.shape != (len(self.FEATURE_NAMES),) or self.std.shape != self.mean.shape:
            raise ValueError("GNN normalization vector shape mismatch")
        self.std[self.std < 1e-6] = 1.0
        self.hidden = int(payload["hidden_dim"])
        self.model = GraphForecaster(len(self.FEATURE_NAMES), self.hidden).cpu()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.adj = torch.tensor(self.adj_np, dtype=torch.float32)
        # Store raw switch-level samples; only infer after all nodes have enough history.
        self.history = {s: deque(maxlen=3) for s in self.switches}
        self.latest = {}
        self.last_inference_signature = None
        self._init_csv()

    def _init_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "switch", "rx_mbps", "tx_mbps",
                    "current_utilization_percent", "predicted_next_utilization_percent",
                    "status", "snapshot_skew_seconds",
                ])

    def update_switch(self, switch, rx_mbps, tx_mbps, utilization_percent, timestamp=None):
        switch = str(switch)
        if switch not in self.history:
            return None
        now = float(timestamp if timestamp is not None else __import__("time").time())
        vals = np.asarray([rx_mbps, tx_mbps, utilization_percent], dtype=float)
        if not np.isfinite(vals).all():
            return None
        rx, tx, util = float(vals[0]), float(vals[1]), float(np.clip(vals[2], 0, 100))
        old = self.latest.get(switch)
        # Avoid creating multiple samples for the same switch in the same stats cycle.
        if old is not None and now <= old["time"]:
            return None
        self.latest[switch] = {"time": now, "rx": rx, "tx": tx, "util": util}
        self.history[switch].append({"time": now, "rx": rx, "tx": tx, "util": util})
        if any(len(self.history[s]) < 3 for s in self.switches):
            return None
        newest_times = [self.history[s][-1]["time"] for s in self.switches]
        skew = max(newest_times) - min(newest_times)
        if skew > self.max_snapshot_skew_seconds:
            return None
        signature = tuple(round(t, 3) for t in newest_times)
        if signature == self.last_inference_signature:
            return None

        x_rows = []
        for s in self.switches:
            h = list(self.history[s])
            util_vals = [p["util"] for p in h]
            rx_vals = [p["rx"] for p in h]
            tx_vals = [p["tx"] for p in h]
            # Training uses current sample as lag1, previous samples as lag2/3.
            util_delta = util_vals[-1] - util_vals[-2]
            rx_delta = rx_vals[-1] - rx_vals[-2]
            tx_delta = tx_vals[-1] - tx_vals[-2]
            row = [
                rx_vals[-1], tx_vals[-1], util_vals[-1],
                util_vals[-2], util_vals[-3],
                float(np.mean(util_vals[-3:])),
                util_delta, rx_delta, tx_delta,
            ]
            x_rows.append(row)
        x = np.asarray(x_rows, dtype=np.float32)
        if not np.isfinite(x).all():
            return None
        x = np.clip((x - self.mean) / self.std, -10, 10).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.tensor(x, dtype=torch.float32), self.adj).numpy()
        pred = np.clip(pred, 0, 100)
        stamp = datetime.now(__import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i, s in enumerate(self.switches):
                latest = self.history[s][-1]
                writer.writerow([
                    stamp, s, round(latest["rx"], 5), round(latest["tx"], 5),
                    round(latest["util"], 5), round(float(pred[i]), 5),
                    "OK_SHADOW", round(float(skew), 3),
                ])
        self.last_inference_signature = signature
        return {s: float(pred[i]) for i, s in enumerate(self.switches)}
