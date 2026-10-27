"""Training model definitions; reused seasonal blend and compact LSTM architecture."""
import numpy as np
import torch
from torch import nn


class CompactLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def fit_baseline(frame):
    return {c: {f"{dow}:{hour}": float(value) for (dow, hour), value in
                frame.groupby(["dayofweek", "hour"])[c].median().items()}
            for c in ["energy_kwh", "sessions"]}


def baseline_predict(frame, profiles):
    values = []
    for c in ["energy_kwh", "sessions"]:
        profile = np.array([profiles[c][f"{int(d)}:{int(h)}"]
                            for d, h in zip(frame.dayofweek, frame.hour)])
        values.append(.5 * frame[f"{c}_lag_24"].to_numpy() + .5 * profile)
    return np.column_stack(values)
