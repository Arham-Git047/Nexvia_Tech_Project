from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch_geometric.nn import SAGEConv

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/baseline"))

class FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x).squeeze(-1)

class OnlineGraphScorer:
    """Stateful, point-in-time graph inference seeded with pre-test history."""
    def __init__(self):
        self.device = torch.device("cpu")
        state_path = MODEL_DIR / "initial_graph_state.joblib"
        model_path = MODEL_DIR / "graphsage_ieee_cis.pt"
        state = joblib.load(state_path)
        self.nodes = state["nodes"]
        self.histories = defaultdict(list, state["history_by_profile"])
        self.next_node_id = int(state["next_node_id"])
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = FraudGraphSAGE(
            input_dim=int(checkpoint.get("input_dim", 64)), hidden_dim=64
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.lock = __import__("threading").RLock()

    @staticmethod
    def _entity(value):
        if value is None or str(value) == "":
            return None
        return str(value)

    def score_and_update(self, event: dict, current_vector: np.ndarray) -> float:
        """Score one event using only current and earlier graph nodes.

        Histories are updated after the prediction. Duplicate transaction IDs are
        handled by the API before this method is called.
        """
        with self.lock:
            entities = event.get("entities") or {}
            profile_specs = [
                ("__card_profile", self._entity(entities.get("card_profile"))),
                ("__device_profile", self._entity(entities.get("device_profile"))),
            ]

            parent_ids = set()
            for column, profile in profile_specs:
                if profile is None:
                    continue
                parent_ids.update(self.histories.get((column, profile), [])[-2:])

            # Keep only graph nodes actually present in the saved seed/live state.
            parent_ids = {pid for pid in parent_ids if pid in self.nodes}
            ancestor_ids = set()
            for parent_id in parent_ids:
                ancestor_ids.update(self.nodes[parent_id].get("parents", []))
            ancestor_ids = {
                aid for aid in ancestor_ids
                if aid in self.nodes and aid not in parent_ids
            }

            ordered_ids = sorted(ancestor_ids) + sorted(parent_ids)
            local_index = {node_id: idx for idx, node_id in enumerate(ordered_ids)}
            current_local = len(ordered_ids)
            feature_rows = [
                np.asarray(self.nodes[node_id]["x"], dtype=np.float32)
                for node_id in ordered_ids
            ]
            feature_rows.append(np.asarray(current_vector, dtype=np.float32))
            x = torch.tensor(np.stack(feature_rows), dtype=torch.float32, device=self.device)

            edge_pairs = []
            # Earlier ancestors -> earlier neighboring transaction.
            for parent_id in parent_ids:
                rec = self.nodes[parent_id]
                for ancestor_id in rec.get("parents", []):
                    if ancestor_id in local_index:
                        edge_pairs.append((local_index[ancestor_id], local_index[parent_id]))
                # Earlier related transactions -> this incoming transaction.
                edge_pairs.append((local_index[parent_id], current_local))

            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=self.device).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x, edge_index)
                probability = float(torch.sigmoid(logits[current_local]).item())

            new_id = self.next_node_id
            self.next_node_id += 1
            self.nodes[new_id] = {
                "x": np.asarray(current_vector, dtype=np.float32),
                "parents": sorted(parent_ids),
            }

            for column, profile in profile_specs:
                if profile is None:
                    continue
                key = (column, profile)
                current_history = list(self.histories.get(key, []))
                current_history.append(new_id)
                self.histories[key] = current_history[-2:]

            return probability
