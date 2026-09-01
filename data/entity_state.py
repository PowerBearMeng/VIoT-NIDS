"""Entity state aggregation and fixed-history temporal sequence construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import FlowDataset


@dataclass
class EntitySequences:
    histories: np.ndarray
    targets: np.ndarray
    capture_ids: np.ndarray
    window_indices: np.ndarray
    entity_ips: np.ndarray
    edge_entity_a: np.ndarray
    edge_entity_b: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def build_entity_sequences(
    dataset: FlowDataset,
    embeddings: np.ndarray,
    assignments: np.ndarray,
    *,
    prototype_count: int,
    indices: np.ndarray,
    history_windows: int,
    include_mean_embedding: bool,
) -> EntitySequences:
    embedding_dim = int(embeddings.shape[1])
    state_dim = 3 + prototype_count + (embedding_dim if include_mean_embedding else 0)
    states: dict[tuple[str, str, int], np.ndarray] = {}
    embedding_sums: dict[tuple[str, str, int], np.ndarray] = {}
    edge_keys: dict[int, tuple[tuple[str, str, int], tuple[str, str, int]]] = {}

    for raw_index in indices.tolist():
        index = int(raw_index)
        capture_id = str(dataset.capture_ids[index])
        window = int(dataset.window_indices[index])
        keys = (
            (capture_id, str(dataset.endpoint_a_ips[index]), window),
            (capture_id, str(dataset.endpoint_b_ips[index]), window),
        )
        edge_keys[index] = keys
        # An IP-to-itself flow is one incident entity edge, not two observations.
        for key in dict.fromkeys(keys):
            state = states.setdefault(key, np.zeros(state_dim, dtype=np.float64))
            state[0] += 1.0
            state[1] += float(dataset.packet_counts[index])
            state[2] += float(dataset.byte_counts[index])
            state[3 + int(assignments[index])] += 1.0
            if include_mean_embedding:
                embedding_sums.setdefault(key, np.zeros(embedding_dim, dtype=np.float64))
                embedding_sums[key] += embeddings[index]

    if include_mean_embedding:
        for key, total in embedding_sums.items():
            states[key][3 + prototype_count :] = total / max(1.0, states[key][0])

    ordered_keys = sorted(states, key=lambda item: (item[0], item[1], item[2]))
    key_to_row = {key: row for row, key in enumerate(ordered_keys)}
    histories = np.zeros((len(ordered_keys), history_windows, state_dim), dtype=np.float32)
    targets = np.zeros((len(ordered_keys), state_dim), dtype=np.float32)
    for row, key in enumerate(ordered_keys):
        capture_id, entity_ip, window = key
        targets[row] = states[key]
        for offset in range(history_windows):
            previous = (capture_id, entity_ip, window - history_windows + offset)
            if previous in states:
                histories[row, offset] = states[previous]

    edge_entity_a = np.full(len(dataset), -1, dtype=np.int64)
    edge_entity_b = np.full(len(dataset), -1, dtype=np.int64)
    for edge, keys in edge_keys.items():
        edge_entity_a[edge] = key_to_row[keys[0]]
        edge_entity_b[edge] = key_to_row[keys[1]]
    feature_names = ["active_flow_segment_count", "total_packet_count", "total_byte_count"]
    feature_names += [f"prototype_{index}_count" for index in range(prototype_count)]
    if include_mean_embedding:
        feature_names += [f"mean_embedding_{index}" for index in range(embedding_dim)]
    return EntitySequences(
        histories=histories,
        targets=targets,
        capture_ids=np.asarray([key[0] for key in ordered_keys]),
        window_indices=np.asarray([key[2] for key in ordered_keys], dtype=np.int64),
        entity_ips=np.asarray([key[1] for key in ordered_keys]),
        edge_entity_a=edge_entity_a,
        edge_entity_b=edge_entity_b,
        feature_names=feature_names,
    )
