"""Role, regime, context, calibration and responsibility statistics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

CONTEXT_NAMES = (
    "flow_count", "total_bytes", "total_packets", "bytes_per_flow_mean",
    "bytes_per_flow_std", "packets_per_flow_mean", "packets_per_flow_std",
    "unique_peer_count", "unique_dst_port_count", "new_peer_count",
)
RESPONSIBILITY_COLUMNS = (0, 1, 2, 7, 8, 9)


class EmpiricalCDF:
    def __init__(self, values: np.ndarray) -> None:
        if len(values) == 0:
            raise ValueError("Calibration requires benign validation rows")
        self.values = np.sort(np.asarray(values, dtype=np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.values, values, side="right") / (len(self.values) + 1)


def prototypes(z: np.ndarray, seconds: np.ndarray,
               channels: list[np.ndarray]) -> np.ndarray:
    output = []
    for indices in channels:
        block = z[indices]
        gaps = np.diff(seconds[indices])
        log_gaps = np.log1p(gaps.astype(np.float64))
        output.append(np.concatenate([
            block.mean(axis=0), block.std(axis=0, ddof=0),
            np.median(block, axis=0), np.percentile(block, 25, axis=0),
            np.percentile(block, 75, axis=0),
            [np.log1p(len(indices)), np.median(log_gaps) if len(gaps) else 0.0,
             log_gaps.std(ddof=0) if len(gaps) else 0.0],
        ]))
    return np.asarray(output, dtype=np.float64)


def discover_roles(proto: np.ndarray, channels: list[np.ndarray],
                   n_flows: int, maximum: int, seed: int) -> tuple[StandardScaler, BayesianGaussianMixture, np.ndarray, dict[int, int]]:
    if len(proto) < 2:
        raise ValueError("At least two benign temporal channels are required")
    scaler = StandardScaler().fit(proto)
    mixture = BayesianGaussianMixture(
        n_components=min(maximum, len(proto)), covariance_type="diag",
        reg_covar=1e-4, random_state=seed, max_iter=200,
        weight_concentration_prior_type="dirichlet_process",
    ).fit(scaler.transform(proto))
    channel_roles = mixture.predict(scaler.transform(proto))
    roles = np.full(n_flows, -1, dtype=np.int32)
    counts: dict[int, int] = {}
    for indices, role in zip(channels, channel_roles):
        roles[indices] = int(role)
        counts[int(role)] = counts.get(int(role), 0) + 1
    assert (roles >= 0).all()
    return scaler, mixture, roles, counts


def select_gmm(x: np.ndarray, maximum: int, seed: int,
               reg_covar: float = 1e-4) -> GaussianMixture:
    if len(x) < 2:
        raise ValueError("GMM needs at least two training rows")
    best: GaussianMixture | None = None
    best_bic = np.inf
    for k in range(1, min(maximum, len(x) // 2) + 1):
        model = GaussianMixture(n_components=k, covariance_type="diag",
                                reg_covar=reg_covar, random_state=seed,
                                max_iter=150, n_init=1).fit(x)
        bic = model.bic(x)
        if bic < best_bic:
            best, best_bic = model, bic
    assert best is not None
    return best


def fit_regimes(z: np.ndarray, roles: np.ndarray, maximum: int,
                seed: int, reg_covar: float) -> dict[int, GaussianMixture]:
    result = {}
    for role in np.unique(roles):
        block = z[roles == role]
        if len(block) >= 2:
            result[int(role)] = select_gmm(block, maximum, seed, reg_covar)
    if not result:
        raise ValueError("No role has two training flows")
    return result


def assign_local(z: np.ndarray, models: dict[int, GaussianMixture]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use conditional P(z|role) only; role frequency/prior is excluded."""
    role_ids = sorted(models)
    raw_by_role = np.empty((len(z), len(role_ids)), dtype=np.float64)
    for start in range(0, len(z), 16384):
        stop = min(start + 16384, len(z))
        for column, role in enumerate(role_ids):
            raw_by_role[start:stop, column] = -models[role].score_samples(z[start:stop])
    best = np.argmin(raw_by_role, axis=1)
    roles = np.asarray(role_ids, dtype=np.int32)[best]
    raw = raw_by_role[np.arange(len(z)), best]
    regimes = np.empty(len(z), dtype=np.int32)
    for role in role_ids:
        ix = np.flatnonzero(roles == role)
        for part in np.array_split(ix, max(1, (len(ix) + 16383) // 16384)):
            if len(part):
                regimes[part] = models[role].predict(z[part])
    return roles, regimes, raw


def context_groups(frame: pd.DataFrame, roles: np.ndarray,
                   seen: dict[tuple[str, int], set[str]] | None = None
                   ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Create contexts and per-flow contribution vectors in timestamp order."""
    seen = seen if seen is not None else defaultdict(set)
    n = len(frame)
    contexts: list[np.ndarray] = []
    members: list[np.ndarray] = []
    contributions = np.zeros((n, 6), dtype=np.float64)
    data = frame.reset_index(drop=True)
    data = data.assign(_role=roles)
    for (second, host, role), group in data.groupby(["timestamp_second", "src_ip", "_role"], sort=True):
        ix = group.index.to_numpy(dtype=np.int64)
        peers = group.dst_ip.astype(str)
        ports = group.dst_port.astype(int)
        peer_counts = peers.value_counts()
        port_counts = ports.value_counts()
        key = (str(host), int(role))
        old = seen.setdefault(key, set())
        new_peers = set(peers) - old
        byte_values = group.byte_count.to_numpy(dtype=np.float64)
        packet_values = group.packet_count.to_numpy(dtype=np.float64)
        contexts.append(np.asarray([
            len(ix), byte_values.sum(), packet_values.sum(),
            byte_values.mean(), byte_values.std(ddof=0),
            packet_values.mean(), packet_values.std(ddof=0),
            peers.nunique(), ports.nunique(), len(new_peers),
        ], dtype=np.float64))
        members.append(ix)
        contributions[ix, 0] = 1.0
        contributions[ix, 1] = byte_values
        contributions[ix, 2] = packet_values
        contributions[ix, 3] = [1.0 / peer_counts[p] for p in peers]
        contributions[ix, 4] = [1.0 / port_counts[p] for p in ports]
        contributions[ix, 5] = [1.0 / peer_counts[p] if p in new_peers else 0.0 for p in peers]
        old.update(peers)
    return np.vstack(contexts) if contexts else np.empty((0, 10)), members, contributions


def group_roles(members: list[np.ndarray], roles: np.ndarray) -> np.ndarray:
    return np.asarray([roles[ix[0]] for ix in members], dtype=np.int32)


def fit_context(contexts: np.ndarray, group_role: np.ndarray, maximum: int,
                seed: int, reg_covar: float) -> tuple[dict[int, RobustScaler], dict[int, GaussianMixture], dict[int, tuple[np.ndarray, np.ndarray]]]:
    scalers = {}
    models = {}
    residual_stats = {}
    for role in np.unique(group_role):
        x = contexts[group_role == role]
        if len(x) < 2:
            continue
        scalers[int(role)] = RobustScaler().fit(x)
        models[int(role)] = select_gmm(scalers[int(role)].transform(x), maximum, seed, reg_covar)
        chosen = x[:, RESPONSIBILITY_COLUMNS]
        residual_stats[int(role)] = (chosen.mean(axis=0), np.maximum(chosen.std(axis=0, ddof=0), 1e-6))
    return scalers, models, residual_stats


def context_raw(contexts: np.ndarray, group_role: np.ndarray,
                scalers: dict[int, RobustScaler], models: dict[int, GaussianMixture]) -> np.ndarray:
    raw = np.full(len(contexts), np.inf, dtype=np.float64)
    for role, model in models.items():
        ix = np.flatnonzero(group_role == role)
        if len(ix):
            raw[ix] = -model.score_samples(scalers[role].transform(contexts[ix]))
    return raw


def responsibility(q: np.ndarray, context: np.ndarray,
                   mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    residual = np.maximum((context[list(RESPONSIBILITY_COLUMNS)] - mean) / std, 0)
    denom = np.linalg.norm(q, axis=1) * np.linalg.norm(residual)
    return np.clip(np.divide(q @ residual, denom, out=np.zeros(len(q)), where=denom > 0), 0, 1)
