"""Normal-only transforms and shared normal behavior prototypes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

@dataclass
class Scale:
    center: np.ndarray
    scale: np.ndarray
    log: bool = True

    @classmethod
    def fit(cls, values, floor=0.05, log=True):
        values = np.asarray(values, dtype=np.float64)
        transformed = np.log1p(values) if log else values
        fitted = RobustScaler().fit(transformed)
        return cls(fitted.center_, np.maximum(fitted.scale_, floor), log)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        out = ((np.log1p(values) if self.log else values) - self.center) / self.scale
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite transformed observations")
        return out.astype(np.float32)


@dataclass
class Behavior:
    scaler: Scale
    centers: np.ndarray
    widths: np.ndarray
    support: np.ndarray

    @classmethod
    def fit(cls, embedding, seconds, k=16, seed=42):
        scaler = Scale.fit(embedding, log=False)
        z = scaler.transform(embedding).astype(np.float64)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
        distances = np.linalg.norm(z - km.cluster_centers_[km.labels_], axis=1)
        floor = max(1e-3, 0.1 * float(np.median(distances)))
        widths = np.array([max(floor, float(np.median(distances[km.labels_ == i])))
                           if (km.labels_ == i).any() else floor for i in range(k)])
        blocks = (seconds - seconds.min()) // 10
        support = np.array([len(np.unique(blocks[km.labels_ == i])) for i in range(k)])
        return cls(scaler, km.cluster_centers_, widths, support)

    def transform(self, embedding):
        z = self.scaler.transform(embedding).astype(np.float64)
        squared = ((z[:, None] - self.centers[None]) ** 2).sum(2)
        log_response = -squared / (2 * self.widths[None] ** 2)
        log_response -= log_response.max(1, keepdims=True)
        response = np.exp(log_response)
        return response / np.linalg.norm(response, axis=1, keepdims=True)
