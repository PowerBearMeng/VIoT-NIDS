"""Evaluate the frozen V6 model on the existing labelled Gotham captures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from viot_nids.benchmark import CAPTURE_GLOB
from viot_nids.evaluation import evaluate, metrics
from viot_nids.features import extract
from .pipeline import rescore_components, score

LOG = logging.getLogger(__name__)


def benchmark(attack_root: Path, model_dir: Path, out: Path,
              v5_cache: Path | None = None, family: str | None = None,
              only: str | None = None, device: str = "cpu") -> dict:
    captures = sorted(attack_root.rglob(CAPTURE_GLOB))
    if family:
        captures = [p for p in captures if p.relative_to(attack_root).parts[0] == family]
    if only:
        captures = [p for p in captures if p.parent.name == only]
    if not captures:
        raise FileNotFoundError("No matching Access-OVS captures")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_y, all_s = [], []
    for i, pcap in enumerate(captures, 1):
        relative = pcap.relative_to(attack_root)
        family_name, name = relative.parts[0], pcap.parent.name
        run = out / family_name / name
        run.mkdir(parents=True, exist_ok=True)
        flows, scores = run / "flows.csv", run / "scores.csv.gz"
        cached = v5_cache / family_name / name if v5_cache else None
        labels = run / "labels.csv"
        if (run / "metrics.json").is_file() and scores.is_file() and labels.is_file():
            LOG.info("reusing %d/%d %s", i, len(captures), name)
            result = json.loads((run / "metrics.json").read_text())
        else:
            LOG.info("evaluating %d/%d %s", i, len(captures), name)
            if cached and (cached / "scores.csv.gz").is_file() and (cached / "labels.csv").is_file():
                extract(pcap, flows)
                # Label mapping was already produced from this exact capture;
                # score() validates every row's time and five-tuple against V5.
                labels.write_bytes((cached / "labels.csv").read_bytes())
                base_scores = cached / "scores.csv.gz"
            else:
                packet_labels = pcap.parent / "labels" / (pcap.stem + ".packet_labels.csv")
                extract(pcap, flows, labels, packet_labels)
                base_scores = None
            score(flows, model_dir, scores, device, base_scores)
            flows.unlink()
            result = evaluate(scores, labels, run, name)
            LOG.info("%s AUROC=%.6f EER=%.6f", name, result["AUROC"], result["EER"])
        summary[name] = result
        scored = pd.read_csv(scores, usecols=["row_id", "final_score"])
        truth = pd.read_csv(labels, usecols=["row_id", "binary_label"])
        joined = scored.merge(truth, on="row_id", validate="one_to_one")
        all_y.append(joined.binary_label.to_numpy(dtype=np.int8))
        all_s.append(joined.final_score.to_numpy(dtype=np.float64))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pooled, roc, pr = metrics(np.concatenate(all_y), np.concatenate(all_s))
    (out / "overall_metrics.json").write_text(json.dumps(pooled, indent=2))
    roc.to_csv(out / "overall_roc.csv", index=False)
    pr.to_csv(out / "overall_pr.csv", index=False)
    return {"captures": len(captures), "overall": pooled, "output": str(out)}


def rescore_benchmark(previous_root: Path, model_dir: Path, out: Path) -> dict:
    """Reuse frozen V6 Local/Context outputs while revising relation evidence."""
    sources = sorted(previous_root.glob("*/*/scores.csv.gz"))
    if not sources:
        raise FileNotFoundError(f"No V6 component scores under {previous_root}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_y, all_s = [], []
    for i, previous in enumerate(sources, 1):
        family_name, name = previous.parent.parent.name, previous.parent.name
        run = out / family_name / name
        run.mkdir(parents=True, exist_ok=True)
        scores, labels = run / "scores.csv.gz", run / "labels.csv"
        if (run / "metrics.json").is_file() and scores.is_file() and labels.is_file():
            result = json.loads((run / "metrics.json").read_text())
        else:
            LOG.info("rescoring %d/%d %s", i, len(sources), name)
            rescore_components(previous, model_dir, scores)
            labels.write_bytes((previous.parent / "labels.csv").read_bytes())
            result = evaluate(scores, labels, run, name)
            LOG.info("%s AUROC=%.6f EER=%.6f", name, result["AUROC"], result["EER"])
        summary[name] = result
        scored = pd.read_csv(scores, usecols=["row_id", "final_score"])
        truth = pd.read_csv(labels, usecols=["row_id", "binary_label"])
        joined = scored.merge(truth, on="row_id", validate="one_to_one")
        all_y.append(joined.binary_label.to_numpy(dtype=np.int8))
        all_s.append(joined.final_score.to_numpy(dtype=np.float64))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pooled, roc, pr = metrics(np.concatenate(all_y), np.concatenate(all_s))
    (out / "overall_metrics.json").write_text(json.dumps(pooled, indent=2))
    roc.to_csv(out / "overall_roc.csv", index=False)
    pr.to_csv(out / "overall_pr.csv", index=False)
    return {"captures": len(sources), "overall": pooled, "output": str(out)}
