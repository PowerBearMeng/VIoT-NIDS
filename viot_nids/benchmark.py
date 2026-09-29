"""Run the frozen V5 model over the existing Gotham attack captures and labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate, metrics
from .features import extract
from .pipeline import score

LOG = logging.getLogger(__name__)
CAPTURE_GLOB = "*Access-OVS_0-0_to_VyOS-R1_0-0.pcap"


def benchmark(attack_root: Path, model_dir: Path, out: Path,
              device: str = "cpu", family_filter: str | None = None,
              only_name: str | None = None) -> dict:
    captures = sorted(attack_root.rglob(CAPTURE_GLOB))
    if family_filter:
        captures = [p for p in captures if p.relative_to(attack_root).parts[0] == family_filter]
    if only_name:
        captures = [p for p in captures if p.parent.name == only_name]
    if not captures:
        raise FileNotFoundError(f"No Access-OVS attack captures under {attack_root}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    all_labels = []
    all_scores = []
    for index, pcap in enumerate(captures, 1):
        relative = pcap.relative_to(attack_root)
        family = relative.parts[0]
        attack_name = pcap.parent.name
        label_path = pcap.parent / "labels" / (pcap.stem + ".packet_labels.csv")
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        run_dir = out / family / attack_name
        run_dir.mkdir(parents=True, exist_ok=True)
        flows = run_dir / "flows.csv"
        labels = run_dir / "labels.csv"
        scores = run_dir / "scores.csv.gz"
        if (run_dir / "metrics.json").is_file() and scores.is_file() and labels.is_file():
            LOG.info("reusing completed attack %d/%d: %s", index, len(captures), attack_name)
            outcome = json.loads((run_dir / "metrics.json").read_text())
            summary[attack_name] = outcome
            scored = pd.read_csv(scores, usecols=["row_id", "final_score"])
            truth = pd.read_csv(labels, usecols=["row_id", "binary_label"])
            joined = scored.merge(truth, on="row_id", validate="one_to_one")
            all_labels.append(joined.binary_label.to_numpy(dtype=np.int8))
            all_scores.append(joined.final_score.to_numpy(dtype=np.float64))
            continue
        LOG.info("attack %d/%d: %s", index, len(captures), attack_name)
        extract(pcap, flows, labels, label_path)
        score(flows, model_dir, scores, device)
        flows.unlink()  # extraction is reproducible from the original PCAP
        outcome = evaluate(scores, labels, run_dir, attack_name)
        summary[attack_name] = outcome
        scored = pd.read_csv(scores, usecols=["row_id", "final_score"])
        truth = pd.read_csv(labels, usecols=["row_id", "binary_label"])
        joined = scored.merge(truth, on="row_id", validate="one_to_one")
        all_labels.append(joined.binary_label.to_numpy(dtype=np.int8))
        all_scores.append(joined.final_score.to_numpy(dtype=np.float64))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        LOG.info("%s: AUROC=%s AUPRC=%s EER=%s", attack_name,
                 outcome["AUROC"], outcome["AUPRC"], outcome["EER"])
    pooled, roc, pr = metrics(np.concatenate(all_labels), np.concatenate(all_scores))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "overall_metrics.json").write_text(json.dumps(pooled, indent=2))
    roc.to_csv(out / "overall_roc.csv", index=False)
    pr.to_csv(out / "overall_pr.csv", index=False)
    return {"captures": len(captures), "overall": pooled,
            "per_attack": summary, "output": str(out)}
