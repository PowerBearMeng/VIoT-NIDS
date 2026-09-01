# Three-stage unsupervised flow NIDS

This directory implements the full method described in `Design.md` and the
event-aware/historical revision in `DesignV2.md`: a
payload-free PCAP-to-flow pipeline, masked-reconstruction TCN and normal
prototypes, entity temporal prediction, reliability-aware spatial context, and
normal-only score calibration.

## Security and learning contract

- The current V3 Gotham configuration groups TCP/UDP packets into
  **directional five-tuples**, matching the evaluated TFusion baseline.
- Endpoint A is the packet source and endpoint B is the packet destination;
  reverse traffic forms a separate Flow. Legacy bidirectional construction is
  retained behind `data.flow_orientation: bidirectional`.
- The current V3 Gotham configuration uses epoch-aligned half-open 3-second
  windows `[kW, (k+1)W)`, matching TFusion. Capture-relative alignment remains
  available through `data.window_alignment: capture`.
- Each segment contains 30 × 100 ms bins and exactly six features: directional
  packet counts, byte counts, and mean frame lengths.
- Payload and application protocol semantics are never parsed. IPs and ports
  are metadata used for aggregation only; they are not numeric neural inputs.
- Flow TCN, prototypes, entity GRU, and spatial MLP all select only configured
  normal training samples. Thresholds and component scales use only normal
  calibration samples. Attack labels are consumed only by evaluation metrics.

## Architecture

```text
PCAP/PCAPNG
  -> directional five-tuple + epoch-aligned 3 s segment
  -> X_flow [30, 6]
  -> event-aware masked TCN + attention/max pooling -> z_flow + reconstruction error
  -> normal MiniBatchKMeans -> prototype distance -> local anomaly

per IP/window normal state
  -> counts + bytes + prototype histogram + optional mean z_flow
  -> GRU(previous states) -> entity anomaly

historical IP entity / unordered IP-pair state
  -> local + embedding-history update reliability (never port novelty)
  -> read prior windows, then commit one update per IP pair per window
  -> MLP(C_a, C_b) predicts z_flow -> spatial anomaly

normal calibration quantiles
  -> normalized components + configured unsupervised weighted final score
  -> deployment threshold at normal quantile 1 - target FPR
```

The entity component is reported independently and has zero final-score weight
by default, preserving the Phase 2 requirement. V2 keeps exactly the original
30 × 6 traffic tensor. Port values, port novelty, and application semantics are
absent from the learned inputs and historical reliability rules.

## Install and configure

Python 3.10+ is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `configs/default.yaml`. Dataset paths can be expressed with `data.sources`
or with a `pcap,label,split` manifest. All relative paths are resolved from the
project root, not from the caller's current directory.

## Commands

Full end-to-end run (preparation, all three normal-only models, calibration, and test evaluation):

```bash
python run_pipeline.py --config configs/default.yaml --mode all
```

To keep preparation and training separate:

```bash
python run_pipeline.py --config configs/default.yaml --mode prepare
python run_pipeline.py --config configs/default.yaml --mode train
```

Evaluate the frozen pipeline on the test split:

```bash
python run_pipeline.py --config configs/default.yaml --mode evaluate
```

Individual stages are also first-class commands:

```bash
python prepare_data.py --config configs/default.yaml
python train_flow.py --config configs/default.yaml
python extract_embeddings.py --config configs/default.yaml
python fit_prototypes.py --config configs/default.yaml
python train_entity.py --config configs/default.yaml
python train_spatial.py --config configs/default.yaml
python calibrate.py --config configs/default.yaml
python evaluate_flow.py --config configs/default.yaml --split test
```

## Smoke test

Generate four tiny deterministic captures and run all stages on CPU:

```bash
python tests/generate_smoke_pcaps.py
python run_pipeline.py --config configs/smoke.yaml --mode all --device cpu
python -m unittest discover -s tests -v
```

Design V3.0 使用正常行为 mode 的 pair/entity composition 和 normal-tail max
fusion。完整设计见 `DesignV3.md`，快速验证命令为：

```bash
python run_pipeline.py --config configs/smoke_v3.yaml --mode all --device cpu
```

Evaluation writes per-flow scores, per-entity temporal scores, and JSON metrics.
Metrics include AUROC, AUPRC, deployment-threshold FPR/TPR, TPR@FPR≤1%, and
TPR@FPR≤0.1%. Runtime separately reports PCAP feature construction, TCN flow
inference, and cached-embedding entity/spatial context throughput. The generated
captures are functional fixtures, not a scientific benchmark dataset.

## Artifacts

The configured output directory contains:

- `flow_model.pt`: TCN weights and train-only feature standardizer.
- `embeddings.npz`: aligned `z_flow` and deterministic reconstruction errors.
- `prototypes.npz`: normal centers, assignments, prototype distances, local scores.
- `entity_model.pt`: GRU, state schema, and train-only state standardizer.
- `spatial_model.pt`: reliability-aware context MLP and embedding standardizer.
- `calibration.json`: normal-only component scales, weights, and quantile thresholds.
- `test_scores.csv` and `test_scores_entities.csv`: segment/entity scores.
- `test_metrics.json`: metrics and scoring throughput.

The project intentionally reuses the baseline BPF-DAG PCAP/PCAPNG parser through
a read-only adapter (`data/pcap_reader.py`) and does not modify baseline code.
