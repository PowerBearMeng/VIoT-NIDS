# V5 Flow-level Video-IoT NIDS

This is the independent V5 implementation of `DesignV5.md`. It reuses the
repository's streaming BPF-DAG PCAP/PCAPNG reader adapter and Gotham packet
label stream. It does not modify the V1–V4 pipelines.

Run from `/home/mfh/Desktop/test/myModel` with the existing `wxy` environment:

```bash
PY=/home/mfh/miniconda3/envs/wxy/bin/python
$PY -m viot_nids extract --pcap /path/to/benign.pcap --out outputs/v5/benign_flows.csv
$PY -m viot_nids train --flows outputs/v5/benign_flows.csv --model-dir outputs/v5/model --config configs/v5.yaml
$PY -m viot_nids inspect --model-dir outputs/v5/model
$PY -m viot_nids score --pcap /path/to/capture.pcap --model-dir outputs/v5/model --out outputs/v5/scores.csv
$PY -m viot_nids evaluate --scores outputs/v5/scores.csv --labels /path/to/flow_labels.csv --out outputs/v5/evaluation
```

For a labeled Gotham attack capture, `extract --packet-labels <existing
packet_labels.csv> --labels-out <flow_labels.csv>` produces 1 s directional
Flow labels under the repository's existing rule: a Flow is positive if any
constituent eligible packet has `binary_label=1`. The `.flow_labels.csv` in
Gotham describes capture-long bidirectional flows, so its rows cannot be
directly joined to V5's directional 1 s slices. Packet labels are used only
to write a separate evaluation file.

Evaluate all Access-OVS attack captures with:

```bash
$PY -m viot_nids benchmark \
  --attack-root /home/mfh/Desktop/test/gotham-iot-testbed/artifacts/attack \
  --model-dir outputs/v5/model --out outputs/v5/attack
```

`benchmark` saves per-capture scores, labels, AUROC/AUPRC/EER, ROC and PR
curves, as well as pooled metrics. It removes each intermediate feature CSV
after scoring, retaining the original PCAP and the reviewable results.

The training set is split chronologically by second. The last 20% is used
only for empirical score calibration. Training uses uniformly selected
channels and truncated length-8 sequences; inference processes complete
channel histories. `channel_fields` in the YAML controls temporal identity.
No attack labels enter fitting, calibration, role selection, or scoring.
The config applies a 0.05 minimum log-feature scale after RobustScaler fitting
because microsecond IAT IQRs otherwise produce six-digit normalized values.

The parser's `wire_length` is IPv4 `ip.len` clamped to 1–1500, as in the
existing adapter. TCP flags and payload are not model inputs. EER is the
closest observed FPR/FNR operating point and is not a deployment threshold.
