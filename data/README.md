# Data layout

The recommended layout is explicit about the role and label of every capture:

```text
data/raw/
├── train/normal/*.pcap
├── calibration/normal/*.pcap
├── test/normal/*.pcap
└── test/attack/*.pcap
```

`data.sources` in the YAML maps each glob to a `split` and `label`. Alternatively,
use a CSV manifest with `pcap,label,split` columns. Relative manifest paths are
resolved relative to that CSV, matching the baseline's manifest behavior.

The training split may physically contain attack captures, but every training
entry point applies a second normal-label guard and excludes them. Calibration
also uses only normal segments. Test labels are read only for metrics.
