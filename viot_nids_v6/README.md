# V6 Video-IoT NIDS

V6 freezes the benign-trained V5 Flow Encoder, Temporal Memory, and
role/regime models. V6 **trains new benign-only models** for context within
`(source host, role, regime, second)` cohorts and for communication-relation
familiarity. The V5 model weights and statistics are copied into the V6 model
directory, so the saved V6 artifact is self-contained.

The context features are the ten V5 count/volume/diversity summaries,
computed for each regime cohort. Cohorts with adequate benign training and
calibration support receive a role-regime GMM. Sparse cells fall back to a
role-level or global cohort model.

For each directional Flow, relation familiarity is the greater of the
benign-training counts for its source port and destination port within the
same `(source host, destination peer, protocol)` pair. Either endpoint can
hold the persistent service port. A previously unseen pair backs off to
normal-training port frequencies for its protocol: a familiar service remains
possible on a new host, while unfamiliar ports can still carry evidence.
No raw IP or port value enters the neural Local model.

Local, cohort, and relation raw scores are independently converted to
upper-tail evidence with the first half of a held-out benign time segment.
The three evidence values are added without attack-label-tuned weights.
`final_score` is the continuous evidence sum used for ranking and AUROC.
`final_percentile` is a held-out benign reference percentile for display;
its finite empirical resolution creates ties, so it is not used for AUROC.
Attack labels enter only the evaluation command.

```bash
/home/mfh/miniconda3/envs/wxy/bin/python -m viot_nids_v6 train \
  --benign-flows outputs/v5/benign_flows.csv \
  --base-model outputs/v5/model_scaled \
  --model-dir outputs/v6/model_final --config configs/v6.yaml

/home/mfh/miniconda3/envs/wxy/bin/python -m viot_nids_v6 benchmark \
  --attack-root /home/mfh/Desktop/test/gotham-iot-testbed/artifacts/attack \
  --model-dir outputs/v6/model_final --out outputs/v6/attack_final \
  --v5-cache outputs/v5/attack_scaled
```

The optional V5 cache reuses exactly aligned, frozen local embeddings and
labels from an earlier evaluation. V6 still re-extracts the original PCAP
to obtain packet and byte counts and checks every Flow's time and five-tuple
against the cache. Without the cache, it embeds the Flow features directly.

Exact port familiarity is highly predictive in this testbed and may be
specific to its fixed topology or generated attacks. Treat it as an
experimental relation feature, not as evidence of cross-deployment
generalization. Evaluate on independent benign port/service changes before
operational use.
