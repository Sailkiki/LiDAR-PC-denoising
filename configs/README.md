# Configurations

The repository provides from-scratch training configurations.  Data-split
records are supplied by the user and are not included in this code-only
release.

| Config | Purpose |
|---|---|
| `v2_base.yaml` | base model, weighted Huber objective, from scratch. |
| `v2_wide.yaml` | wide model, weighted Huber objective, from scratch. |
| `v2_plain.yaml` | ablation: plain Huber without RBF/composite weights. |
| `v2_wide_hn.yaml` | wide model trained with stronger noise. |

Training a configuration writes checkpoints, logs, and validation history to
`runs/<run_name>/`.  These generated artefacts are not included in the
repository.

Example:

```bash
bash scripts/train.sh 0 configs/v2_wide.yaml
```
