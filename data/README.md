# Dataset

Put the Breast Cancer Wisconsin (Diagnostic) dataset here as:

```text
data/data.csv
```

The CSV is not included in this repository. The training code expects the original `data.csv` columns, including `id`, `diagnosis`, the 30 diagnostic features, and `Unnamed: 32`.

`id` and `Unnamed: 32` are removed before training.
