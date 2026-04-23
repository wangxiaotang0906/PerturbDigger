# PerturbDigger

PerturbDigger is a PyTorch implementation of perturbation prediction and mechanistic subgraph discovery on partially trusted prior graphs. The repository includes a lightweight demo dataset and a complete Adamson pipeline built around Reactome pathway structure.

## Repository layout

```text
PerturbDigger/
|- configs/
|  |- base.yaml
|  |- demo.yaml
|  `- adamson.yaml
|- data/
|  |- demo/
|  `- adamson/
|- scripts/
|  |- build_demo_data.py
|  |- prepare_adamson.py
|  |- train.py
|  `- explain.py
|- src/perturbdigger/
|  |- data/
|  |- explain/
|  |- graph/
|  |- model/
|  |- preprocess/
|  `- training/
|- adamson/
|- _database/pathways/Reactome/
|- requirements.txt
`- pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Demo pipeline

The demo dataset is small and can be regenerated at any time.

```bash
python scripts/build_demo_data.py --output data/demo
python scripts/train.py --config configs/demo.yaml
python scripts/explain.py --config configs/demo.yaml --checkpoint runs/demo/checkpoints/perturbdigger.pt
```

## Adamson pipeline

Required inputs:

- `adamson/perturb_processed.h5ad`
- `adamson/go.csv`
- `_database/pathways/Reactome/ReactomePathways.gmt`
- `_database/pathways/Reactome/ReactomePathways.txt`
- `_database/pathways/Reactome/ReactomePathwaysRelation.txt`

Prepare the processed dataset:

```bash
python scripts/prepare_adamson.py --config configs/adamson.yaml
```

Train the model:

```bash
python scripts/train.py --config configs/adamson.yaml
```

Export explanations from a trained checkpoint:

```bash
python scripts/explain.py --config configs/adamson.yaml --checkpoint runs/adamson/checkpoints/perturbdigger.pt
```

## Processed dataset format

```text
data/your_dataset/
|- genes.csv
|- pathways.csv
|- edges_gg.csv
|- edges_tg.csv
|- edges_gp.csv
|- edges_pp.csv
|- samples.csv
|- x.npy
|- y.npy
`- gene_metadata.npy
```

Expected columns:

- `genes.csv`: `gene`
- `pathways.csv`: `pathway`
- edge tables: `src`, `dst`
- `samples.csv`: `sample_id`, `split`, `is_control`, `condition`, `perturbed_genes`

`x.npy` stores reference expression and `y.npy` stores post-perturbation expression. Control rows may use `y = x`.
