# GRACE: Graph-Reasoning Augmented Clinical rEport Generation (DeepEyeNet)

This repository contains a modular, publication-oriented implementation of **GRACE** for the DeepEyeNet dataset.

## 1) What Is Implemented

The implementation maps your proposed methodology to concrete modules:

- **MSVE-PAFP**: multi-scale visual encoder with pathology-aware feature pyramid and lesion context gating.
  - `grace/models/msve_pafp.py`
- **RD-KGE**: retinal disease graph encoder using relation-aware graph convolution.
  - `grace/models/rdkge.py`
  - graph construction: `grace/data/graph.py`
- **BCMA**: bidirectional cross-modal attention with learned fusion gate.
  - `grace/models/bcma.py`
- **UACD**: uncertainty-aware decoding with MC dropout and token entropy.
  - `grace/models/decoder.py`
  - integrated in `grace/models/grace_model.py`
- **CSCL (composite objective)**: weighted CE + semantic consistency + KG grounding.
  - `grace/losses.py`

Architecture diagram:
- `docs/ARCHITECTURE.md`

## 2) Design Choices and Justification

- **EfficientNetV2-S backbone** (`torchvision`) is selected for robust transfer from ImageNet with strong lesion-scale representation and Colab compatibility.
- **Top-down FPN + level-gating + learnable scale weights** preserves small and large retinal pathology cues.
- **R-GCN-style message passing** over relation-specific adjacency enables explicit disease-concept structure instead of flat keyword embeddings.
- **Bidirectional V<->G attention** prevents one-way semantic drift and improves concept-image alignment.
- **MC Dropout at inference** gives practical uncertainty without architectural overhead.
- **Composite loss** prioritizes clinical terms and graph-consistent statements, not only token overlap.

## 3) Evaluation Metrics

Standard linguistic metrics:
- BLEU-1/2/3/4
- ROUGE-L
- CIDEr-lite (TF-IDF cosine variant)

Clinical reliability metrics:
- **C3S**: Clinical Concept Coverage Score
- **CHR**: Clinical Hallucination Rate (lower is better)
- **UCS**: Uncertainty Calibration Score (1 - ECE)

Implemented in: `grace/metrics.py`

## 4) Generated Outputs (Publication-Ready)

Running training/evaluation produces:

- `outputs/test_metrics.json`
- `outputs/test_predictions.csv`
- `outputs/training_history.json`
- `outputs/metric_bar.png`
- `outputs/metric_radar.png`
- `outputs/calibration_curve.png`
- `outputs/uncertainty_hist.png`
- `outputs/training_curves.png`
- `outputs/knowledge_graph_snapshot.png`

## 5) Local Run

```bash
pip install -r requirements.txt
python scripts/train_grace.py --config grace/configs/default.yaml
```

Edit `grace/configs/default.yaml` to point to your dataset root where `train.csv`, `valid.csv`, `test.csv` and image paths are valid.

Quick local example with explicit paths:

```bash
python scripts/check_dataset.py \
  --dataset-root /Users/fs525/Desktop/PhD/2026/March/Deepeyenet/DeepEyeNet \
  --train-csv train.csv --valid-csv valid.csv --test-csv test.csv

python scripts/train_grace.py \
  --config grace/configs/local_deepeyenet_example.yaml
```

Or without editing config:

```bash
python scripts/train_grace.py \
  --config grace/configs/default.yaml \
  --dataset-root /Users/fs525/Desktop/PhD/2026/March/Deepeyenet/DeepEyeNet \
  --train-csv train.csv --valid-csv valid.csv --test-csv test.csv \
  --output-dir /Users/fs525/Desktop/PhD/2026/March/Deepeyenet/outputs
```

## 6) Google Colab Run

Use notebook:
- `notebooks/GRACE_DeepEyeNet_Colab.ipynb`

It mounts Drive, installs dependencies, sets `dataset_root` to your Drive folder, trains, and renders all key charts.

## 7) GitHub End-to-End Workflow

Yes, you can put this project on GitHub and run fully end-to-end after cloning.

```bash
git clone <your_repo_url>.git
cd Deepeyenet
pip install -r requirements.txt

# Optional but recommended: verify dataset first
python scripts/check_dataset.py --dataset-root /path/to/DeepEyeNet

# Train + evaluate end-to-end
python scripts/train_grace.py \
  --config grace/configs/default.yaml \
  --dataset-root /path/to/DeepEyeNet \
  --output-dir ./outputs
```

## 8) Recommended Paper Sections Enabled by This Code

- Architecture diagram and module-wise rationale (GRACE vs M3T)
- Ablation discussion via loss terms and graph module toggling (extend configs)
- Reliability analysis (UCS + calibration curves)
- Clinical faithfulness analysis (C3S and CHR)
- Qualitative case studies from `test_predictions.csv`

## 9) Notes

- This implementation uses a lightweight lexical concept extraction for C3S/CHR by default for reproducibility.
- If needed, replace extractor with medical NER (e.g., SciSpacy/UMLS linker) without changing the training pipeline API.
