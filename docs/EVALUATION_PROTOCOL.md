# GRACE Evaluation Protocol (DeepEyeNet)

## Goals

1. Measure text quality and clinical correctness jointly.
2. Quantify hallucination and uncertainty calibration.
3. Provide reproducible visual analytics for publication.

## Core Metrics

- Text overlap: BLEU-1/2/3/4, ROUGE-L, CIDEr-lite.
- Clinical content: C3S.
- Safety: CHR.
- Reliability: UCS (1 - ECE).

## Statistical Reporting

- Report mean across full test set.
- For final paper, run 3-5 seeds and report mean +- std for each metric.
- Use paired bootstrap confidence intervals for BLEU-4, C3S, CHR, UCS.

## Figures Produced Automatically

1. `metric_bar.png`: aggregate metric comparison.
2. `metric_radar.png`: clinical-linguistic balance profile.
3. `calibration_curve.png`: confidence vs empirical accuracy.
4. `uncertainty_hist.png`: entropy distribution.
5. `training_curves.png`: optimization dynamics.
6. `knowledge_graph_snapshot.png`: graph reasoning substrate.

## Suggested Extra Tables For Journal Submission

- Error taxonomy table: omission vs hallucination vs severity mismatch.
- Per-pathology performance table (stratified by keyword groups).
- Uncertainty-stratified accuracy table (low/medium/high entropy buckets).

## Reproducibility Checklist

- Fix random seed in config.
- Pin package versions (`requirements.txt`).
- Keep train/valid/test splits unchanged.
- Archive `default.yaml`, metrics JSON, and predictions CSV with the manuscript.
