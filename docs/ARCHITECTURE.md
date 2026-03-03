# GRACE Architecture (Implemented)

```mermaid
flowchart LR
    A[Retinal Image] --> B[MSVE-PAFP]
    K[Keyword Sequence] --> G[RD-KGE]
    B --> C[BCMA V↔G Fusion]
    G --> C
    C --> D[UACD Transformer Decoder]
    D --> E[Clinical Report]
    D --> U[Token Entropy Uncertainty]
    E --> M[BLEU ROUGE CIDEr-lite]
    E --> C3[C3S]
    E --> H[CHR]
    U --> Q[UCS Calibration]
```

## Module-to-File Map

- MSVE-PAFP: `grace/models/msve_pafp.py`
- RD-KGE: `grace/models/rdkge.py`
- BCMA: `grace/models/bcma.py`
- UACD: `grace/models/decoder.py`
- End-to-end model: `grace/models/grace_model.py`
- Composite losses: `grace/losses.py`
- Metrics: `grace/metrics.py`
- Training + evaluation + figures: `grace/train.py`
