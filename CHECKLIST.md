# CHECKLIST.md — Challenge 6, Group 4

## Dataset
- Name: NCES CCD Public Elementary/Secondary School Universe Survey SY 2022-23
- Source: https://nces.ed.gov/ccd/files.asp
- Records: 95,332 schools | Features: 11 (same as Challenge 5)

## Model Architectures

### AutoEncoder (AE)
- Architecture: 11 → 128 → 64 → 16 (latent) → 64 → 128 → 11
- Activations: ReLU (hidden), Identity (output)
- Optimizer: Adam, lr=1e-3 | Epochs: 100 | Batch size: 256
- Trained on: normal schools only (top 5% ISO anomalies excluded)

### Variational AutoEncoder (VAE)
- Architecture: 11 → 128 → 64 → [mu, logvar](16) → 64 → 128 → 11
- Loss: MSE + β·KL | β=1.0 | KL warmup over 30 epochs
- Optimizer: Adam, lr=1e-3 | Epochs: 100 | Batch size: 256

### Isolation Forest
- n_estimators: 200 | contamination: 0.05 | random_state: 42

## Anomaly Thresholds
- AE:  95th percentile of training errors = 0.002180 → rate: 8.0%
- VAE: 95th percentile of training errors = 0.554322 → rate: 7.4%
- ISO: 95th percentile of training scores  → rate: 9.7%

## Spearman Rank Correlations
- AE vs VAE:  ρ = 0.3430
- AE vs ISO:  ρ = 0.7216
- VAE vs ISO: ρ = 0.4062

## Silhouette Scores (C5 cluster labels as reference)
- Raw feature space: 0.2280
- AE latent space:   0.1872
- VAE latent space:  0.2818

## Cross-Challenge Synthesis (max 200 words)
Challenge 5 (clustering) revealed 6 structural archetypes in U.S. public schools
primarily driven by racial/ethnic composition and poverty level. Challenge 6 (AE/VAE)
adds two complementary insights: (1) anomaly detection identifies schools that deviate
from their archetype peers — these anomalous schools tend to be small, have extreme
demographic compositions, or unusual student-teacher ratios, and may represent
under-resourced high-performers or data quality issues; (2) the VAE latent space
provides a smooth, continuous manifold of school characteristics where the C5 cluster
boundaries become visible as soft transitions rather than hard partitions, revealing
that some clusters are genuinely distinct while others blend gradually. The Isolation
Forest agrees moderately with the AE (ρ=0.722), confirming that deep
reconstruction captures anomaly signal beyond what tree-based isolation detects.
Together, the three challenges provide a complete picture: supervised signal extraction
(C2), structural segmentation (C5), and anomaly/representation discovery (C6).

## Seeds
- Python/NumPy: 42 | PyTorch: 42 | AE seeds: 42, 7, 123 | VAE seeds: 42, 7, 123
