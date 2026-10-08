# Public Notebook Index (all 29, pulled via Kaggle API Oct 2026)

| Notebook (author) | Public LB | Votes | Approach in one line |
|---|---:|---:|---|
| goodpjw2008/wear-hasca-learned-links-counts | **0.92942** | 19 | Base stack + L3 two-tower/SSL links + bagging + learned counts + adapted T + chain refiner (best public) |
| yeashusemwal/ranked-timelines-boundary-refiner | 0.92875 | 4 | Artifacts-only: LambdaRank links, 8 ranked timelines, whitened graph, consensus refiner, LB log |
| honghanhhh/wear-hasca-lb-0-9 | 0.92485 | 27 | Base stack + soft-bout L3 + 15 baggings + 3-pass counts + gated null specialist |
| akhyar2612/lb-0-891-timeline-graph ≡ woominyo/lb-0-890 ≡ **udaken10 timeline family (ours)** | 0.890–0.90759 | 0–6 | "Briano" Timeline+Graph base (LGBM links → Hungarian → graph → Sinkhorn-97) |
| jiweiliu/public-fast-gpu-inference | 0.90770 | 8 | Weights-only port of base stack; prepared-input caches, dual-GPU |
| avikdas567/multimodal-videomae-inertial-cross-attention-har | (n/a) | 21 | Cross-attention fusion net; single fold, no post-processing (window ≈0.67) |
| nomannic19/ts-emb-3wdc-temporal-fusion-ensemble | 0.61278 | 13 | Tiny fusion models, no CV, no sequence machinery; sensor-geometry mismatch |
| swathia/aligned-video-inertial-fusion | 0.67126 | 0 | "Timeline" built on shuffled `id` → no-op smoothing |
| sibamsamanta07/wear-hasca-improved | 0.65265 | 4 | Hand-crafted features + hierarchical LGBM; no deep/sequence |
| akhyar2612/wear-hasca-chain-lb-0-75 | 0.71512 | 4 | 3 experts + cosine chain + duration bands (pre-Sinkhorn pathology) |
| akhyar2612/wear-hasca-hungarian-chain-viterbi-lb-0-72 | 0.72916 | 0 | Same + Hungarian assignment, no count calibration |
| evelynyang02/wear-hasca-2026-baseline-v1-repro | — | 2 | LGBM + left-limb mirroring trick + transductive PCA |
| lakhindarpal/3rd-wear-dataset-challenge | ~0.5s | 8 | CatBoost on 12 stats; probe-selection leakage |
| hmnshudhmn24/3rd-wear-dataset-challenge | ≤0.4 | 6 | RandomForest on right-arm-only means (ignores random-limb design) |
| stmugiwara/wear-{dummy,cnn-lstm-v3,v5,v6,v7} | ≤0.05 | 0–1 | Dummy copy / syntax error / 18-class head / string labels |
| binasalama/wear-hasca-baseline | — | 2 | (Inception+Transformer direction, per discussion thread) |
| swathiha/eda-temporal-attention-inertial-activity | — | 1 | EDA + temporal attention experiments |
| abhinavm2811/3rd-wear-dataset-challenge | — | 1 | Baseline variant |

Full sources: `wear_research/notebooks/text/` (local). The three top notebooks' full dissection: [docs/02](../docs/02-competitor-deep-dive.md).
