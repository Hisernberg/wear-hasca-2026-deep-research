---
license: mit
task_categories:
- time-series-classification
tags:
- human-activity-recognition
- wearable
- egocentric-vision
- kaggle
- har
- hasca-2026
- wear-dataset
- videomae
- imu
size_categories:
- n<1K
---

# WEAR @HASCA 2026 — Deep Research Dossier (3rd WEAR Dataset Challenge)

Research & analysis artifacts for the **3rd WEAR Dataset Challenge @HASCA 2026** (Kaggle):
multimodal Human Activity Recognition from 1-second inertial windows (50 Hz, one random limb) +
VideoMAEv2-Base egocentric video features, evaluated by macro F1 over 19 activity classes
(12,234 shuffled test windows).

## What's inside

| File | Content |
|---|---|
| `README.md` (on GitHub) | TL;DR of 12 key findings + public LB scoreboard |
| `docs/01-my-submission-analysis.md` | Our (udaken10 / team "really?") 6-notebook trajectory, diff-by-diff, what hurt/helped, LB 0.90759 anatomy |
| `docs/02-competitor-deep-dive.md` | All 29 public notebooks dissected: architectures, recipes, post-processing, scores, bugs; the 0.92+ "five pillars" |
| `docs/03-external-research.md` | WEAR 2024/2025 winner reports (Signal Sleuths, 3KA, HARMA), SHL/CMI/Ion-Switching playbooks, fusion SOTA, LLM-for-HAR evidence |
| `docs/04-techniques-matrix.md` | 45+ techniques per layer: helps ✅ / hurts ❌ / risky ⚠️ with measured deltas |
| `docs/05-bugs-setbacks-lessons.md` | Bug hall of fame: official video bug, label lags, fake timelines, CV leaks + 10 distilled lessons |
| `docs/06-improvement-plan.md` | Multi-track plan: algorithmic (links/counts/boundaries/oracle-fusion), LLM-enabled, engineering, publication |
| `docs/07-roadmap.md` | 4-week execution roadmap + validation protocol + risk register |
| `artifacts/leaderboard_public_top97.csv` | Full public LB snapshot |
| `artifacts/discussions/` | All 5 competition threads (full text) |
| `artifacts/notebooks-index.md` | Every public notebook + score + one-line summary |

## Key findings (TL;DR)

1. Sequence reconstruction (learned links → Hungarian 1:1 → cycle-break) is worth ≈ **+0.07 LB alone**, +0.18–0.20 with count calibration — the dominant lever.
2. Window-level accuracy (~0.725 OOF macro-F1) is NOT the bottleneck; post-processing reaches ~0.93.
3. Per-subject Sinkhorn count calibration (≈97 tiles/exercise, null floor) fixes the null-painting pathology.
4. Video's value flows through geometry (kNN graphs, boundary linking), not classification logits.
5. The video–inertial oracle-late-fusion gap (91–94 vs ~75 realizable, WEAR paper) is the biggest unclaimed headroom.
6. Test-like OOF validation + transductive per-subject statistics + ensembling discipline separate the 0.92+ cluster from the 0.61–0.75 graveyard.

## Provenance & citation

- Competition: M. Bock, "3rd WEAR Dataset Challenge @HASCA 2026", Kaggle 2026.
- Dataset paper: Bock et al., "WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition", IMWUT 8(4), 2024 (arXiv:2304.05088).
- Public notebooks analyzed with full credit to their authors (see artifacts/notebooks-index.md).
- Prepared by [@Nabidnur](https://huggingface.co/Nabidnur) — companion GitHub repo: https://github.com/Hisernberg/wear-hasca-2026-deep-research
