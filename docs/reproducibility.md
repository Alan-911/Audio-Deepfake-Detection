# Reproducibility and evaluation protocol

This file describes the evidence required to reproduce a speech deepfake classification result. The repository does not include the challenge dataset or trained model checkpoints.

## Experiment record (fill for every published run)

| Field | Value |
| --- | --- |
| Git commit SHA | To be recorded |
| Dataset and authorized access source | To be recorded |
| Dataset version/checksum | To be recorded |
| Five-class mapping | To be recorded |
| Speaker-disjoint validation/test split | To be verified and recorded |
| Random seeds | To be recorded |
| Python, CUDA, PyTorch, transformers versions | To be recorded |
| Encoder checkpoint and revision | To be recorded |
| Optimizer / learning-rate schedule / batch size | To be recorded |
| Compute hardware | To be recorded |
| Evaluation script and command | To be recorded |
| Macro-F1 / EER and class-wise results | To be recorded with artifacts |

## Evaluation hygiene

- Use the dataset provider's official evaluation split and scoring protocol.
- Keep speakers, sessions and recording conditions isolated where the benchmark requires it.
- Do not use the held-out test set to select hyperparameters.
- Record performance by class as well as aggregate macro-F1.
- Provide confidence intervals or variation across seeds when possible.
- Distinguish local experimental results from official challenge-leaderboard scores.
- Do not commit data, recordings or checkpoints without permission.
- Make run artifacts available only when their licenses permit redistribution.

## Engineering follow-up

The lightweight GitHub Actions workflow currently checks Python syntax and essential research file presence **without downloading large models or datasets**. GPU integration tests, model shape tests, data schema checks and a reproducible benchmark pipeline remain future work.
