# Neural Networks and Deep Learning

**Module:** CE889 — Neural Networks and Deep Learning
**Student:** Muhammed Baris Govercin
**Registration No:** 2501385

This repository contains two projects completed for CE889.

---

## 1. Lunar Lander — Neural Network Controller (`lunar-lander/`)

A multilayer perceptron **implemented from scratch in pure Python** (no
NumPy, no ML frameworks) that learns to fly and land a lunar lander in a
Pygame environment. Training data is collected by playing the game manually;
the network then maps the lander's position relative to the landing pad to
the control velocities.

**Network architecture**

| Property | Value |
|----------|-------|
| Topology | `2 → 3 → 2` (input → hidden → output) |
| Inputs | normalised `(x, y)` offset to the landing pad |
| Outputs | `(velocity_x, velocity_y)` |
| Hidden activation | sigmoid `1 / (1 + 2⁻ˣ)` |
| Output activation | linear |
| Training | backpropagation with momentum |
| Learning rate | 0.01 |
| Momentum | 0.6 |
| Epochs | 111 |
| Weight init | uniform `U(−0.5, 0.5)`, custom LCG RNG (seed 1234567) |
| Loss | mean squared error |

The hyperparameters mirror an equivalent MATLAB reference setup
(`trainParam.lr / .mc / .epochs`). Inputs are **min-max normalised** to
`[0, 1]` using statistics stored in `scaler.py`.

**Pipeline**

1. `DataCollection.py` — record state/action pairs while playing
2. `Normalisation.py` / `scaler.py` — min-max normalise the collected data
3. `TrainNetwork.py` — train via backprop + momentum, export `weights.py`
4. `NeuralNetHolder.py` — load `weights.py` + `scaler.py` and drive the lander
5. `Main.py` — run the game (manual or neural-network controlled)

```bash
cd lunar-lander
pip install -r requirements.txt
python TrainNetwork.py      # trains and writes weights.py
python Main.py              # play / watch the trained controller fly
```

`docs/` holds the write-ups (data collection, normalisation, training).

---

## 2. Rossmann Store Sales Forecasting (`rossmann/`)

A regression project predicting daily sales for Rossmann drug stores
(Kaggle "Rossmann Store Sales"), using a **PyTorch feed-forward neural
network**.

**Workflow** — merge the train and store tables, engineer date features,
drop unused columns, label-encode categoricals, `StandardScaler` the numeric
features, then train on **813,767 × 19** samples with a `StepLR` schedule.

**Training (15 epochs, MSE loss)**

| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|---------|-----|
| 1 | 5.7134 | 0.1604 | 7.0e-4 |
| 5 | 0.0413 | 0.0422 | 3.5e-4 |
| 10 | 0.0338 | 0.0357 | 8.7e-5 |
| 15 | **0.0317** | **0.0337** | 2.2e-5 |

The learning rate is stepped down across training and validation loss tracks
training loss closely (no significant overfitting). Predictions are written
to `submission.csv` in Kaggle format.

- `rossmann.ipynb` — full analysis, modelling, and training notebook
- `submission.csv` — generated predictions
