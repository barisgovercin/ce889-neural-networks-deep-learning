# Neural Networks and Deep Learning

**Module:** CE889 — Neural Networks and Deep Learning
**Student:** Muhammed Baris Govercin
**Registration No:** 2501385

This repository contains two projects completed for CE889.

---

## 1. Lunar Lander — Neural Network Controller (`lunar-lander/`)

A feed-forward neural network that learns to fly and land a lunar lander
in a Pygame environment. Training data is collected by playing the game
manually; the network then predicts the lander's control outputs from its
state (position / velocity relative to the landing pad).

**Pipeline**

1. `DataCollection.py` — record state/action pairs while playing
2. `Normalisation.py` / `scaler.py` — z-score normalise the collected data
3. `TrainNetwork.py` — train the network with backpropagation
4. `NeuralNetHolder.py` / `weights.py` — load the trained weights and drive the lander in-game
5. `Main.py` — run the game (manual or neural-network controlled)

```bash
cd lunar-lander
pip install -r requirements.txt
python Main.py
```

The `ce889_data*.csv` files contain the collected and normalised training data,
and `docs/` holds the write-ups (data collection, normalisation, training).

---

## 2. Rossmann Store Sales Forecasting (`rossmann/`)

A regression project predicting daily sales for Rossmann drug stores
(Kaggle "Rossmann Store Sales"). The notebook covers data cleaning,
feature engineering (store, promo, holiday, date features), model training,
and generation of a Kaggle submission.

- `rossmann.ipynb` — full analysis and modelling notebook
- `submission.csv` — generated predictions
