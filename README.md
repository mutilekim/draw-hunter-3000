# Draw Hunter 3000 ⚽

### A Specialised Random Forest Classifier for Predicting Football Draws

A machine learning project that solves one of the hardest problems in sports analytics: identifying draw outcomes in football matches. Built with Python, pandas, and scikit-learn across five European leagues.

**Author:** Ruth Mutile Kimeu
**Project type:** Personal machine learning project
**Case study:** [Read the full write up](https://mutilekim.github.io/draw-hunter-3000.html)

---

## Why this project exists

In football analytics, draws are the minority outcome. They occur in roughly 26 percent of matches across European leagues. Standard classification models optimise for overall accuracy, which means they learn to predict the majority outcomes (home wins and away wins) while almost completely ignoring draws.

A naive model that predicts "Home Win" for every match achieves around 46 percent accuracy on EPL data but catches zero draws. That is the failure mode this project was built to solve.

**Draw Hunter 3000** treats draw detection as a specialised classification task rather than a general prediction problem. It uses class balancing, recall optimised evaluation, and threshold tuning to identify draws that standard models miss.

---

## Results

| Metric | Score |
|--------|-------|
| AUC ROC | 0.683 |
| Draw recall at threshold 0.45 | 77% |
| Draw precision at threshold 0.45 | 37% |
| Baseline draw rate in data | 26% |
| Matches analysed | 1,752 |
| Leagues covered | 5 |

The model correctly identifies 77 percent of actual draws when the decision threshold is tuned to 0.45. Precision of 37 percent represents a 1.4x improvement over the 26 percent random baseline.

---

## Data

- **Source:** football-data.co.uk (free, historical match data)
- **Leagues:** English Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- **Season:** 2023 to 2024
- **Total matches:** 1,752
- **Draws in dataset:** 463 (26.4 percent)

Match level data was loaded directly from public URLs using pandas, combined into a single dataframe, and validated for missing values and consistency before modelling.

---

## Methodology

### 1. Exploratory data analysis

Analysed draw distribution across the five leagues. Confirmed that draws represented 26.4 percent of all matches, making them the minority class. Verified that standard accuracy metrics would be misleading and that recall focused evaluation was required.

### 2. Feature engineering

Created new predictive features from raw match statistics, including:

- Half time goal difference
- Home and away shot accuracy ratios
- Implied draw probability from betting odds (1 divided by B365D)

A final feature set of 21 variables was selected, covering shots, corners, fouls, cards, and bookmaker odds.

### 3. Model training and comparison

Three algorithms were trained and evaluated on draw detection performance, not overall accuracy. Class weight was set to "balanced" across all models to account for the draw minority class. A stratified 80 to 20 train and test split maintained draw proportions in both sets.

| Model | Draw Recall | Draw Precision | Overall Accuracy | Decision |
|-------|-------------|----------------|------------------|----------|
| Logistic Regression | 59% | 32% | 56% | Rejected |
| **Random Forest** | **66%** | **39%** | **64%** | **Selected** |
| Gradient Boosting | 10% | 29% | 70% | Rejected |

*Results in this table use the default 0.5 decision threshold. After threshold tuning to 0.45 (described in step 4), Random Forest reaches 77 percent recall at 37 percent precision.*

#### Why Gradient Boosting was rejected despite higher accuracy

Gradient Boosting achieved 70 percent overall accuracy, the highest of the three models. But it only caught 10 percent of actual draws. A model that misses 9 out of every 10 draws is useless for draw prediction regardless of how well it predicts home and away wins. This is exactly the failure mode the project was designed to avoid.

### 4. Threshold optimisation

Generated precision recall curves across decision thresholds from 0.30 to 0.60. Selected 0.45 as the optimal balance, catching 77 percent of draws at 37 percent precision.

### 5. Model persistence

The trained model, fitted scaler, and feature list were serialised using joblib and JSON, allowing prediction on new match data without retraining.

---

## Top predictive features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | Half time goal difference | 0.150 | Matches level at half time are far more likely to end as draws |
| 2 | Home shot accuracy | 0.069 | Clinical home teams convert chances and avoid dropped points |
| 3 | Home win odds (B365H) | 0.068 | High home win odds signal competitive matches where draws are plausible |
| 4 | Away win odds (B365A) | 0.064 | Bookmaker odds encode rich pre match information |
| 5 | Away shot accuracy | 0.056 | Wasteful away teams struggle to win, raising draw probability |

The strongest single signal is the half time score line. This aligns with football intuition: matches level at half time are structurally more likely to remain level.

---

## Repository contents

```
draw-hunter-3000/
├── Football_Draw_Prediction_Model.ipynb   # Full notebook with EDA, modelling, and evaluation
├── draw_hunter_model.pkl                  # Trained Random Forest model (joblib serialised)
├── draw_hunter_scaler.pkl                 # Fitted StandardScaler
├── feature_cols.json                      # Feature list for new predictions
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## How to use this project

### Run the notebook

```bash
# Clone the repository
git clone https://github.com/mutilekim/draw-hunter-3000.git
cd draw-hunter-3000

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Football_Draw_Prediction_Model.ipynb
```

### Use the saved model for predictions

```python
import joblib
import json
import pandas as pd

# Load the model and supporting files
model = joblib.load("draw_hunter_model.pkl")
scaler = joblib.load("draw_hunter_scaler.pkl")
with open("feature_cols.json") as f:
    feature_cols = json.load(f)

# Prepare new match data with the same 21 features
new_match = pd.DataFrame([...])  # Same feature columns as training
new_match_scaled = scaler.transform(new_match[feature_cols])

# Predict draw probability
draw_probability = model.predict_proba(new_match_scaled)[:, 1]
draw_prediction = (draw_probability >= 0.45).astype(int)
```

---

## Tech stack

- **Language:** Python 3
- **Core libraries:** pandas, NumPy, scikit-learn
- **Visualisation:** matplotlib, seaborn
- **Model persistence:** joblib, JSON
- **Environment:** Google Colab and local Jupyter

---

## Limitations and future work

This is a first iteration of a draw specialised model. Known limitations and planned improvements include:

1. **Single season training data.** The model was trained on 2023 to 2024 data only. Multi season training would improve generalisation and capture longer term team form patterns.
2. **No team specific features yet.** The current model relies on match level statistics. Adding team level features (recent form, head to head history, home and away splits) is the highest priority next step.
3. **Threshold is fixed.** A future version could adapt the decision threshold per league, since draw rates vary across the five leagues.
4. **No live data integration.** The current pipeline runs on historical CSVs. A future version could pull live fixtures and odds through an API for prediction on upcoming matches.
5. **No probability calibration.** Predicted probabilities are not calibrated. Adding Platt scaling or isotonic regression would make probability estimates more reliable.

---

## What I learned

This project taught me several lessons that go beyond the textbook view of classification:

1. **Accuracy is the wrong metric when classes are imbalanced.** A 70 percent accuracy model that misses 90 percent of the minority class is worse than a 64 percent accuracy model that catches two thirds of it. Choosing the right metric is the entire game.
2. **Threshold tuning is underrated.** Most tutorials stop at model selection. Moving the decision threshold from 0.5 to 0.45 lifted recall from 66 percent to 77 percent on the same model. That is free performance most people leave on the table.
3. **Domain knowledge improves feature engineering.** Half time goal difference is the single strongest predictor of a draw. That is not something a model can discover on its own. It came from thinking about how football matches actually behave.
4. **Compare models on the right axis.** Gradient Boosting "won" on overall accuracy but lost decisively on the metric that mattered. Defining success before training is essential.

---

## Case study

For a full write up with methodology, model comparison, and feature analysis, see the project case study:

[Draw Hunter 3000 case study on my portfolio](https://mutilekim.github.io/draw-hunter-3000.html)

---

## Author

**Ruth Mutile Kimeu**
Data Scientist, Product Builder, and Founder of BriefDesk Solutions
Nairobi, Kenya

- Portfolio: [mutilekim.github.io](https://mutilekim.github.io)
- GitHub: [github.com/mutilekim](https://github.com/mutilekim)
- LinkedIn: [Ruth Mutile Kimeu](https://www.linkedin.com/in/ruth-kimeu-32a977142/)
- Email: hello@briefdeskke.co.ke

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

*Built as part of ongoing machine learning practice during the ALX Africa Data Science programme (Cohort 10, Machine Learning Track) and a BSc in Data Science at the Open University of Kenya.*
