# Draw Hunter 3000 ⚽
### Football Draw Prediction Model — European Leagues

A machine learning model specialised in predicting draw outcomes 
across five European football leagues.

## Overview
Draws are the hardest outcome to predict in football — occurring 
in roughly 26% of matches but frequently missed by standard models 
that optimise for accuracy. This model is specifically tuned to 
identify draws.

## Data
- Source: football-data.co.uk (free, historical match data)
- Leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1
- Season: 2023/24
- Total matches: 1,752

## Model
- Algorithm: Random Forest Classifier (200 trees)
- Class balancing: Applied to handle draw minority class
- Recommended threshold: 0.45

## Results
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.683 |
| Draw Recall @ 0.45 | 77% |
| Draw Precision @ 0.45 | 37% |
| Baseline draw rate | 26% |

## Top Predictors
1. Half-time goal difference
2. Home shot accuracy
3. Home win odds (B365H)
4. Away win odds (B365A)
5. Away shot accuracy

## Key Finding
A match level at half time is the single strongest 
signal for a draw outcome.

## Files
- Football_Draw_Prediction_Model.ipynb
- draw_hunter_model.pkl
- draw_hunter_scaler.pkl
- feature_cols.json

## Author
Ruth Mutile Kimeu | github.com/mutilekim
