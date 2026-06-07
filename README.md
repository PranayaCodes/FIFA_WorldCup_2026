# FIFA_WorldCup_2026
# FIFA World Cup 2026 Winner Prediction 🏆

A machine learning project that simulates the 2026 FIFA World Cup bracket and predicts win probabilities for each team using historical international football data.

## Project Overview
This project builds an end-to-end ML pipeline that:
- Trains on 10 years of international football match data (2016–2026)
- Engineers features like rolling form, head-to-head records, and goal averages
- Uses XGBoost and Poisson regression to predict match outcomes
- Runs 10,000 Monte Carlo simulations to estimate each team's chance of winning the World Cup
- Visualises results in an interactive Streamlit dashboard

## Project Structure
Data/                     → Raw datasets (download from Kaggle)
Cleaned_Data/             → Filtered and processed data
Feature_Engineering/      → Feature engineering notebooks
Model_Training/           → ML model training notebooks
Evaluate/                 → Model evaluation and backtesting
Monte_Carlo_Simulation/   → Bracket simulation
StreamLite_Dashboard/     → Streamlit web app

## Dataset
Download from Kaggle and place in `Data/`:
- [International Football Results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

Files needed: `results.csv`, `goalscorers.csv`, `shootouts.csv`

## Tech Stack
Python · pandas · scikit-learn · XGBoost · SciPy · matplotlib · Streamlit

## Status
🚧 In progress
