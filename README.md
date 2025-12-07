# California Housing Price Prediction

This project trains a couple of machine learning models to predict median house values in California. It uses the California Housing dataset from scikit-learn, which includes features like income, house age, rooms per household, population, and location (lat/long).

The goal was to build a small end-to-end pipeline: load data → clean it → train models → evaluate → generate a few plots.

## Dataset

To get the dataset, run the small script inside the `scripts` folder: python scripts/grabdata.py (
This downloads the California housing data and saves it to `data/housing.csv`), or or produce your own.

## Project Structure
data/
housing.csv
housing_processed.csv
X_test.csv
y_test.csv

scripts/
grabdata.py
preprocess.py
train_model.py
evaluate_model.py

outputs/
model_lr.pkl
model_rf.pkl
scaler.pkl
label_encoders.pkl
plots/
predicted_vs_actual.png
feature_importance.png

## How to Run Everything
1. Install the deps: 
pip install -r requirements.txt

2. Get the dataset:
python scripts/grabdata.py

3. Preprocess it:
python scripts/preprocess.py

4. Train the models:
python scripts/train_model.py

5. Evaluate:
python scripts/evaluate_model.py

## Results

Both models trained and evaluated without issues. The metrics below are from an 80/20 test split using the current setup without tuning:

**Linear Regression**
- RMSE: 0.75  
- MAE: 0.53  
- R²: 0.5758  

**Random Forest**
- RMSE: 0.54  
- MAE: 0.37  
- R²: 0.7737  

As expected, the random forest performed noticeably better across all metrics.

## What I Want to Try Next

Random Forest worked pretty well, so I'm curious if XGBoost would do even better. Might also experiment with creating new parameters, like combining income and rooms or something like that.

Eventually want to make a simple web app where you can input values and get a prediction. Also need to do proper cross-validation instead of just splitting once.

The hyperparameters are all defaults right now, so tuning them is probably the next in line for improving the model.