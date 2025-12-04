# Models Directory Structure

## quali_training/models/

### TRAINED MODELS (Ready to Use)

- **xgbranker_quali_model.json** (XGBoost ranking model for qualifying)
- **xgbranker_model.json** (XGBoost ranking model for race)
- **xgb_quali_classifier.joblib** (XGBoost classifier for qualifying)
- **xgb_classifier.joblib** (XGBoost classifier for race)
- **neural_nets_quali.pth** (Neural network for qualifying)
- **neural_nets.pth** (Neural network for race)
- **race3_linear.pickle** (Linear regression baseline for race)
- **linear_scaler.pkl** (Feature normalization scaler)

### TRAINING SCRIPTS (Retrain Models)

- **trainmodel_quali.py** (Trains qualifying ranker model)
- **trainmodel.py** (Trains race ranker model)
- **xgb_script_quali.py** (Trains qualifying classifier model)
- **xgb_script.py** (Trains race classifier model)
- **nn_script_quali.py** (Trains qualifying neural network)
- **nn_script.py** (Trains race neural network)
- **pairwise_ranking_quali.py** (Helper functions for qualifying neural network)
- **pairwise_ranking.py** (Helper functions for race neural network)
- **F1_models.py** (Trains linear/SVM/random forest models)

### PREDICTION SCRIPTS (Use Models)

- **generatepredictions_quali.py** (Generates qualifying predictions)
- **generatepredictions.py** (Generates race predictions)

### RESULTS/LOGS

- **trainmodel_results.txt** (XGBoost ranker training logs)
- **xgb_classifier_results.txt** (XGBoost classifier training logs)
- **nn_results.txt** (Neural network training logs)
