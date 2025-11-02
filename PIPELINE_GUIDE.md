# F1ML2 Quick Start Guide

Simple guide to run F1 predictions for qualifying and races.

---

## 📦 Setup (One-Time)

### Install Required Packages

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2
pip install -r requirements.txt
```

**If on Mac and XGBoost fails:**

```bash
brew install libomp
```

---

## 🎯 Run the Dashboard

**View predictions for any race:**

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2
streamlit run dashboard.py
```

Opens in browser at `http://localhost:8501`

**Features:**

- Select any season and round from dropdowns
- View qualifying predictions (pole position, grid order)
- View race predictions (podium finishers)
- Compare predictions vs actual results

---

## 🏁 Qualifying Predictions (Manual)

**What it does:** Predicts Saturday qualifying results - who gets pole position and starting grid order

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2/quali_training/models
python generatepredictions_quali.py
```

**Output:** Predicted qualifying order from P1 (pole) to P20

**Note:** Edit line 49 in the script to point to your prediction CSV file

---

## 🏆 Race Predictions (Manual)

**What it does:** Predicts Sunday race results - who finishes on the podium (1st, 2nd, 3rd)

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2/quali_training/models
python generatepredictions.py
```

**Output:** Predicted finishing order (lowest score = best finish)

**Note:** Edit line 19 in the script to point to your prediction CSV file

---

## 🔄 Retrain Models

**Retrain qualifying model:**

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2/quali_training/models
python trainmodel_quali.py
```

**Retrain race model:**

```bash
cd /Users/temiadebowale/Documents/CDS/F1ML2/quali_training/models
python trainmodel.py
```

---

## 📝 Understanding the Output

**Prediction Scores (XGBoost Ranker models):**

- Lower (more negative) score = Better predicted position
- Example: -0.95 beats -0.85
- The model ranks drivers relative to each other
- Only the order matters, not the actual numbers

**Note:** This applies to the XGBoost Ranker models used in the dashboard. Other models (XGBoost Classifier, Neural Network) produce different output formats.

---

## 🚀 Quick Reference

```bash
# Setup
pip install -r requirements.txt

# Run dashboard (easiest way)
streamlit run dashboard.py

# Manual qualifying predictions
cd quali_training/models && python generatepredictions_quali.py

# Manual race predictions
cd quali_training/models && python generatepredictions.py

# Retrain models
cd quali_training/models && python trainmodel_quali.py
cd quali_training/models && python trainmodel.py
```
