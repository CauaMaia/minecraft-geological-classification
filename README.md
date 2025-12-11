# Minecraft Geological Classification

Machine learning models to automatically classify geological types of Minecraft chunks and predict ore quantities based on block composition.

## Overview

This project applies statistical modeling and machine learning algorithms to analyze Minecraft world data. Using block composition data from 10,201 chunks, we developed classification models (achieving 100% accuracy) and regression models (R² = 0.66) to predict geological patterns.

## Dataset

- **Source:** Kaggle - Minecraft Chunks Dataset
- **Size:** 10,201 chunks (100x100 world area)
- **Features:** 302 columns (299 block types + coordinates + dominant biome)
- **License:** CC0 Public Domain

## Models Implemented

### Classification
- Baseline (DummyClassifier)
- Naive Bayes
- Logistic Regression
- Gradient Boosting Classifier (100% accuracy)

### Regression
- Baseline (Mean Predictor)
- Simple Linear Regression
- Multiple Linear Regression (R² = 0.66)
- Polynomial Regression (degree 2)

## Key Results

- **Best Classification Model:** Gradient Boosting Classifier (100% accuracy)
- **Best Regression Model:** Multiple Linear Regression (MAE = 40.69 ores, R² = 0.66)
- **Geological Types Identified:** Underground (96.7%), Cave (2.3%), Mixed (1.0%)
- **Visualizations:** 20 professional charts including geological comparison maps

## Technologies

- **Python 3.11**
- **Core Libraries:** pandas, numpy, scikit-learn, statsmodels
- **Visualization:** matplotlib, seaborn
- **AutoML:** PyCaret
- **Statistical Tests:** ANOVA, Shapiro-Wilk, VIF

## Project Structure

```
.
├── maps.ipynb                    # Main Jupyter notebook with all analysis
├── minecraft_100x100_1.csv       # Dataset
├── outputs/                      # Generated visualizations (20 PNG files)
├── venv_projeto/                 # Python virtual environment
└── logs.log                      # Execution logs
```

## Installation

```bash
# Create virtual environment
python3.11 -m venv venv_projeto
source venv_projeto/bin/activate  # On Windows: venv_projeto\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pycaret scipy --break-system-packages
```

## Usage

Open and run `maps.ipynb` in Jupyter Notebook or JupyterLab. The notebook contains:

1. Exploratory Data Analysis (EDA)
2. Statistical tests (ANOVA)
3. Feature engineering
4. Model training and evaluation
5. Hyperparameter optimization (PyCaret + GridSearchCV)
6. Geological map visualizations

## Key Features

- **Comprehensive EDA:** Includes correlation analysis, distribution plots, pairplots, and statistical hypothesis testing
- **Model Comparison:** Systematic comparison of 5+ classification and 4+ regression algorithms
- **Optimization:** Cross-validation, GridSearchCV, and AutoML with PyCaret
- **Residual Diagnostics:** Normality tests, homoscedasticity analysis, and VIF for multicollinearity
- **Interactive Maps:** Visual comparison of real vs predicted geological types

## Results Highlights

### Classification Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Gradient Boosting | 100% | 100% | 100% | 100% |
| Logistic Regression | 98.3% | 98.5% | 98.3% | 98.4% |

### Regression Metrics
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Multiple Linear | 40.69 | 52.65 | 0.664 |
| Polynomial (deg 2) | 40.71 | 52.78 | 0.662 |

## Visualizations

The project generates 20 professional visualizations including:
- Geological type distributions
- Correlation heatmaps
- Confusion matrices
- ROC curves
- Residual diagnostic plots
- **Geological comparison maps** (real vs predicted)

## Academic Requirements

This project fulfills all requirements for the Statistical Modeling course:
- ✅ Complete EDA with statistical tests
- ✅ Regression models (simple, multiple, polynomial)
- ✅ Classification models (Naive Bayes, Logistic Regression)
- ✅ Performance metrics (MAE, RMSE, R², Accuracy, Precision, Recall, F1, AUC-ROC)
- ✅ Residual diagnostics and multicollinearity analysis (VIF)
- ✅ Cross-validation and hyperparameter tuning
- ✅ PyCaret AutoML integration
- ✅ Professional documentation


## Author
Cauã Maia 
Antonio Heitor
Developed as part of the Statistical Modeling course at Centro Universitário do Estado do Pará.

## References

- Minecraft Chunks Dataset (Kaggle)
- Scikit-learn Documentation
- PyCaret Documentation
- Statsmodels Documentation
