## Bayesian Convolutional Neural Networks for Probabilistic Atmospheric Prediction

This repository presents research on Bayesian Convolutional Neural Networks (B-CNNs) for short-term weather forecasting with explicit uncertainty quantification.

The framework integrates Monte Carlo stochastic approximation to model epistemic uncertainty in atmospheric sensor data and significantly outperforms traditional machine learning baselines.

## Research Paper:

https://github.com/asundar0128/Bayesian-Weather-Forecasting-Research/blob/main/Bayesian_CNN_Weather_Forecasting_Research_2024.pdf

## System Architecture

## Core Model: Bayesian Convolutional Neural Network (B-CNN)

<img width="1536" height="1024" alt="System Architecture" src="https://github.com/user-attachments/assets/59dbdcc4-656a-4cd1-885f-a06ac4c7fa46" />

<img width="1463" height="613" alt="CNN and Transformers" src="https://github.com/user-attachments/assets/4a18fafd-689f-456e-914b-826dbc705ac2" />

<img width="2000" height="1522" alt="More Images" src="https://github.com/user-attachments/assets/1f05211a-78ca-4c32-ba6b-de27f4d6b6bc" />

## Architectural Highlights

- Convolutional layers capture local temporal dependencies
- Bayesian weight distributions replace deterministic weights
- Monte Carlo sampling approximates posterior predictive distribution

Outputs include both:

- Mean forecast
- Predictive variance (uncertainty bounds)
- This transforms forecasting from point prediction → probabilistic prediction.

## Methodology

## Background Research

- Evaluated deterministic deep learning models for time-series forecasting.
- Identified limitations in uncertainty representation.
- Investigated Bayesian deep learning approaches for atmospheric modeling.
- Reviewed literature on stochastic approximation and probabilistic CNNs.

## Model Design

## Bayesian CNN (B-CNN)

Instead of fixed weights:

<img width="219" height="49" alt="W Normal" src="https://github.com/user-attachments/assets/d1c59aeb-19a6-4512-8064-7fef10167281" />

Forward passes sample from weight distributions to approximate:

<img width="126" height="45" alt="posterior" src="https://github.com/user-attachments/assets/cfaa5f06-ddb3-42c3-81df-752e8834e5d4" />

This captures epistemic uncertainty — uncertainty due to limited data or model knowledge.

## Uncertainty Quantification

## Monte Carlo Stochastic Approximation

- Multiple stochastic forward passes
- Aggregated predictions
- Computed predictive mean and variance

<img width="290" height="130" alt="Monte Carlo" src="https://github.com/user-attachments/assets/42ba1d1c-8612-4f7e-a772-7eebbab85705" />

This provides confidence intervals for short-term atmospheric forecasts.

## Baseline Comparison

We benchmarked against:

- Support Vector Machines (SVM)
- Decision Trees

<img width="655" height="158" alt="Performance Improvements" src="https://github.com/user-attachments/assets/4d547585-e5a6-495b-a049-3f77495b76d5" />

## Key Results

- Significant RMSE reduction for short-term sequences
- 40% improvement in computational efficiency
- Robust performance across varying atmospheric inputs
- Improved stability under noisy sensor data

## Forecasting Workflow

<img width="1357" height="448" alt="Monte Carlo Diagram" src="https://github.com/user-attachments/assets/0874d5fd-a1e7-4d3a-ad31-2f4e7852a72a" />

## Performance & Evaluation

## Metrics Used

- RMSE (Root Mean Squared Error)
- MAE
- Predictive Variance
- Computational runtime benchmarking

## Observed Improvements

- Lower RMSE for short-term temporal forecasting
- Increased robustness under sparse data conditions
- Improved predictive calibration

## Why Bayesian CNN Over Classical Models

| Classical ML              | Bayesian CNN                     |
| ------------------------- | -------------------------------- |
| Deterministic predictions | Probabilistic predictions        |
| No uncertainty modeling   | Epistemic uncertainty quantified |
| Sensitive to noise        | Robust under sensor variability  |
| Single inference pass     | Monte Carlo posterior sampling   |

## Technical Stack

- Python 3.8+
- PyTorch / TensorFlow
- NumPy, Pandas
- Matplotlib / Seaborn
- Jupyter Notebook
- GPU Acceleration (Optional)
- Docker (Optional)

## Quickstart

1. Clone the repository

git clone https://github.com/asundar0128/Bayesian-Weather-Forecasting-Research.git
cd Bayesian-Weather-Forecasting-Research

2. Install dependencies

pip install -r requirements.txt

3. Run training

python train.py

4. Run evaluation

python evaluate.py

## Future Scope

- Hierarchical Bayesian CNNs for multi-location modeling
- Gaussian Process hybrid architectures
- Spatiotemporal attention mechanisms
- Ensemble Bayesian deep learning
- Deployment as probabilistic forecasting API
