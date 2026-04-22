# Bike Demand Prediction: Statistical Modeling to Time Series Transition

## Overview

This project models bike rental demand using statistical and time series approaches on a dataset sourced from Kaggle. The goal was to identify an appropriate modeling framework for count-based demand data while addressing distributional assumptions and temporal dependencies.

---

## Dataset

* Source: Kaggle Bike Demand Dataset
* Type: Time-indexed count data
* Target Variable: Bike rental demand

---

## Methodology

### Baseline Model

* Linear Regression

  * Used as an initial benchmark
  * Assumes constant variance, which is violated in this dataset

---

### Generalized Linear Models

* Poisson Regression

  * Assumes mean equals variance
  * Observed overdispersion, leading to poor fit

* Negative Binomial

  * Accounts for overdispersion
  * Achieved the best fit among GLMs based on deviance

* Inverse Gaussian

  * Explored as an alternative distributional assumption

---

## Model Limitations

* Linear Regression

  * Failed due to heteroscedasticity

* Poisson

  * Invalid due to violation of mean equals variance assumption

* Negative Binomial

  * Improved deviance significantly
  * Residual diagnostics revealed:

    * High variability
    * Temporal dependence not captured by the model

---

## Results

* Negative Binomial achieved the lowest deviance among GLMs
* Poisson underperformed due to overdispersion
* GLMs failed to capture autocorrelation and time-dependent patterns

---

## Diagnostics and Insights

* Overdispersion observed: variance significantly greater than mean

* Residual analysis indicated:

  * Heavy-tailed behavior
  * Autocorrelation in residuals

* Key observation: a lower deviance does not imply a complete model when temporal structure is ignored

---

## Transition to Time Series Modeling

Due to the presence of autocorrelation and time-dependent structure in residuals, the modeling approach was extended to time series methods to better capture sequential dependencies in demand.

---

## Visual Analysis

The notebook includes:

* Residual vs time plots
* Autocorrelation (ACF) plots
* Distribution diagnostics

These visuals support the transition from GLMs to time series models.

---

## Project Structure

```bash
Project.ipynb   # Main analysis and modeling
src/            # Reserved for modular pipeline (future work)
```

---

## How to Run

```bash
1. Clone the repository
2. Open Project.ipynb
3. Run all cells sequentially
```

---

## Key Learnings

* Importance of validating distributional assumptions
* Limitations of GLMs for time-dependent data
* Role of residual diagnostics in model selection
* Need for time series models when autocorrelation is present

---

## Future Work

* Implement ARIMA or SARIMA models
* Explore hybrid approaches combining GLM and time series
* Build an API for predictions
* Convert the project into a full MLOps pipeline

---

## Key Takeaway

A statistically good fit does not guarantee a good model. Ignoring temporal structure can lead to misleading conclusions even when traditional metrics indicate strong performance.
