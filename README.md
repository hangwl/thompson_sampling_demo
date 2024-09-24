# Thompson Sampling Demo for Fraud Detection

Welcome to the Thompson Sampling Demo for Fraud Detection! This application is designed to simulate and visualize the use of Thompson Sampling in selecting predictive models for detecting fraudulent transactions. The app provides an interactive interface to experiment with various parameters and observe how the algorithm adapts over time.

---

## Introduction

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem, balancing exploration and exploitation by sampling from the posterior distributions of the rewards. In the context of fraud detection, it can help in dynamically selecting the best predictive model when dealing with non-stationary environments and delayed feedback.

This demo application simulates a scenario where two models (Model A and Model B) are available for predicting fraudulent transactions. Users can adjust various parameters to observe how the Thompson Sampling algorithm selects models and updates its beliefs over time.

## Key Features

- **Interactive Simulation**: Adjust parameters like recall rates, feedback delay, fraud rate, and decay rate to see real-time changes.
- **Visualization of Priors**: Observe how the Beta distributions (priors) for each model evolve.
- **Performance Metrics**: Track true positives, false negatives, and recall over time.
- **Model Selection Tracking**: Monitor how often each model is selected by the algorithm.
- **Verbose Logging**: View detailed logs of prior updates for each model.

---

## Getting Started

### Installation

1. Clone the Repository
    ```bash
    git clone https://github.com/hangwl/thompson_sampling_demo.git
    cd thompson_sampling_demo
    ```
2. Create a Virtual Environment
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    ```bash
    venv\Scripts\activate # windows
    source venv/bin/activate # Unix/MacOS
    ```
3. Install Required Packages
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the App
Start the Streamlit app by running:
```bash
streamlit run app.py
```
This command will launch the application in your default web browser.

---

## Understanding the Simulation Parameters

The app provides several parameters that you can adjust to simulate different scenarios. Understanding each parameter will help you interpret the results and the behavior of the Thompson Sampling algorithm.

### 1. Recall Rate of Models
- Recall Rate of Model A (`recall_A`)
- Recall Rate of Model B (`recall_B`)

**Definition**: The recall rate is the proportion of actual fraudulent transactions that the model correctly identifies (true positives) out of all actual fraudulent transactions.

**Significance**:
- Higher Recall Rate: The model is better at detecting fraud but might not be perfect.
- Lower Recall Rate: The model misses more fraudulent transactions (higher false negatives).

**Usage in Simulation**:
- Adjusting the recall rates simulates models with different performance levels.
- Observe how the Thompson Sampling algorithm favors models with higher recall rates over time.

### 2. Feedback Delay
- Feedback Delay (iterations) (`feedback_delay`)

**Definition**: The number of iterations (transactions) that pass before the outcome of a prediction (whether it was correct) is known and can be used to update the model's prior.

**Significance**:
- Zero Delay: Immediate feedback is available, and priors are updated right away.
- Positive Delay: Simulates real-world scenarios where verification of a transaction (e.g., customer disputes) takes time.

**Usage in Simulation**:
- Affects how quickly the algorithm can learn from its predictions.
- Longer delays slow down the adaptation of the priors.

### 3. Fraud Rate
- Fraud Rate (`fraud_rate`)

**Definition**: The proportion of transactions in the dataset that are fraudulent.

**Significance**:
- Higher Fraud Rate: More fraudulent transactions are present; models may have more opportunities to detect fraud.
- Lower Fraud Rate: Fraudulent transactions are rarer, which can make detection and learning more challenging.

**Usage in Simulation**:
- Adjusting the fraud rate can simulate different environments, such as high-risk or low-risk periods.
- Affects the number of positive cases the models can learn from.

### 4. Decay Rate
- Decay Rate (0.90 - 1.00) (`decay_rate`)

**Definition**: The rate at which older data is "forgotten" in the priors. It's an exponential decay applied to the Beta distribution parameters before updating with new data.

**Significance**:

- Decay Rate Close to 1.00: Slow forgetting; older data has a lasting impact.
- Lower Decay Rate (e.g., 0.90): Faster forgetting; the algorithm quickly adapts to new data.

**Usage in Simulation**:

- Helps the algorithm adapt in non-stationary environments where model performance can drift over time.
- Balances the trade-off between stability (using more historical data) and adaptability (favoring recent data).

---

## How the App Works

### 1. Model Selection with Thompson Sampling

- **Thompson Sampling Algorithm**:
    - For each model, sample a recall rate from its Beta distribution (the prior).
    - Select the model with the highest sampled recall rate.

- **Updating Priors**:
    - After receiving feedback, update the Beta distribution parameters (alpha, beta) based on the outcome.
    - Apply exponential decay to priors before updating to favor newer data.

### 2. Delayed Feedback Mechanism

- Predictions are added to a feedback queue with an associated delay.
- After the specified feedback delay, feedback is processed:
    - The outcome (true positive or false negative) is determined.
    - Priors are updated accordingly.
- Simulates real-world scenarios where outcomes are not immediately known.


### 3. Exponential Decay of Priors

- Before updating with new outcomes, priors are decayed:
    - `alpha *= decay_rate`
    - `beta *= decay_rate`
- Ensures that older observations have less influence over time.
- Allows the algorithm to adapt to changes in model performance or data distribution.

---

## Navigating the App

1. **Simulation Parameters**: Use the sidebar to adjust parameters:
    - Set recall rates for both models.
    - Specify the feedback delay.
    - Adjust the fraud rate.
    - Set the decay rate for priors.

2. **Run Simulation**:
    - Input the number of simulation steps (iterations) to run.
    - Click the "**Run Simulation**" button.

3. **Results and Visualizations**:
    - **Last Selected Model**: Displays the model chosen in the last iteration.
    - **Model Selection Counts**: Table showing how many times each model has been selected.
    - **Model Priors**: Plots the Beta distributions for each model's prior.
    - **Performance Metrics Over Time**: Graphs true positives and false negatives over iterations.
    - **Recall Over Time**: Plots the recall metric to show how it changes.
    - **Prior Update Log**: Table showing recent updates to model priors, including old and new values.
    - **Current Iteration and Overall Recall**: Displays the current iteration number and the overall recall.

**Interactive Visualizations**
- **Beta Distributions**: Visualize how the algorithm's belief in each model's recall rate evolves.
- **Performance Metrics**: Observe trends in true positives and false negatives.
- **Recall Over Time**: Understand how effective the models are at detecting fraud as the simulation progresses.