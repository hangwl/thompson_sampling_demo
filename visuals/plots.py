# visuals/plots.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

def plot_beta_distributions(priors):
    """
    Plots the Beta distributions representing the priors for each model.

    Parameters:
        priors (dict): A dictionary where keys are model names and values are dictionaries
                       with 'alpha' and 'beta' parameters of the Beta distribution.

    Displays:
        A matplotlib plot embedded in the Streamlit app showing the Beta distributions
        for each model's prior.
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    for model_name, params in priors.items():
        alpha = params['alpha']
        beta_param = params['beta']
        y = beta.pdf(x, alpha, beta_param)
        ax.plot(x, y, label=model_name)
    ax.set_title('Priors (Beta Distributions)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)

def plot_performance_metrics(metrics_history):
    """
    Plots the performance metrics (True Positives and False Negatives) over time.

    Parameters:
        metrics_history (list): A list of metrics dictionaries collected over iterations.
                                Each dictionary contains counts of 'true_positives' and 'false_negatives'.

    Displays:
        A matplotlib plot embedded in the Streamlit app showing the counts of True Positives
        and False Negatives over iterations.
    """
    if not metrics_history:
        st.write("No metrics to display yet.")
        return

    iterations = list(range(1, len(metrics_history) + 1))
    tps = [metrics['true_positives'] for metrics in metrics_history]
    fns = [metrics['false_negatives'] for metrics in metrics_history]

    fig, ax = plt.subplots()
    ax.plot(iterations, tps, label='True Positives', color='green')
    ax.plot(iterations, fns, label='False Negatives', color='red')
    ax.set_title('Model Performance Over Time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

def plot_recall_over_time(metrics_history):
    """
    Plots the recall metric over time.

    Parameters:
        metrics_history (list): A list of metrics dictionaries collected over iterations.

    Displays:
        A matplotlib plot embedded in the Streamlit app showing the recall over iterations.
    """
    if not metrics_history:
        st.write("No metrics to display yet.")
        return

    iterations = list(range(1, len(metrics_history) + 1))
    recalls = []
    for metrics in metrics_history:
        tp = metrics['true_positives']
        fn = metrics['false_negatives']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    fig, ax = plt.subplots()
    ax.plot(iterations, recalls, label='Recall', color='blue')
    ax.set_title('Recall Over Time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Recall')
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)
