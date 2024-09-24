# app.py

import streamlit as st
import pandas as pd  # Import pandas for displaying tables
from simulations.simulation import Simulation
from visuals.plots import plot_beta_distributions, plot_performance_metrics, plot_recall_over_time
from utils.helpers import calculate_recall

def main():
    st.title("Thompson Sampling Demo for Fraud Detection")

    # Sidebar controls for model parameters
    st.sidebar.header("Simulation Parameters")
    recall_A = st.sidebar.slider('Recall Rate of Model A', 0.0, 1.0, 0.8)
    recall_B = st.sidebar.slider('Recall Rate of Model B', 0.0, 1.0, 0.6)
    feedback_delay = st.sidebar.slider('Feedback Delay (iterations)', 0, 50, 10)
    fraud_rate = st.sidebar.slider('Fraud Rate', 0.0, 0.5, 0.05)
    decay_rate = st.sidebar.slider('Decay Rate (0.90 - 1.00)', 0.90, 1.00, 1.00, step=0.01)

    # Initialize or update the simulation
    if 'simulation' not in st.session_state:
        st.session_state.simulation = Simulation(
            recall_A, recall_B, feedback_delay, fraud_rate, decay_rate=decay_rate
        )
    else:
        st.session_state.simulation.update_parameters(
            recall_A, recall_B, feedback_delay, fraud_rate, decay_rate=decay_rate
        )

    # Initialize selected_model in session_state if it doesn't exist
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    # Run simulation steps
    num_steps = st.sidebar.number_input('Number of Simulation Steps', min_value=1, max_value=1000, value=1)

    if st.button('Run Simulation'):
        for _ in range(num_steps):
            selected_model = st.session_state.simulation.run_step()
        st.session_state.selected_model = selected_model  # Store the last selected model
        st.success(f"Ran {num_steps} simulation steps.")

    # Display selected model from last step
    if st.session_state.simulation.current_iteration > 0 and st.session_state.selected_model:
        st.write(f"Last Selected Model: **{st.session_state.selected_model}**")
    else:
        st.write("No model selected yet.")

    # Display model selection counts
    st.subheader("Model Selection Counts")
    model_counts = st.session_state.simulation.model_selection_counts
    counts_df = pd.DataFrame.from_dict(model_counts, orient='index', columns=['Selection Count'])
    counts_df.index.name = 'Model'
    counts_df = counts_df.reset_index()
    st.table(counts_df)

    # Plot priors
    st.subheader("Model Priors")
    plot_beta_distributions(st.session_state.simulation.thompson_sampler.priors)

    # Plot performance metrics
    st.subheader("Performance Metrics Over Time")
    plot_performance_metrics(st.session_state.simulation.metrics_history)

    # Plot recall over time
    st.subheader("Recall Over Time")
    plot_recall_over_time(st.session_state.simulation.metrics_history)

    # Display prior updates
    st.subheader("Prior Update Log")
    prior_update_log = st.session_state.simulation.prior_update_log
    if prior_update_log:
        # Display last N updates
        N = 10
        recent_updates = prior_update_log[-N:]
        updates_df = pd.DataFrame(recent_updates)
        st.table(updates_df)
    else:
        st.write("No prior updates yet.")

    # Display current iteration and recall
    st.write(f"Current Iteration: **{st.session_state.simulation.current_iteration}**")
    recall = calculate_recall(st.session_state.simulation.metrics)
    st.write(f"Overall Recall: **{recall:.2f}**")

if __name__ == "__main__":
    main()
