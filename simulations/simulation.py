# simulations/simulation.py

from collections import deque
from models.model import SyntheticModel
from models.thompson_sampling import ThompsonSampling
from data.transactions import TransactionGenerator
from utils.helpers import calculate_metrics, initialize_metrics

class Simulation:
    """
    Manages the simulation of model selection using Thompson Sampling with delayed feedback.

    Attributes:
        transaction_generator (TransactionGenerator): Generates synthetic transactions.
        models (list): List of SyntheticModel instances.
        thompson_sampler (ThompsonSampling): Instance of the ThompsonSampling algorithm.
        feedback_delay (int): Number of iterations to delay feedback.
        feedback_queue (deque): Queue to manage delayed feedback.
        current_iteration (int): Current iteration count of the simulation.
        metrics (dict): Dictionary to track performance metrics.
        metrics_history (list): List of metrics over time for visualization.
        model_selection_counts (dict): Counts of how many times each model was selected.
        prior_update_log (list): Log of prior updates for each model.
    """

    def __init__(self, recall_A, recall_B, feedback_delay, fraud_rate=0.05, decay_rate=1.0):
        """
        Initializes the Simulation.

        Parameters:
            recall_A (float): Initial recall rate for Model A (between 0 and 1).
            recall_B (float): Initial recall rate for Model B (between 0 and 1).
            feedback_delay (int): Delay in iterations before feedback is received.
            fraud_rate (float, optional): Base fraud rate in the transaction data (between 0 and 1). Default is 0.05.
            decay_rate (float, optional): Decay rate for exponential decay in priors (0 < decay_rate ≤ 1). Default is 1.0.

        Raises:
            ValueError: If any of the parameters are outside their valid ranges.
        """
        # Validate input parameters
        if not 0 <= recall_A <= 1:
            raise ValueError("recall_A must be between 0 and 1.")
        if not 0 <= recall_B <= 1:
            raise ValueError("recall_B must be between 0 and 1.")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative.")
        if not 0 <= fraud_rate <= 1:
            raise ValueError("fraud_rate must be between 0 and 1.")
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 (exclusive) and 1 (inclusive).")

        self.transaction_generator = TransactionGenerator(fraud_rate)
        self.models = [
            SyntheticModel('Model A', recall_A),
            SyntheticModel('Model B', recall_B)
        ]
        self.thompson_sampler = ThompsonSampling(self.models, decay_rate=decay_rate)
        self.feedback_delay = feedback_delay
        self.feedback_queue = deque()
        self.current_iteration = 0
        self.metrics = initialize_metrics()
        self.metrics_history = []

        # Initialize model selection counts
        self.model_selection_counts = {model.name: 0 for model in self.models}

        # Initialize prior update log
        self.prior_update_log = []

    def update_parameters(self, recall_A, recall_B, feedback_delay, fraud_rate=None, decay_rate=1.0):
        """
        Updates the simulation parameters, allowing for changes during runtime.

        Parameters:
            recall_A (float): New recall rate for Model A (between 0 and 1).
            recall_B (float): New recall rate for Model B (between 0 and 1).
            feedback_delay (int): New feedback delay in iterations.
            fraud_rate (float, optional): New base fraud rate (between 0 and 1).
            decay_rate (float, optional): New decay rate for priors (0 < decay_rate ≤ 1).

        Raises:
            ValueError: If any of the parameters are outside their valid ranges.
        """
        # Update model recall rates
        self.models[0].update_recall_rate(recall_A)
        self.models[1].update_recall_rate(recall_B)

        # Update feedback delay
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be non-negative.")
        self.feedback_delay = feedback_delay

        # Update fraud rate if provided
        if fraud_rate is not None:
            if not 0 <= fraud_rate <= 1:
                raise ValueError("fraud_rate must be between 0 and 1.")
            self.transaction_generator.set_fraud_rate(fraud_rate)

        # Update decay rate
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 (exclusive) and 1 (inclusive).")
        self.thompson_sampler.decay_rate = decay_rate

    def run_step(self):
        """
        Executes a single iteration of the simulation.

        - Increments the iteration counter.
        - Generates a new transaction.
        - Selects a model using Thompson Sampling.
        - Gets a prediction from the selected model.
        - Adds the prediction to the feedback queue.
        - Processes the feedback queue.

        Returns:
            str: The name of the model selected in this iteration.
        """
        self.current_iteration += 1
        transaction = self.transaction_generator.generate_transaction()
        selected_model_name = self.thompson_sampler.select_model()
        selected_model = next(model for model in self.models if model.name == selected_model_name)

        # Increment model selection count
        self.model_selection_counts[selected_model_name] += 1

        prediction = selected_model.predict(transaction)
        self.add_to_feedback_queue(transaction, prediction, selected_model_name)
        self.process_feedback_queue()
        return selected_model_name

    def add_to_feedback_queue(self, transaction, prediction, model_name):
        """
        Adds a prediction to the feedback queue with the specified delay.

        Parameters:
            transaction (dict): The transaction data.
            prediction (int): The model's prediction (1 for fraud, 0 for legitimate).
            model_name (str): The name of the model that made the prediction.
        """
        feedback = {
            'iteration_due': self.current_iteration + self.feedback_delay,
            'transaction': transaction,
            'prediction': prediction,
            'model_name': model_name
        }
        self.feedback_queue.append(feedback)

    def process_feedback_queue(self):
        """
        Processes the feedback queue, updating priors and performance metrics when feedback is due.
        Additionally, logs prior updates for verbosity.
        """
        while self.feedback_queue and self.feedback_queue[0]['iteration_due'] <= self.current_iteration:
            feedback = self.feedback_queue.popleft()
            transaction = feedback['transaction']
            prediction = feedback['prediction']
            model_name = feedback['model_name']

            # Determine the outcome for Bayesian update
            if transaction['label'] == 1:
                outcome = 1 if prediction == 1 else 0  # 1: True Positive, 0: False Negative

                # Store old priors before update for logging
                old_alpha = self.thompson_sampler.priors[model_name]['alpha']
                old_beta = self.thompson_sampler.priors[model_name]['beta']

                # Update the prior for the selected model
                self.thompson_sampler.update_prior(model_name, outcome)

                # Store new priors after update
                new_alpha = self.thompson_sampler.priors[model_name]['alpha']
                new_beta = self.thompson_sampler.priors[model_name]['beta']

                # Log the prior update
                log_entry = {
                    'Iteration': self.current_iteration,
                    'Model': model_name,
                    'Outcome': 'TP' if outcome == 1 else 'FN',
                    'Old Alpha': old_alpha,
                    'Old Beta': old_beta,
                    'New Alpha': new_alpha,
                    'New Beta': new_beta
                }
                self.prior_update_log.append(log_entry)

                # Update performance metrics
                self.metrics = calculate_metrics(self.metrics, transaction, prediction)
                self.metrics_history.append(self.metrics.copy())
            else:
                # Optionally update metrics for legitimate transactions
                pass  # Currently focusing on recall (fraud detection)
