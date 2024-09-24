# models/model.py

import numpy as np

class SyntheticModel:
    """
    Represents a synthetic predictive model with an adjustable recall rate.
    """

    def __init__(self, name: str, recall_rate: float):
        """
        Initializes the SyntheticModel with a given name and recall rate.

        Parameters:
            name (str): The name of the model.
            recall_rate (float): The recall rate of the model, between 0 and 1.

        Raises:
            ValueError: If recall_rate is not between 0 and 1.
        """
        if not 0 <= recall_rate <= 1:
            raise ValueError("recall_rate must be between 0 and 1.")
        self.name = name
        self.recall_rate = recall_rate

    def predict(self, transaction: dict) -> int:
        """
        Simulates a prediction for a given transaction based on the model's recall rate.

        Parameters:
            transaction (dict): A dictionary representing a transaction with a 'label' key.

        Returns:
            int: The model's prediction (1 for fraud, 0 for legitimate).

        Assumptions:
            - The model is assumed to have perfect precision (no false positives).
            - The recall rate determines the probability of correctly identifying fraudulent transactions.
        """
        if transaction['label'] == 1:
            # Fraudulent transaction
            prediction = 1 if np.random.rand() < self.recall_rate else 0
        else:
            # Legitimate transaction
            prediction = 0  # Assuming perfect precision
        return prediction

    def update_recall_rate(self, new_recall_rate: float):
        """
        Updates the model's recall rate to simulate model drift.

        Parameters:
            new_recall_rate (float): The new recall rate to be set, between 0 and 1.

        Raises:
            ValueError: If new_recall_rate is not between 0 and 1.
        """
        if not 0 <= new_recall_rate <= 1:
            raise ValueError("new_recall_rate must be between 0 and 1.")
        self.recall_rate = new_recall_rate
