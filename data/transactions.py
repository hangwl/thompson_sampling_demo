# data/transactions.py

import numpy as np

class TransactionGenerator:
    """
    Generates synthetic credit card transaction data with a specified fraud rate.
    """

    def __init__(self, fraud_rate: float = 0.05):
        """
        Initializes the TransactionGenerator.

        Parameters:
            fraud_rate (float, optional): The probability that a transaction is fraudulent.
                                          Must be between 0 and 1. Default is 0.05.

        Raises:
            ValueError: If fraud_rate is not between 0 and 1.
        """
        if not 0 <= fraud_rate <= 1:
            raise ValueError("fraud_rate must be between 0 and 1.")
        self.fraud_rate = fraud_rate

    def generate_transaction(self) -> dict:
        """
        Generates a single synthetic transaction with a ground truth label.

        Returns:
            dict: A dictionary containing transaction details:
                - 'id' (int): Unique transaction identifier.
                - 'features' (dict): Placeholder for transaction features.
                - 'label' (int): Ground truth label (1 for fraudulent, 0 for legitimate).
        """
        # Simulate transaction features (for simplicity, we'll just use a transaction ID)
        transaction_id = np.random.randint(1, 1_000_000)

        # Determine if the transaction is fraudulent based on the fraud rate
        is_fraud = 1 if np.random.rand() < self.fraud_rate else 0

        transaction = {
            'id': transaction_id,
            'features': {},  # Placeholder for transaction features
            'label': is_fraud
        }
        return transaction

    def set_fraud_rate(self, new_fraud_rate: float):
        """
        Updates the fraud rate for generating transactions.

        Parameters:
            new_fraud_rate (float): The new fraud rate to be set. Must be between 0 and 1.

        Raises:
            ValueError: If new_fraud_rate is not between 0 and 1.
        """
        if not 0 <= new_fraud_rate <= 1:
            raise ValueError("new_fraud_rate must be between 0 and 1.")
        self.fraud_rate = new_fraud_rate