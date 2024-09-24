# models/thompson_sampling.py

import numpy as np

class ThompsonSampling:
    """
    Implements the Thompson Sampling algorithm for model selection with an exponential decay on priors.

    Attributes:
        models (list): A list of models to select from.
        priors (dict): A dictionary containing the Beta distribution parameters ('alpha' and 'beta') for each model.
        decay_rate (float): The rate at which to decay the priors (0 < decay_rate ≤ 1).
    """

    def __init__(self, models, decay_rate: float = 1.0):
        """
        Initializes the ThompsonSampling instance.

        Parameters:
            models (list): A list of model instances.
            decay_rate (float, optional): The decay rate for priors. Must be between 0 (exclusive) and 1 (inclusive). Default is 1.0.

        Raises:
            ValueError: If decay_rate is not between 0 and 1.
        """
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be between 0 (exclusive) and 1 (inclusive).")
        self.models = models
        self.priors = {model.name: {'alpha': 1.0, 'beta': 1.0} for model in models}
        self.decay_rate = decay_rate

    def select_model(self) -> str:
        """
        Selects a model based on Thompson Sampling by sampling from each model's Beta distribution.

        Returns:
            str: The name of the selected model with the highest sampled recall.
        """
        sampled_recalls = {}
        for model in self.models:
            alpha = self.priors[model.name]['alpha']
            beta = self.priors[model.name]['beta']
            sampled_recalls[model.name] = np.random.beta(alpha, beta)
        selected_model_name = max(sampled_recalls, key=sampled_recalls.get)
        return selected_model_name

    def update_prior(self, model_name: str, outcome: int):
        """
        Updates the prior Beta distribution for the specified model based on the outcome, applying exponential decay.

        Parameters:
            model_name (str): The name of the model to update.
            outcome (int): The outcome of the model's prediction (1 for true positive, 0 for false negative).

        Raises:
            ValueError: If outcome is not 0 or 1.
        """
        if outcome not in [0, 1]:
            raise ValueError("Outcome must be 1 (true positive) or 0 (false negative).")

        # Apply decay to priors
        self.priors[model_name]['alpha'] *= self.decay_rate
        self.priors[model_name]['beta'] *= self.decay_rate

        # Update with new outcome
        if outcome == 1:
            self.priors[model_name]['alpha'] += 1
        else:
            self.priors[model_name]['beta'] += 1
