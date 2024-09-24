# utils/helpers.py

def bayesian_update(prior, outcome):
    """
    Updates the Beta distribution parameters (alpha and beta) based on the outcome.

    **Note**: If you are using the `update_prior` method in the `ThompsonSampling` class
    (which handles exponential decay), you may not need this function directly.
    However, it's included here for completeness and potential future use.

    Parameters:
        prior (dict): Current prior with keys 'alpha' and 'beta'.
        outcome (int): Outcome of the prediction (1 for true positive, 0 for false negative).

    Returns:
        dict: Updated prior with incremented 'alpha' or 'beta'.
    """
    updated_prior = prior.copy()
    if outcome == 1:
        updated_prior['alpha'] += 1  # True positive
    elif outcome == 0:
        updated_prior['beta'] += 1   # False negative
    else:
        raise ValueError("Outcome must be 1 (true positive) or 0 (false negative).")
    return updated_prior

def calculate_metrics(metrics, transaction, prediction):
    """
    Updates performance metrics based on the transaction label and model's prediction.

    Parameters:
        metrics (dict): Current performance metrics with keys:
                        'true_positives', 'false_negatives', 'false_positives', 'true_negatives'.
        transaction (dict): The transaction data containing the 'label' key.
                            - 'label' (int): The ground truth label (1 for fraud, 0 for legitimate).
        prediction (int): The model's prediction (1 for fraud, 0 for legitimate).

    Returns:
        dict: Updated metrics dictionary with incremented counts.
    """
    updated_metrics = metrics.copy()
    true_label = transaction['label']
    if true_label == 1 and prediction == 1:
        updated_metrics['true_positives'] += 1
    elif true_label == 1 and prediction == 0:
        updated_metrics['false_negatives'] += 1
    elif true_label == 0 and prediction == 1:
        updated_metrics['false_positives'] += 1
    elif true_label == 0 and prediction == 0:
        updated_metrics['true_negatives'] += 1
    else:
        raise ValueError("Invalid transaction label or prediction value.")
    return updated_metrics

def initialize_metrics():
    """
    Initializes the performance metrics dictionary.

    Returns:
        dict: A dictionary with zeroed performance metrics:
              'true_positives', 'false_negatives', 'false_positives', 'true_negatives'.
    """
    metrics = {
        'true_positives': 0,
        'false_negatives': 0,
        'false_positives': 0,
        'true_negatives': 0
    }
    return metrics

def calculate_recall(metrics):
    """
    Calculates the recall from the performance metrics.

    Recall is the ratio of true positives to all actual positives (true positives + false negatives).

    Parameters:
        metrics (dict): The performance metrics containing 'true_positives' and 'false_negatives'.

    Returns:
        float: The calculated recall value (between 0 and 1).
    """
    tp = metrics['true_positives']
    fn = metrics['false_negatives']
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall

def calculate_precision(metrics):
    """
    Calculates the precision from the performance metrics.

    Precision is the ratio of true positives to all positive predictions (true positives + false positives).

    Parameters:
        metrics (dict): The performance metrics containing 'true_positives' and 'false_positives'.

    Returns:
        float: The calculated precision value (between 0 and 1).
    """
    tp = metrics['true_positives']
    fp = metrics['false_positives']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision
