def compute_single_sample_accuracy(expected_answers, actual_answers, tolerance=0.005):
    """
    Compare the expected and actual answers for a single sample and compute accuracy metrics.

    For each corresponding pair of answers:
      - If both answers can be interpreted as numbers, compute the absolute error.
        If the error is within the specified tolerance, mark it as correct (score 1),
        otherwise as incorrect (score 0).
      - If either answer is not numeric, compare the normalized strings (lowercased with spaces removed)
        for an exact match.
      - If one list is longer than the other, treat the missing items as incorrect.

    Args:
        expected_answers (list of str): The list of expected answers.
        actual_answers (list of str): The list of actual answers produced by the model.
        tolerance (float): The allowed tolerance for numeric comparisons. Default is 0.005.

    Returns:
        dict: A dictionary containing:
          - "accuracy_measurements": list of 0/1 scores for each answer comparison.
          - "mean_accuracy": average accuracy over all comparisons.
          - "mae": mean absolute error for numeric answers (None if no numeric comparisons).
          - "mse": mean squared error for numeric answers (None if no numeric comparisons).
          - "numeric_errors": list of numeric errors for the numeric comparisons (empty if none).
    """

    accuracy_measurements = []
    numeric_errors = []

    max_len = max(len(expected_answers), len(actual_answers))
    for i in range(max_len):
        if i >= len(expected_answers) or i >= len(actual_answers):
            accuracy_measurements.append(0)
        else:
            exp = expected_answers[i]
            act = actual_answers[i]

            # Attempt numeric comparison.
            try:
                exp_num = float(exp)
                act_num = float(act)
                error = abs(exp_num - act_num)
                numeric_errors.append(error)
                accuracy_measurements.append(1 if error <= tolerance else 0)
            except (ValueError, TypeError):
                # Fall back to string comparison if numeric conversion fails.
                accuracy_measurements.append(1 if str(exp) == str(act) else 0)

    mean_accuracy = (
        sum(accuracy_measurements) / len(accuracy_measurements)
        if accuracy_measurements
        else 0
    )

    if numeric_errors:
        mae = sum(numeric_errors) / len(numeric_errors)
        mse = sum(error**2 for error in numeric_errors) / len(numeric_errors)
    else:
        mae = None
        mse = None

    return {
        "accuracy_measurements": accuracy_measurements,
        "mean_accuracy": mean_accuracy,
        "mae": mae,
        "mse": mse,
        "numeric_errors": numeric_errors,
    }
