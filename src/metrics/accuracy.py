import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401

from src.agent import agent_executor_builder, system_prompt, tools
from src.utils import (
    extract_selected_threads_processed,
    get_exact_answers,
    open_json_file,
)


def measure_accuracy(
    data_path,
    model,
    provider,
    prompt_style,
    tolerance=0.005,
    number_samples=10,
    tools=tools,
    temperature=0,
    memory_flag=False,
):
    """
    Measure accuracy by comparing the predicted answers with the exact answers over a number of random samples.
    For numeric answers, a tolerance is applied and errors are used to compute MAE and MSE.
    For non-numeric answers (e.g., "yes" or "no"), a strict string comparison (after lowercasing and removing spaces) is used.
    If there is a mismatch in the number of answers between the prediction and the ground truth,
    the missing positions are scored as 0 (incorrect).

    Args:
        data_path (str): Path to the JSON data file.
        model: The model to be used.
        provider: The provider for the agent.
        prompt_style (str): The style of prompt (affecting the system input).
        tolerance (float): Tolerance for numeric comparisons.
        number_samples (int): Number of random samples to evaluate.
        tools: Tools to be used by the agent.
        temperature (float): Temperature for the model.
        memory_flag (bool): Whether memory is enabled.

    Returns:
        tuple: (all_accuracy_measurements, overall_mean_accuracy, overall_mae, overall_mse)
               where all_accuracy_measurements is a list of 0/1 scores for each answer across samples,
               overall_mean_accuracy is the average accuracy score,
               overall_mae is the mean absolute error for numeric answers (None if no numeric answers),
               and overall_mse is the mean squared error (None if no numeric answers).
    """
    # Open the JSON file and load the data.
    data = open_json_file(data_path)

    # Generate a list of random indices equal to the number of samples from the data.
    random_indices = random.sample(range(len(data)), number_samples)

    # Initialize lists to aggregate metrics across all samples.
    all_accuracy_measurements = []
    all_numeric_errors = []

    # Loop through the random indices and process each sample.
    for index in random_indices:
        # Print initialization message.
        print(
            f"Initializing agent executor for sample {index + 1} of {number_samples}..."
        )

        # Extract the sample at the current index.
        data_processed = extract_selected_threads_processed(data)

        # Extract the exact answers from the sample.
        exact_answers, single_sample = get_exact_answers(data_processed[index])

        # Build the agent executor.
        agent_executor, memory = agent_executor_builder(
            model=model,
            provider=provider,
            temperature=temperature,
            tools=tools,
            prompt_style=prompt_style,
            memory_flag=memory_flag,
            verbose=True,
        )

        # Invoke the agent based on prompt style.
        if prompt_style == "json-chat":
            response = agent_executor.invoke(
                {
                    "input": f"Given the input extract the and then answer the questions {single_sample}."
                }
            )
        else:
            response = agent_executor.invoke(
                {
                    "input": f"{system_prompt}. Given the input extract the and then answer the questions {single_sample}"
                }
            )

        # Normalize predicted answers.
        processed_answers = []
        for raw_answer in response["output"].split(","):
            answer = raw_answer.strip().lower().replace(" ", "")
            # Try converting to float; if conversion fails, keep as string.
            try:
                processed_answers.append(round(float(answer), 4))
            except ValueError:
                processed_answers.append(answer)

        # Compare pairwise with the exact answers.
        # If the lists have different lengths, add a score of 0 for missing answers.
        sample_accuracy = []
        max_len = max(len(processed_answers), len(exact_answers))
        for i in range(max_len):
            if i >= len(processed_answers) or i >= len(exact_answers):
                sample_accuracy.append(0)
            else:
                pred = processed_answers[i]
                exact = exact_answers[i]
                # Try numeric comparison with tolerance.
                try:
                    pred_num = float(pred)
                    exact_num = float(exact)
                    error = abs(pred_num - exact_num)
                    all_numeric_errors.append(error)
                    if error <= tolerance:
                        sample_accuracy.append(1)
                    else:
                        sample_accuracy.append(0)
                except ValueError:
                    # Compare as strings.
                    if pred == exact:
                        sample_accuracy.append(1)
                    else:
                        sample_accuracy.append(0)
        # Append the sample's accuracy results to the overall measurements.
        all_accuracy_measurements.extend(sample_accuracy)

        # final message for the sample
        print(
            f"Analysis completed for sample {index + 1} of {number_samples}...\n"
            f"Exact answers: {exact_answers}\n"
            f"Predicted answers: {processed_answers}\n"
        )

    # Compute overall mean accuracy from the aggregated measurements.
    overall_mean_accuracy = (
        (sum(all_accuracy_measurements) / len(all_accuracy_measurements))
        if all_accuracy_measurements
        else 0
    )

    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE) for numeric answers.
    if all_numeric_errors:
        overall_mae = sum(all_numeric_errors) / len(all_numeric_errors)
        overall_mse = sum(error**2 for error in all_numeric_errors) / len(
            all_numeric_errors
        )
    else:
        overall_mae = None
        overall_mse = None

    return {
        "accuracy_measurements": all_accuracy_measurements,
        "mean_accuracy": overall_mean_accuracy,
        "mae": overall_mae,
        "mse": overall_mse,
    }


if __name__ == "__main__":
    # Example usage
    data_path = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/train.json"
    model = "gpt-4o"
    provider = "openai"
    prompt_style = "react"  # or any other style you want to test
    tolerance = 0.005
    number_samples = 3

    metrics = measure_accuracy(
        data_path, model, provider, prompt_style, tolerance, number_samples
    )
    print("Outcomes:", metrics["accuracy_measurements"])
    print("Mean Accuracy:", metrics["mean_accuracy"])
    print("Mean Absolute Error (MAE):", metrics["mae"])
    print("Mean Squared Error (MSE):", metrics["mse"])
