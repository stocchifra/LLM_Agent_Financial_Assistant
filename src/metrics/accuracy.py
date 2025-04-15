import random
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401

from src.agent import agent_executor_builder, system_prompt, tools
from src.metrics.compute_metrics import compute_single_sample_accuracy
from src.metrics.llm_as_a_judge import evaluate_answer
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
    verbose=True,
    seed=42,
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
        verbose (bool): Whether to print detailed output.

    Returns:
        dict: A dictionary containing:
              - "accuracy_measurements": aggregated list of 0/1 scores for each answer across samples.
              - "mean_accuracy": overall average accuracy score.
              - "mae": overall mean absolute error for numeric answers (None if no numeric answers).
              - "mse": overall mean squared error (None if no numeric answers).
              - "llm_average_score": overall average score from the LLM judge.
    """
    # Open and load the JSON data.
    data = open_json_file(data_path)

    # set the random seed for reproducibility
    random.seed(seed)

    # Select random indices for sample evaluation.
    random_indices = random.sample(range(len(data)), number_samples)

    # Lists to accumulate metrics from each sample.
    all_accuracy_measurements = []
    all_numeric_errors = []
    all_llm_scores = []
    sample = 0

    for index in random_indices:
        print(
            f"Initializing agent executor for sample index:{index} of trials {sample} out of {number_samples}"
        )

        # Extract and process the sample.
        data_processed = extract_selected_threads_processed(data)
        exact_answers, single_sample = get_exact_answers(data_processed[index])

        # Build the agent executor.
        agent_executor, memory = agent_executor_builder(
            model=model,
            provider=provider,
            temperature=temperature,
            tools=tools,
            prompt_style=prompt_style,
            memory_flag=memory_flag,
            verbose=verbose,
        )

        # # Invoke the agent according to the prompt style.
        # if prompt_style == "json-chat":
        #     response = agent_executor.invoke(
        #         {
        #             "input": f"Given the input extract the and then answer the questions {single_sample}."
        #         }
        #     )
        # else:
        #     response = agent_executor.invoke(
        #         {
        #             "input": f"{system_prompt}. Given the input extract the and then answer the questions {single_sample}"
        #         }
        #     )

        # Process the model's response.
        processed_answers = []
        response = agent_executor.invoke(
            {
                "input": f"""{system_prompt}. Pay attention to the format. Task: Given the following input extract the questions in the qa, qa_0, qa_1 fields and usign the provided context answer the questions. 
            Do not output any explanantion in the output only the answer {single_sample}"""
            }
        )
        # Process the model's response with original method
        processed_answers = []
        if isinstance(response["output"], str):
            for raw_answer in response["output"].split(","):
                # Normalize the answer: strip, lower, and remove spaces.
                answer = raw_answer.strip().lower().replace(" ", "")
                # Remove any surrounding parentheses, brackets, or braces using regex.
                answer = re.sub(r"^[\(\[\{](.*?)[\)\]\}]$", r"\1", answer).strip()
                try:
                    processed_answers.append(round(float(answer), 4))
                except ValueError:
                    processed_answers.append(answer)
        elif isinstance(response["output"], list):
            for raw_answer in response["output"]:
                if isinstance(raw_answer, dict):
                    answer = raw_answer.get("text", "")
                else:
                    answer = raw_answer
                # Normalize the answer: strip, lower, and remove spaces.
                answer = answer.strip().lower().replace(" ", "")
                # Remove any surrounding parentheses, brackets, or braces using regex.
                answer = re.sub(r"^[\(\[\{](.*?)[\)\]\}]$", r"\1", answer).strip()
                try:
                    processed_answers.append(round(float(answer), 4))
                except ValueError:
                    processed_answers.append(answer)

        # Use the compute_single_sample_accuracy function.
        sample_metrics = compute_single_sample_accuracy(
            expected_answers=exact_answers,
            actual_answers=processed_answers,
            tolerance=tolerance,
        )

        # Aggregate the per-sample accuracy measurements and numeric errors.
        all_accuracy_measurements.extend(sample_metrics["accuracy_measurements"])
        all_numeric_errors.extend(sample_metrics["numeric_errors"])

        # --- LLM Judge Evaluation ---
        # For each (expected, candidate) answer pair, evaluate using the LLM judge.
        sample_llm_scores = []
        for expected, candidate in zip(exact_answers, processed_answers):
            # Convert expected and candidate to string if needed
            eval_result = evaluate_answer(
                question=single_sample,
                expected_answer=str(expected),
                candidate_answer=str(candidate),
            )
            score = eval_result.get("score", 0)
            explanation = eval_result.get("explanation", "No explanation provided")
            sample_llm_scores.append(score)
            # print(
            #     f"LLM Evaluation for sample {index + 1}: "
            #     f"(Expected: {expected}, Candidate: {candidate}) => "
            #     f"Score: {score}, Explanation: {explanation}"
            # )
        # Append current sample's LLM scores to the overall list.
        all_llm_scores.extend(sample_llm_scores)

        # Final message for the sample.
        print(
            f"Analysis completed for sample {index} of {number_samples}...\n"
            f"Exact answers: {exact_answers}\n"
            f"Predicted answers: {processed_answers}\n"
            f"Sample Mean Accuracy: {sample_metrics['mean_accuracy']}, "
            f"Sample MAE: {sample_metrics['mae']}, Sample MSE: {sample_metrics['mse']}\n"
            f"LLM Scores: {sample_llm_scores}\n"
            f"LLM Explanations: {explanation}\n"
        )

        sample += 1

    # Compute overall mean accuracy from all sample measurements.
    overall_mean_accuracy = (
        (sum(all_accuracy_measurements) / len(all_accuracy_measurements))
        if all_accuracy_measurements
        else 0
    )

    # Calculate the overall MAE and MSE for numeric answers across all samples.
    if all_numeric_errors:
        overall_mae = sum(all_numeric_errors) / len(all_numeric_errors)
        overall_mse = sum(error**2 for error in all_numeric_errors) / len(
            all_numeric_errors
        )
    else:
        overall_mae = None
        overall_mse = None

    # Compute overall LLM average score.
    overall_llm_average_score = (
        sum(all_llm_scores) / len(all_llm_scores) if all_llm_scores else 0
    )

    return {
        "accuracy_measurements": all_accuracy_measurements,
        "mean_accuracy": overall_mean_accuracy,
        "mae": overall_mae,
        "mse": overall_mse,
        "llm_average_score": overall_llm_average_score,
    }
