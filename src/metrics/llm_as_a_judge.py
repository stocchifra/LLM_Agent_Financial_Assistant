import json

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()


def evaluate_answer(question, expected_answer, candidate_answer):
    """
    Evaluates the candidate answer using an LLM as a judge.

    Parameters:
        question (str): The question that was asked.
        expected_answer (str): The gold-standard or expected answer.
        candidate_answer (str): The answer to be evaluated.

    Returns:
        dict: An evaluation result containing a score and an explanation.
    """
    # Constructing the evaluation prompt with clear instructions
    prompt = f"""You are an expert evaluator.

    Question: {question}

    Expected Answer: {expected_answer}

    Candidate Answer: {candidate_answer}

    Please evaluate the candidate answer as follows:

    If the expected answer is a number:
    - Compare the candidate answer numerically.
    - If the candidate answer is within a 1% relative difference of the expected answer, then consider it correct.
   
    If the expected answer is a yes/no answer:
    - Compare the candidate answer as a string ignoring case.
    - If the candidate answer matches the expected answer, consider it correct.

    Provide your evaluation output as a binary result:
    - Output "1" if the candidate answer is correct.
    - Output "0" if the candidate answer is not correct.

    In addition, provide a short explanation of your evaluation.

    Output your result in the following JSON format:
    {{
    "score": <score>,   // 1 if correct, 0 if not
    "explanation": "<your explanation>"
    }}

    Below are a few examples to illustrate how you should judge the answers:

    Example 1:
    Question: "What is the result of 2 + 2?"
    Expected Answer: "4"
    Candidate Answer: "The sum of 2 + 2 is 4"
    Evaluation Output:
    {{
    "score": 1,
    "explanation": "The candidate number is within 1% of the expected value."
    }}

    Example 2:
    Question: "Is the sky blue?"
    Expected Answer: "yes"
    Candidate Answer: "No"
    Evaluation Output:
    {{
    "score": 0,
    "explanation": "The candidate answer does not match the expected yes/no response."
    }}

    Now, please evaluate the candidate answer.
    """

    # Initialize a LangChain ChatOpenAI model
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # adjust to your desired model
        temperature=0.0,  # setting temperature to 0 for deterministic output
    )

    # Build messages: system message to set behavior, and human message with the prompt
    messages = [
        SystemMessage(content="You are a fair and accurate evaluator."),
        HumanMessage(content=prompt),
    ]

    # Get the response from the LLM
    try:
        result = llm.invoke(messages)
        answer = result.content

        # Parse and return the evaluation (assuming valid JSON output)
        evaluation = json.loads(answer)
        return evaluation
    except Exception as e:
        return {"score": 0, "explanation": f"Error during evaluation: {str(e)}"}


# Example usage:
if __name__ == "__main__":
    question = "What is the percentage change in net cash from operating activities from 2008 to 2009?"
    expected_answer = "14.1%"
    candidate_answer = "The net cash increased by 14.1 percent."
    evaluation_result = evaluate_answer(question, expected_answer, candidate_answer)
    print(evaluation_result)
