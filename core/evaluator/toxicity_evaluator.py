from openevals.llm import create_llm_as_judge
from openevals.prompts import TOXICITY_PROMPT
from mlflow.genai import scorer
from mlflow.entities import Feedback

toxicity_evaluator = create_llm_as_judge(
    prompt=TOXICITY_PROMPT,
    feedback_key="toxicity",
    continuous=True,  # if you want numeric scores
    model="openai:o3-mini",
)

@scorer
def toxicity_scorer(inputs, outputs) -> Feedback:
    """
    MLflow scorer that evaluates toxicity of a text using an OpenEvals LLM judge.
    
    Args:
        inputs: dict, must contain 'text' key
        outputs: the text to evaluate (can also use inputs['text'])
    """
    text_to_evaluate = inputs.get("text", outputs)  # fallback to outputs if needed

    # Call the LLM judge
    result = toxicity_evaluator(text=text_to_evaluate)

    # If continuous=True, result may have a numeric score
    numeric_score = float(result.get("score", 0.0))

    return Feedback(
        name="toxicity_score",
        value=numeric_score,
        rationale=f"Toxicity score evaluated as {numeric_score:.2f}"
    )
