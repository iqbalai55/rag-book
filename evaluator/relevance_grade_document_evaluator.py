from openevals.llm import create_llm_as_judge
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

RELEVANCE_GRADING_PROMPT = """
You are an expert evaluator assessing the relevance of a retrieved document to a user question. Your task is to assign a numeric relevance score between 0 and 1.

<question>
{question}
</question>

<document>
{document}
</document>
"""

# Create the judge once
relevance_grading_evaluator = create_llm_as_judge(
    prompt=RELEVANCE_GRADING_PROMPT,
    continuous=True,
    model="openai:o3-mini",
)

@scorer
def relevance_grading_scorer(inputs, outputs) -> Feedback:
    question = inputs["question"]
    document = outputs

    result = relevance_grading_evaluator(question=question, document=document)
    numeric_score = result.get("score", 0.0)

    return Feedback(
        name="relevance_document_score",
        value=numeric_score,
        rationale=f"Evaluated as {numeric_score:.2f}"
    )

