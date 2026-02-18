from openevals.llm import create_llm_as_judge

RELEVANCE_GRADING_PROMPT = """
You are an expert evaluator assessing the relevance of a retrieved document to a user question. Your task is to assign a binary relevance score based on the rubric below.

<Rubric>
  Assign a score of "yes" or "no" based on the following criteria:
  - yes: The document contains keyword(s) or semantic meaning related to the user question.
  - no: The document does not contain keyword(s) or semantic meaning related to the user question.
</Rubric>

<question>
{question}
</question>

<document>
{document}
</document>
"""

relevance_grading_evaluator = create_llm_as_judge(
    prompt=RELEVANCE_GRADING_PROMPT,
    feedback_key="document_grade",
    model="openai:o3-mini",
)