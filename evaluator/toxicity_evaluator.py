from openevals.llm import create_llm_as_judge
from openevals.prompts import TOXICITY_PROMPT 

toxicity_evaluator = create_llm_as_judge(
    prompt=TOXICITY_PROMPT,
    feedback_key="toxicity",
    model="openai:o3-mini",
)