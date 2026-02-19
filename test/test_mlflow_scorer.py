import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluator.relevance_grade_document_evaluator import relevance_grading_scorer
import mlflow
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Evaluation dataset with multiple questions
eval_dataset = [
    {
        "inputs": {"question": "How much has the price of doodads changed in the past year?"},
        "outputs": "Doodads have increased in price by 10% in the past year."
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Paris."
    },
    {
        "inputs": {"question": "How is the weather in San Francisco?"},
        "outputs": "Currently sunny and 90 degrees in San Francisco."
    }
]

# Run evaluation
evaluation_result = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[relevance_grading_scorer],
)

# Print aggregated metrics
print("Aggregated metrics:")
for key, value in evaluation_result.metrics.items():
    print(f"{key}: {value:.3f}")

# Show full row-level results
print("\nRow-level results:")
print(evaluation_result.result_df)

# Optional: convert to pandas DataFrame for easier inspection
df = pd.DataFrame(evaluation_result.result_df)
print("\nDataFrame view:")
print(df)
