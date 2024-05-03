import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from typing import Dict

from .kaggle_metric_utilities import ParticipantVisibleError, safe_call_score


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculate a version of the macro-averaged ROC-AUC score that only includes classes with
    positive true labels in the solution set.

    Args:
        solution (pd.DataFrame): The ground truth labels.
        submission (pd.DataFrame): The participant's submitted labels.
        row_id_column_name (str): The column name in both dataframes to be removed before scoring.

    Returns:
        float: The computed macro-averaged ROC-AUC score.

    Raises:
        ParticipantVisibleError: If non-numeric data types are found in the submission.
    """
    # Remove the identifier columns as they are not needed for scoring
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Check if all columns in submission are of numeric data types
    if not is_numeric_dtype(submission.values):
        bad_dtypes: Dict[str, str] = {
            column: dtype for column, dtype in submission.dtypes.items()
            if not is_numeric_dtype(submission[column])
        }
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    # Determine which columns have positive true labels to be scored
    solution_sums = solution.sum(axis=0)
    scored_columns = [col for col, sum in solution_sums.items() if sum > 0]
    assert scored_columns, "No columns with positive true labels found."

    # Calculate the ROC-AUC score using only the relevant columns
    return safe_call_score(
        roc_auc_score,
        solution[scored_columns].values,
        submission[scored_columns].values,
        average='macro'
    )
