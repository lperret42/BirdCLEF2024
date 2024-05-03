import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int) -> None:
    """
    Sets the seed for random number generation in torch, numpy, and random
    to ensure reproducibility.

    Args:
    seed (int): The seed value to set for all generators.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to False for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_weight_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a weight column to the DataFrame based on normalized ratings.
    If 'rating' column is not present, it sets a default weight of 1.0.

    Args:
    df (pd.DataFrame): DataFrame to which the weight column will be added.

    Returns:
    pd.DataFrame: Updated DataFrame with the new weight column.
    """
    if 'rating' in df.columns:
        df['weight'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    else:
        df['weight'] = 1.0  # Default constant weight if 'rating' does not exist

    return df


def print_duration(duration: int) -> None:
    """
    Prints the duration in hours, minutes, and seconds.

    Args:
    duration (int): Duration in seconds.
    """
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        print(f"Duration: {int(hours)} hour(s), {int(minutes)} minute(s), {seconds:.2f} second(s)")
    elif minutes > 0:
        print(f"Duration: {int(minutes)} minute(s), {seconds:.2f} second(s)")
    else:
        print(f"Duration: {seconds:.2f} second(s)")
