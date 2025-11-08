import numpy as np
import pandas as pd

def generate_dummy_timeseries_simple(n_timepoints=10_000, n_features=4, seed=42):
    """
    Generates a simple dummy multivariate time series dataset where each feature
    is independent Gaussian white noise.

    Really nothing of substance here, just to test out the dataloader.

    Args:
        n_timepoints (int): The number of time steps/rows to generate.
        n_features (int): The number of features/columns.
        seed (int): The random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing the dummy time series data.
    """
    np.random.seed(seed)
    data = np.random.normal(loc=0, scale=1.0, size=(n_timepoints, n_features))
    column_names = [f'Feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=column_names)

    return df

dummy_series = generate_dummy_timeseries_simple()
