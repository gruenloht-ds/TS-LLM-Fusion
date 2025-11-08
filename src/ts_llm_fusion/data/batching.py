import torch
from torch.utils.data import DataLoader, Dataset, Sampler

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for supervised time-series forecasting using sliding windows.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing the full multivariate time series.
        Each row is a time step; columns represent features.
    input_len : int
        Number of past time steps to use as the input window.
    output_len : int
        Number of future time steps to predict.

    Notes
    -----
    The dataset returns windows using:
        - x = data[idx : idx + input_len]
        - y = data[idx + input_len : idx + input_len + output_len]

    The data is converted to float32 PyTorch tensors.
    """
    def __init__(self, data, input_len, output_len):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len

    def __getitem__(self, idx):
        window = idx + self.input_len
        x = self.data[idx:window]
        y = self.data[window:(window + self.output_len)]
        return torch.tensor(x.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).squeeze(0)

class SlidingWindowSampler(Sampler):
    """
    A sampler that yields start indices for sliding windows over a TimeSeriesDataset.

    Parameters
    ----------
    data : TimeSeriesDataset
        The dataset from which window start indices are drawn.
    stride : int, optional
        Number of steps to move between windows.
        Defaults to `data.input_len` if not specified, resulting in non-overlapping windows.

    Notes
    -----
    The sampler precomputes and stores the indices:
        indices = range(0, len(data), stride)
    """
    def __init__(self, data, stride=None):
        self.data = data
        # Default to length of the input sequence
        self.stride = self.data.input_len if stride is None else stride
        self.indices = list(range(0, len(self.data), self.stride))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)  