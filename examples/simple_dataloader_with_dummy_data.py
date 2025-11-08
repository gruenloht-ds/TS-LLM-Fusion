from torch.utils.data import DataLoader

from ts_llm_fusion.synthetic import dummy_series
from ts_llm_fusion.data.batching import TimeSeriesDataset, SlidingWindowSampler

context_time_points = 10
time_points_to_predict = 1
stride = 1 # how far to move the sliding window after each sample
batch_size = 10

dataset = TimeSeriesDataset(data = dummy_series, input_len = context_time_points, output_len = time_points_to_predict)
sampler = SlidingWindowSampler(data = dataset, stride = stride)
dataloader = DataLoader(dataset, sampler=sampler, batch_size = batch_size, shuffle=False, drop_last=True)

item = next(iter(dataloader))
x, y = item

print("Total batches:", len(dataloader))
print("Shape of X:", x.shape)
print("Shape of y:", y.shape)
