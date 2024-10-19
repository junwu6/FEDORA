from typing import Tuple
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


class DatasetForecasting(torch.utils.data.Dataset):
    def __init__(
        self, csv_file: str, input_size: int, forcast_horizon: int, stride: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forcast_horizon = forcast_horizon
        self.stride = stride
        self.df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        # self.df, self.min_val, self.max_val = normalize(self.df)
        self.min_val, self.max_val = 0, 1  # TODO: normalize during runtime

        self.X, self.y = self.run_sliding_window(self.df)

    def run_sliding_window(self, df: pd.DataFrame) -> Tuple[np.array, np.array]:
        """Creates the input-output pairs for the forecasting task.
        Discard windows with NaN values.

        Args:
            df (pd.DataFrame): Time series data.

            Returns:
                X (np.array): Model input data (N, input_size, n_features).
                y (np.array): forecast target data (N, forcast_horizon, n_features).
        """
        timeseries = df.values
        X, y = [], []
        in_st = 0
        out_en = in_st + self.input_size + self.forcast_horizon
        while out_en < len(timeseries):
            in_end = in_st + self.input_size
            out_end = in_end + self.forcast_horizon
            seq_x, seq_y = timeseries[in_st:in_end], timeseries[in_end:out_end]
            if np.isnan(seq_x).any() or np.isnan(seq_y).any():
                in_st += self.stride
                out_en = in_st + self.input_size + self.forcast_horizon
                continue
            X.append(seq_x)
            y.append(seq_y)
            in_st += self.stride
            out_en = in_st + self.input_size + self.forcast_horizon
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.X)


def get_client_data(
    csv_file: str,
    input_size: int,
    forcast_horizon: int,
    stride: int,
    batch_size: int,
    valid_set_size: int,
    test_set_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:

    dataset = DatasetForecasting(csv_file, input_size, forcast_horizon, stride)
    train_size = int(len(dataset) * (1 - valid_set_size - test_set_size))
    valid_size = int(len(dataset) * valid_set_size)
    test_size = len(dataset) - train_size - valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )
    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    validloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader, validloader, testloader, dataset.min_val, dataset.max_val


class BaseNodes_load:
    def __init__(
        self,
        data_path,
        n_nodes,
        batch_size=32,
        input_size=24 * 6,
        forecast_horizon=24,
        stride=24,
        valid_set_size=0.15,
        test_set_size=0.15,
    ) -> None:
        self.data_path = data_path
        self.n_nodes = n_nodes
        self.batch_size = batch_size

        dataset_files = glob.glob(os.path.join(self.data_path, "*.csv"))[:n_nodes]
        self.train_loaders, self.val_loaders, self.test_loaders = [], [], []
        for dataset_file in dataset_files:
            train_loader, val_loader, test_loader, _, _ = get_client_data(
                dataset_file,
                input_size,
                forecast_horizon,
                stride,
                batch_size,
                valid_set_size,
                test_set_size,
            )
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
            self.test_loaders.append(test_loader)

    def __len__(self) -> int:
        return len(self.n_nodes)


if __name__ == "__main__":
    nodes = BaseNodes_load("data/REFIT", 20)
    print(len(nodes.train_loaders), len(nodes.val_loaders), len(nodes.test_loaders))
    for k in range(20):
        print(
            len(nodes.train_loaders[k]),
            len(nodes.val_loaders[k]),
            len(nodes.test_loaders[k]),
        )
