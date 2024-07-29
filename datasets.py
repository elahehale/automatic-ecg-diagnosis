import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            y = pd.read_csv(path_to_csv, index_col="exam_id")
            self.y = y.astype(np.float32)

        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")

        # list all datasets in hdf5 file
        with h5py.File(path_to_hdf5, "r") as hdf:
            # Recursively list all datasets in the file
            def list_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(name)
            hdf.visititems(list_datasets)

        # get tracings and remove last index which is empty
        self.x = self.f[hdf5_dset]
        self.x = self.x[:-1]

        # list of exam_id's of tracings
        tracing_ids = self.f['exam_id']

        # sort labels to match tracings index (they are not in same order)
        df = self.y.reindex(tracing_ids, fill_value=False, copy=True)
        self.y = df

        # eliminiate unnecessary columns
        self.y = self.y[['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']].values

        # get tracings and remove last index which is empty
        self.y = self.y[:-1]
        self.y = self.y.astype(np.float32)

        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Print shapes of x and y
        print(f"Shape of x: {self.x.shape}")
        if self.y is not None:
            print(f"Shape of y: {self.y.shape}")

        print(tracing_ids[0], 'first index in hdf5 file')
        print("First data sample in x:", self.x[0])

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
