import numpy as np


class Loader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.current_index = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.X):
            raise StopIteration

        start = self.current_index
        end = self.current_index + self.batch_size
        batch_indices = self.indices[start:end]

        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        self.current_index = end
        return X_batch, y_batch
