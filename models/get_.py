import numpy as np
import pandas as pd


class data_:
    def __init__(self, path, drop_outlier=True):
        self.path = path
        self.outlier = set()
        self.drop = drop_outlier

    def read(self):
        return pd.read_csv(self.path, index_col='id')

    def zScore(self, data):
        numeric_columns = data.select_dtypes(include=np.number)
        z_score_rec = dict()
        ZScore = pd.DataFrame()

        for column in numeric_columns:
            col_median = np.median(data[column])
            MAD = np.median(data[column].apply(lambda x: np.abs(x - col_median)))
            ZScore[column] = data[column].apply(lambda x: 0.6745 * (x - col_median) / MAD)

        for column in numeric_columns:
            z_score_rec[column] = data[column].iloc[np.where(np.abs(ZScore[column]) >= 3)].shape[0]
            self.outlier.update(data[column].iloc[np.where(np.abs(ZScore[column]) >= 3)].index)

        return z_score_rec, self.outlier

    def norm_data(self, data):
        if self.drop:
            data.drop(self.outlier, axis=0, inplace=True)
