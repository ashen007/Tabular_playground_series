import numpy as np
import pandas as pd


class data_:
    def __init__(self, path):
        self.path = path

    def read(self):
        return pd.read_csv(self.path, index_col='id')

    def zScore(self, data):
        numeric_columns = data.select_dtypes(include=np.number)
        z_score_rec = dict()
        outliers = set()
        ZScore = pd.DataFrame()

        for column in numeric_columns:
            col_median = np.median(column)
            MAD = np.median(data[column].apply(lambda x: np.abs(x - col_median)))
            ZScore[column] = data[column].apply(lambda x: 0.6745 * (x - col_median) / MAD)

        for column in numeric_columns:
            z_score_rec[column] = data[column].iloc[np.where(np.abs(ZScore[column]) >= 3)].shape[0]
            outliers.update(data[column].iloc[np.where(np.abs(ZScore[column]) >= 3)].index)

        return z_score_rec, outliers
