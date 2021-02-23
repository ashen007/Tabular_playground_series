import numpy as np
import pandas as pd


class data_:
    """
    input: data file path
    read data files, calculate z-score and can drop outliers
    """
    def __init__(self, path, drop_outlier=True):
        self.path = path
        self.outlier = set()
        self.drop = drop_outlier

    def read(self):
        """
        :input: file path where csv files are
        :return: dataframe with index id column"""
        return pd.read_csv(self.path, index_col='id')

    def zScore(self, data):
        """
        :param data: dataframe
        :return: outlier count by each feature(dict), outlier indexes
        """
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
        """
        :param data: dataframe
        :return: no returns inplace drop of outliers
        """
        if self.drop:
            data.drop(self.outlier, axis=0, inplace=True)
