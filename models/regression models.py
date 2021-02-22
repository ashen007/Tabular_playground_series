import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


class OLS:
    @staticmethod
    def fit(x, y):
        """
        :param x: predicts features
        :param y: predicting feature
        :return: coefficients and intercept
        """
        theta, intercept = np.linalg.lstsq(x, y, rcond=None)[0]
        return theta, intercept

    @staticmethod
    def prediction(x, theta, intercept):
        predict_y = np.dot(x, theta) + intercept
        return predict_y

    @staticmethod
    def score(y_true, y_predict):
        r2Score = r2_score(y_true,y_predict)
        mse = mean_squared_error(y_true,y_predict)
        return r2Score,mse
