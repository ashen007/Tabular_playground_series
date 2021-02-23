import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


class Evaluate:
    @staticmethod
    def prediction(x, theta, intercept):
        """
        :param x: predicts
        :param theta: coefficients
        :param intercept: intercept
        :return: predicted y values
        """
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        predict_y = np.dot(x, theta)
        return predict_y

    @staticmethod
    def score(y_true, y_predict):
        """
        :param y_true: real target feature values
        :param y_predict: model predicted target values
        :return: square error and mean square error
        """
        r2Score = r2_score(y_true, y_predict)
        mse = mean_squared_error(y_true, y_predict)
        return r2Score, mse


class OLS(Evaluate):
    @staticmethod
    def fit(x, y):
        """
        :param x: predicts features
        :param y: predicting feature
        :return: coefficients and intercept
        """
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        theta = np.linalg.lstsq(x, y, rcond=None)[0]
        return theta


class GradientDecent(Evaluate):
    @staticmethod
    def cost_function(x, y, theta):
        cost = np.sum((np.dot(x, theta) - y) ** 2) / 2 * len(y)
        return cost

    @staticmethod
    def fit(x, y, learning_rate, epochs):
        theta = np.zeros(x.shape[1] + 1)
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        cost_ = []

        for i in range(epochs):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / len(y)
            theta = theta - gradient * learning_rate
            cost = GradientDecent.cost_function(x, y, theta)
            cost_.append(cost)

        return theta, cost_


class RidgeRegression(Evaluate):
    def __init__(self, alpha):
        self.alpha = alpha

    def cost_function(self, x, y, theta):
        cost = (np.sum((np.dot(x, theta) - y) ** 2) + self.alpha * np.sum(theta ** 2)) / 2 * len(y)
        return cost

    def fit(self, x, y, learning_rate, epochs):
        theta = np.zeros(x.shape[1] + 1)
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        cost_ = []

        for i in range(epochs):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = (np.dot(x.T, loss) + self.alpha * theta) / len(y)
            theta = theta - gradient * learning_rate
            cost = self.cost_function(x, y, theta)
            cost_.append(cost)

        return theta, cost_


class LassoRegression(Evaluate):
    def __init__(self, alpha):
        self.alpha = alpha

    def cost_function(self, x, y, theta):
        cost = (np.sum((np.dot(x, theta) - y) ** 2) + self.alpha * np.sum(np.abs(theta))) / 2 * len(y)
        return cost

    def fit(self, x, y, learning_rate, epochs):
        theta = np.zeros(x.shape[1] + 1)
        ones = np.ones(x.shape[0])
        x = np.hstack((np.asarray(x), ones.reshape(-1, 1)))
        cost_ = []

        for i in range(epochs):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.zeros(x.shape[1])

            for j in range(x.shape[1]):
                if theta[j] > 0:
                    gradient[j] = (np.dot(x[:, j].T, loss) + self.alpha / 2) / len(y)
                else:
                    gradient[j] = (np.dot(x[:, j].T, loss) - self.alpha / 2) / len(y)

            theta = theta - gradient * learning_rate
            cost = self.cost_function(x, y, theta)
            cost_.append(cost)

        return theta, cost_
