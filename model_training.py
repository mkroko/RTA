from sklearn.datasets import load_iris
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


iris = load_iris()
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])


class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

model = Perceptron(0.001, 1000)

X = iris.iloc[:, 0:2].values
Y = iris.iloc[:, 4].values
X_t, X_test, Y_t, Y_test = train_test_split(X, Y, test_size=0.3)

X_t = X_t.astype(float)
Y_t = Y_t.astype(float)

model.fit(X_t, Y_t)

with open('mod.pkl', 'wb') as model1:
    pickle.dump(model, model1)
