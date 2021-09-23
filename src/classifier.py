import numpy as np


def _y_to_svm_format(y):
    return np.array([1 if y_item > 0 else -1 for y_item in y])


class Classifier:

    def __init__(self, c, lr, kernel, steps):
        self._w = None
        self._c = c
        self._lr = lr
        self._steps = steps
        self._x_fit = None
        self._kernel = kernel

    def fit(self, x, y):
        self._x_fit = x
        y = _y_to_svm_format(y)
        x_kernel = self._apply_kernel(x)

        x_bias = np.ones((x.shape[0], 1)).reshape(-1, 1)
        x_a = np.hstack((x_kernel, x_bias))

        self._w = np.random.random(x_a.shape[1])

        n = x_a.shape[0]
        for step in range(self._steps):
            for index, x_item in enumerate(x_a):
                y_item = y[index]
                gradient = (1 / n) * self.gradient(x_item, y_item)
                self._w -= self._lr * gradient

            print(f"Step {step}. Cost {self.cost(x_a, y)} ")

    def predict(self, x):
        x_bias = np.ones((x.shape[0], 1)).reshape(-1, 1)
        x_kernel = self._apply_kernel(x)
        x_a = np.hstack((x_kernel, x_bias))
        predictions = [self.h(x_item) for x_item in x_a]
        return np.array([1 if p >= 0 else 0 for p in predictions])

    def cost(self, x: np.ndarray, y: np.ndarray):
        n = x.shape[0]
        regularization_loss = 0.5 * sum(self._w**2)

        hinge_losses = [self._c * self.hinge_loss(y[index], x_item) for index, x_item in enumerate(x)]
        hinge_loss = (1 / n) * np.sum(hinge_losses)
        return regularization_loss + hinge_loss

    def hinge_loss(self, y_i, x_i):
        return np.max([0, 1 - y_i * self.h(x_i)])

    def h(self, x_i):
        return np.dot(x_i, self._w)

    def gradient(self, x_item, y_item):
        hinge_loss = self.hinge_loss(y_item, x_item)
        if hinge_loss == 0:
            return self._w
        return self._w - self._c * y_item * x_item

    def _apply_kernel(self, x):
        return self._kernel(x, self._x_fit)

    @property
    def weights(self):
        return self._w
