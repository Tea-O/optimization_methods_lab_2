from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt

# Генерируем синтетические данные для линейной регрессии
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 2.6 * X + 3.9 + np.random.randn(100, 1)  # Измененное уравнение


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def stepped_lr(epoch, lr):
    if epoch % 20 == 0:
        lr *= 0.9
    return lr



#@profile
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=100, batch_size=1, eps=1e-3, momentum=0.5):
    n_samples, n_features = X.shape
    theta = np.random.randn(1, 1)
    velocity = np.zeros((1, 1))
    theta_history = []

    for epoch in range(100):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            gradient = -2 / batch_size * X_batch.T.dot(y_batch - X_batch.dot(theta))
            ##
            velocity = momentum * velocity + learning_rate * gradient

            theta -= velocity
        learning_rate = stepped_lr(epoch, learning_rate)

        y_pred = X.dot(theta)
        mse = mean_squared_error(y, y_pred)
        print(f'Epoch {epoch + 1}/{n_epochs}, MSE: {mse:.4f}')

        theta_history.append(theta.copy())
        if np.linalg.norm(gradient) < eps:
            break

    return theta, theta_history


final_theta, theta_history = stochastic_gradient_descent(X, y, learning_rate=0.1, n_epochs=100, batch_size=10)


plt.scatter(X, y, label='Data Points')  # Точки данных
plt.plot(X, X.dot(final_theta), color='red', label='Linear Regression')  # Линейная регрессия
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Stochastic Gradient Descent')
plt.legend()
plt.show()
