import numpy as np
import matplotlib.pyplot as plt

# Генерируем синтетические данные для линейной регрессии
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 2.6 * X + 3.9 + np.random.randn(100, 1)  # Измененное уравнение

# Функция для вычисления среднеквадратичной ошибки
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Функция для обучения линейной регрессии с использованием SGD
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=100, batch_size=1):
    n_samples, n_features = X.shape
    theta = np.random.randn(1, 1)  # Инициализируем случайными значениями

    theta_history = []  # Для отслеживания истории параметров

    for epoch in range(n_samples):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            gradient = -2 / batch_size * X_batch.T.dot(y_batch - X_batch.dot(theta))
            theta -= learning_rate * gradient

        # Вычисляем среднеквадратичную ошибку на всей выборке
        y_pred = X.dot(theta)
        mse = mean_squared_error(y, y_pred)
        print(f'Epoch {epoch + 1}/{n_epochs}, MSE: {mse:.4f}')

        theta_history.append(theta.copy())  # Сохраняем параметры на каждой итерации

    return theta, theta_history

# Функция для визуализации регрессии и точек на каждой итерации
def plot_regression(X, y, theta_history):
    n_epochs = len(theta_history)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label='Данные', alpha=0.6)
    plt.xlabel('Признак')
    plt.ylabel('Целевая переменная')
    plt.title(f'Итерация {n_epochs}/{n_epochs}')

    y_pred = X.dot(theta_history[n_epochs - 1])
    plt.plot(X, y_pred, 'r-', label='Линейная регрессия')
    plt.legend()
    plt.show()

# Обучение с разными размерами батча
batch_sizes = [1, 5, 10, 50, 100]  # 1 - SGD, 2 - Minibatch GD, len(X) - GD
for batch_size in batch_sizes:
    print(f"Batch size: {batch_size}")
    theta, theta_history = stochastic_gradient_descent(X, y, batch_size=batch_size)
    print(f"Final theta: {theta}")
    plot_regression(X, y, theta_history)
