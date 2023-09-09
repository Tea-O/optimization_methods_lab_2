# # #moment
from memory_profiler import profile, memory_usage
# # Define the quadratic function (example: f(x) = ax^2 + bx + c)
import numpy as np
import matplotlib.pyplot as plt


# Define the 2D quadratic function
def quadratic_function(x):
    return 5 * (x[0] - 2) ** 2 + (x[1] - 3) ** 2


# Define the gradient of the 2D quadratic function
def gradient_quadratic(x):
    return np.array([10 * x[0] - 20, 2 * x[1] - 6])


def momentum_gradient_descent(x, learning_rate, momentum, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))

    velocity = 0

    for i in range(num_iterations):
        grad = gradient_quadratic(x)
        velocity = momentum * velocity + learning_rate * grad
        points[i] = x
        x -= velocity

        if np.linalg.norm(grad) < eps:
            break
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent
initial_x = -4.0
initial_y = 3.0
learning_rate = 0.01
momentum = 0.8
num_iterations = 1000

# Perform momentum gradient descent
points, iter = momentum_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
print(points, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# Plot the contour lines
contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
plt.colorbar(contour, label='Function Value')

# Plot the gradient descent path
plt.plot(points[:, 0], points[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)

# Mark the starting point
plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent on 2D Quadratic Function')
plt.grid(True)
plt.xlim(-4, 30)  # Adjust the x-axis limits
plt.ylim(-4, 14)  # Adjust the y-axis limits
plt.show()


# nesterov
def nesterov_gradient_descent(x, learning_rate, alpha, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))

    velocity = 0

    for i in range(num_iterations):
        grad = gradient_quadratic(x - alpha * velocity)
        velocity = alpha * velocity + learning_rate * grad
        points[i] = x
        x -= velocity

        if np.linalg.norm(grad) < eps:
            break
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent

learning_rate = 0.01
momentum = 0.8
num_iterations = 1000

# Perform momentum gradient descent
points, iter = nesterov_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
print(points, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# Plot the contour lines
contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
plt.colorbar(contour, label='Function Value')

# Plot the gradient descent path
plt.plot(points[:, 0], points[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)

# Mark the starting point
plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent on 2D Quadratic Function')
plt.grid(True)
plt.xlim(-4, 30)  # Adjust the x-axis limits
plt.ylim(-4, 14)  # Adjust the y-axis limits
plt.show()


# AdaGrad
def adagrad_gradient_descent(x, learning_rate, alpha, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    G = 0

    for i in range(num_iterations):
        grad = gradient_quadratic(x)
        G += grad * grad.T

        points[i] = x
        x -= learning_rate * grad / np.sqrt(G + eps)

        if np.linalg.norm(grad) < eps:
            break
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent

learning_rate = 7
momentum = 0.8
num_iterations = 1000

# Perform momentum gradient descent
points, iter = adagrad_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
print(points, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# Plot the contour lines
contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
plt.colorbar(contour, label='Function Value')

# Plot the gradient descent path
plt.plot(points[:, 0], points[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)

# Mark the starting point
plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent on 2D Quadratic Function')
plt.grid(True)
plt.xlim(-4, 30)  # Adjust the x-axis limits
plt.ylim(-4, 14)  # Adjust the y-axis limits
plt.show()


# RMSProp
def rmsprop_gradient_descent(x, learning_rate, gamma, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    G = 0
    alpha = 0

    for i in range(num_iterations):
        grad = gradient_quadratic(x)
        G = grad * grad.T
        alpha = gamma * alpha + (1 - gamma) * G
        points[i] = x
        x -= learning_rate * grad / np.sqrt(alpha + eps)

        if np.linalg.norm(grad) < eps:
            break
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent
learning_rate = 0.6
momentum = 0.9
num_iterations = 1000

# Perform momentum gradient descent
points, iter = rmsprop_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
print(points, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 3000, 125)

# Plot the contour lines
contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
plt.colorbar(contour, label='Function Value')

# Plot the gradient descent path
plt.plot(points[:, 0], points[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)

# Mark the starting point
plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent on 2D Quadratic Function')
plt.grid(True)
plt.xlim(-4, 30)  # Adjust the x-axis limits
plt.ylim(-4, 14)  # Adjust the y-axis limits
plt.show()


# adam
memory = []

def adam_gradient_descent(x, alpha, beta, num_iterations, lr=0.7, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    m = 0
    v = 0
    for i in range(1, num_iterations + 1):
        grad = gradient_quadratic(x)
        G = grad * grad.T
        m = alpha * m + (1 - alpha) * grad
        v = beta * v + (1 - beta) * G

        vHat = v / (1 - beta ** i)
        mHat = m / (1 - alpha ** i)
        points[i - 1] = x
        x -= lr * mHat / (np.sqrt(vHat) + eps)
        memory.append(memory_usage(max_usage=True))
        if np.linalg.norm(grad) < eps:
            break
    return points[:i], i



learning_rate = 0.8
momentum = 0.999
num_iterations = 1000

points, iter = adam_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
print(points, iter)
print(memory)

x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))


levels = np.arange(0, 3000, 125)


contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
plt.colorbar(contour, label='Function Value')


plt.plot(points[:, 0], points[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)

plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent on 2D Quadratic Function')
plt.grid(True)
plt.xlim(-4, 30)  # Adjust the x-axis limits
plt.ylim(-4, 14)  # Adjust the y-axis limits
plt.show()

