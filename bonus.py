import numpy as np
import matplotlib.pyplot as plt

def L1(L, w):
    return np.sum(np.abs(w)) * L


def L2(L, w):
    return np.sum(w * w.T) * L


def elastic(l1, l2, w):
    return L1(l1, w) + L2(l2, w)


def SGD_with_adam(X, y, deg, alpha, beta, ll, learning_rate=0.01, n_epochs=1000, batch_size=10, eps=1e-3):
    n_samples, n_features = X.shape
    theta = np.ones(deg + 1)
    theta_history = [theta]
    m = 0
    v = 0

    def get_gradient(w, batch_indexes):
        gradient = np.zeros_like(w)
        for i in range(batch_indexes):
            for j in range(len(w)):
                gradient[j] += ((sum([w[q] * X[i] ** q for q in range(len(w))]) - y[i])
                                * (X[i] ** j))
        gradient *= 2 / n_samples

        gradient += L2(ll, w) / n_samples
        return gradient

    for epoch in range(1, n_epochs + 1):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            gradient = get_gradient(theta,batch_size)
            m = alpha * m + (1 - alpha) * gradient
            v = beta * v + (1 - beta) * (gradient * gradient.T)
            mHat = m / (1 - alpha ** epoch)
            vHat = v / (1 - beta ** epoch)

            theta -= learning_rate * mHat / (np.sqrt(vHat) + eps)
            theta_history.append(theta)
        if np.linalg.norm(gradient) < eps:
            break
    return theta, theta_history

coefA = 2
coefB = -9
coefC = 3
coefD = 6
coefE = 10
def reg(x):
    return coefA * x ** 4 + coefB * x ** 3 + coefC * x ** 2 + coefD * x + coefE


def generate_points(f, num, max_x=2, disp=0):
    X = np.random.uniform(low=-max_x, high=max_x, size=num)
    X.sort()
    Y = f(X)
    max_y = 4000
    for i in range(len(Y)):
        Y[i] = Y[i] + np.random.uniform(-max_y, max_y)
    return X, Y



# Generate 10 random data points
num_points = 50
maxX = 10
X, Y = generate_points(reg, num_points, max_x= maxX)

# Perform linear regression using SGD with Adam
deg = 5  # Degree of the regression polynomial (linear regression)
alpha = 0.9  # Adam parameter alpha
beta = 0.999  # Adam parameter beta
learning_rate = 0.01
n_epochs = 1000
batch_size = 10
eps = 1e-3
theta, theta_history = SGD_with_adam(X[:, np.newaxis], Y, deg, alpha, beta,0.2, learning_rate, n_epochs, batch_size, eps)

# Create a range of X values for plotting the regression line
X_range = np.linspace(-maxX, maxX, 100)
Y_pred = np.polyval(theta[::-1], X_range)  # Evaluate the regression polynomial

# Plot the points and the regression line
plt.scatter(X, Y, label='Data Points')
plt.plot(X_range, Y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression using SGD with Adam, Deg: ' + str(deg))
plt.grid(True)
plt.savefig('metOptLab3TaskBonus' + str(deg)+ '.png')
plt.show()
