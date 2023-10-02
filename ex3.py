# # #moment
from memory_profiler import profile, memory_usage
import psutil
# # Define the quadratic function (example: f(x) = ax^2 + bx + c)
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from datetime import datetime
nameFunc = 'F3'
actual_min = [1, -4]
def calculate_actual_min(final_res):
    ans = ((abs(actual_min[0]/final_res[0]) + abs(actual_min[1]/final_res[1]))/2)
    if(ans > 1):
        return (2 - ans)
    return ans
# Getting % usage of virtual_memory ( 3rd field)

# Define the 2D quadratic function
def quadratic_function(x):
    return 2 * (x[0] - 1) ** 2 + 2 * (x[1] + 4) ** 2


# Define the gradient of the 2D quadratic function
def gradient_quadratic(x):
    return np.array(nd.Gradient(quadratic_function)(x))

memcoef = 1024 * 1024
def momentum_gradient_descent(x, learning_rate, momentum, num_iterations, eps=1e-3):
    start_time = datetime.now()
    momentum_mem = 0
    prevMem = psutil.virtual_memory()[3] / 1000000000
    points = np.zeros((num_iterations, len(x)))
    velocity = 0
    momentum_itr = 0
    final_point = [0, 0]
    arif_operation = 0
    for i in range(num_iterations):
        prevMem = psutil.virtual_memory()[3] / 1000000000
        grad = gradient_quadratic(x)
        velocity = momentum * velocity + learning_rate * grad
        points[i] = x
        final_point = x
        x -= velocity
        arif_operation += 4
        if np.linalg.norm(grad) < eps:
            break
        momentum_itr += 1
        curMem = psutil.virtual_memory()[3] / 1000000000
        if (curMem - prevMem > 0):
            momentum_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('Momentum acc = ' + str(calculate_actual_min(final_point)))
    print('Momentum time = ' + 'Duration: {}'.format(end_time - start_time))
    print('Momentum iteration = ' + str(momentum_itr))
    print('Momentum mem(KB) = ' + str(momentum_mem * memcoef))
    print('Momentum arifm = ' + str(arif_operation))
    return points[:i + 1], i + 1

# Hyperparameters for momentum gradient descent
initial_x = -4.0
initial_y = 10.0
learning_rate = 0.01
momentum = 0.8
num_iterations = 1000

# Perform momentum gradient descent
points_momentum, iter = momentum_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
#print(points_momentum, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# # Plot the contour lines
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
# # Plot the gradient descent path
# plt.plot(points_momentum[:, 0], points_momentum[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
#
# # Mark the starting point
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with momentum')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45Momentum' + nameFunc + '.png')
#plt.show()


# nesterov
def nesterov_gradient_descent(x, learning_rate, alpha, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    velocity = 0
    nesterov_itr = 0
    nesterov_mem = 0
    start_time = datetime.now()
    prevMem = psutil.virtual_memory()[3] / 1000000000
    final_point = [0, 0]
    arif_operation = 0
    for i in range(num_iterations):
        prevMem = psutil.virtual_memory()[3] / 1000000000
        grad = gradient_quadratic(x - alpha * velocity)
        velocity = alpha * velocity + learning_rate * grad
        points[i] = x
        final_point = x
        x -= velocity
        nesterov_itr += 1
        arif_operation += 7
        if np.linalg.norm(grad) < eps:
            break
        curMem = psutil.virtual_memory()[3] / 1000000000

        if (curMem - prevMem > 0):
            nesterov_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('Nesterov acc = ' + str(calculate_actual_min(final_point)))
    print('Nesterov time = ' + 'Duration: {}'.format(end_time - start_time))
    print('Nesterov iteration ' + str(nesterov_itr))
    print('Nesterov mem(KB) = ' + str(nesterov_mem * memcoef))
    print('Nesterov arifm = ' + str(arif_operation))
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent

learning_rate = 0.01
momentum = 0.8
# num_iterations = 100

# Perform momentum gradient descent
points_nesterov, iter = nesterov_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
#print(points_nesterov, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# Plot the contour lines
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
# # Plot the gradient descent path
# plt.plot(points_nesterov[:, 0], points_nesterov[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
#
# # Mark the starting point
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with nesterov')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45Nesterov' + nameFunc + '.png')
#plt.show()


# AdaGrad
def adagrad_gradient_descent(x, learning_rate, alpha, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    G = 0
    adagrad_itr = 0
    nesterov_mem = 0
    adagrad_mem = 0
    prevMem = psutil.virtual_memory()[3] / 1000000000
    start_time = datetime.now()
    final_point = [0, 0]
    arif_operation = 0
    for i in range(num_iterations):
        prevMem = psutil.virtual_memory()[3] / 1000000000
        grad = gradient_quadratic(x)
        G += grad * grad.T
        points[i] = x
        final_point = x
        x -= learning_rate * grad / np.sqrt(G + eps)
        adagrad_itr += 1
        arif_operation += (grad.size * grad.size)
        arif_operation += grad.size * 2
        arif_operation += 3
        if np.linalg.norm(grad) < eps:
            break
        curMem = psutil.virtual_memory()[3] / 1000000000

        if (curMem - prevMem > 0):
            adagrad_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('AdaGrad acc = ' + str(calculate_actual_min(final_point)))
    print('Adagrad time = ' + 'Duration: {}'.format(end_time - start_time))
    print('Adagrad itr ' + str(adagrad_itr))
    print('AdaGrad mem(KB) = ' + str(adagrad_mem * memcoef))
    print('AdaGrad arifm = ' + str(arif_operation))
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent

learning_rate = 7
momentum = 0.8
# num_iterations = 100

# Perform momentum gradient descent
points_adagrad, iter = adagrad_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
#print(points_adagrad, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 2000, 125)

# Plot the contour lines
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
# # Plot the gradient descent path
# plt.plot(points_adagrad[:, 0], points_adagrad[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
#
# # Mark the starting point
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with adagrad')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45adagrad' + nameFunc + '.png')
#plt.show()


# RMSProp
def rmsprop_gradient_descent(x, learning_rate, gamma, num_iterations, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    G = 0
    alpha = 0
    rmsprop_itr = 0
    rmsprop_mem = 0
    start_time = datetime.now()
    final_point = 0
    arif_operation = 0
    for i in range(num_iterations):
        prevMem = psutil.virtual_memory()[3] / 1000000000
        grad = gradient_quadratic(x)
        G = grad * grad.T
        alpha = gamma * alpha + (1 - gamma) * G
        points[i] = x
        final_point = x
        x -= learning_rate * grad / np.sqrt(alpha + eps)
        rmsprop_itr += 1
        arif_operation += (grad.size * grad.size)
        arif_operation += 4
        arif_operation += (grad.size + 2)
        if np.linalg.norm(grad) < eps:
            break
        curMem = psutil.virtual_memory()[3] / 1000000000
        if (curMem - prevMem > 0):
            rmsprop_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('rmsprop acc = ' + str(calculate_actual_min(final_point)))
    print('rmsprop time = ' + 'Duration: {}'.format(end_time - start_time))
    print('rmsprop iteration ' + str(rmsprop_itr) + '\n')
    print('RMSprop mem(KB) = ' + str(rmsprop_mem * memcoef))
    print('RMSprop arifm = ' + str(arif_operation))
    return points[:i + 1], i + 1


# Hyperparameters for momentum gradient descent
learning_rate = 0.6
momentum = 0.9
# num_iterations = 40

# Perform momentum gradient descent
points_rmsprop, iter = rmsprop_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
#print(points_rmsprop, iter)
# Generate contour plot of the quadratic function

# Generate contour plot of the quadratic function
x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))

# Create a colormap for contour lines
levels = np.arange(0, 3000, 125)

# # Plot the contour lines
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
# # Plot the gradient descent path
# plt.plot(points_rmsprop[:, 0], points_rmsprop[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
#
# # Mark the starting point
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with RMSprop')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45RMSProp' + nameFunc + '.png')
#plt.show()


# adam
memory = []
prevMem =  psutil.virtual_memory()[3] / 1000000000
momentum_mem = 0
def adam_gradient_descent(x, alpha, beta, num_iterations, lr=0.7, eps=1e-3):
    points = np.zeros((num_iterations, len(x)))
    m = 0
    v = 0
    adam_itr = 0
    adam_mem = 0
    start_time = datetime.now()
    final_point = [0, 0]
    arif_operation = 0
    for i in range(1, num_iterations + 1):
        prevMem = psutil.virtual_memory()[3] / 1000000000
        grad = gradient_quadratic(x)
        G = grad * grad.T
        m = alpha * m + (1 - alpha) * grad
        v = beta * v + (1 - beta) * G

        vHat = v / (1 - beta ** i)
        mHat = m / (1 - alpha ** i)
        points[i - 1] = x
        final_point = x
        x -= lr * mHat / (np.sqrt(vHat) + eps)
        adam_itr += 1
        arif_operation += (i + 4)
        arif_operation += grad.size
        arif_operation += 6
        arif_operation += (grad.size * grad.size)
        if np.linalg.norm(grad) < eps:
            break
        curMem = psutil.virtual_memory()[3] / 1000000000
        if (curMem - prevMem > 0):
            adam_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('Adam acc = ' + str(calculate_actual_min(final_point)))
    print('Adam time = ' + 'Duration: {}'.format(end_time - start_time))
    print('Adam iteration: ' + str(adam_itr) + '\n')
    print('Adam mem(KB) = ' + str(adam_mem * memcoef) + '\n')
    print('Adam arifm = ' + str(arif_operation))
    return points[:i], i



learning_rate = 0.8
momentum = 0.999
# num_iterations = 100

points_adam, iter = adam_gradient_descent(np.array([initial_x, initial_y]), learning_rate, momentum, num_iterations)
#print(points_adam, iter)
#print(str(memory) + 'Adam memory \n')

x = np.linspace(-4, 30, 500)
y = np.linspace(-4, 14, 500)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(np.array([X, Y]))


# levels = np.arange(0, 3000, 125)
#
#
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
#
# plt.plot(points_adam[:, 0], points_adam[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
#
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with adam')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45Adam' + nameFunc + '.png')
# #plt.show()
# levels = np.arange(0, 3000, 125)



eposh = 200
st = np.array([initial_x, initial_y])
points = np.zeros((eposh, 2))
points[0] = st
xscale = 1.5
yscale = 0.4
def F(x):
    return 2 * (x[0] - 1) ** 2 + 2 * (x[1] + 4) ** 2
def Grad(func, x):
    return np.array(nd.Gradient(func)(x))


def Gr_m(x1, x2):
    i = 0
    iteration = 1
    alpha = 0.03  # Шаг сходимости
    eps = 0.0000001  # точность
    max_iteration = num_iterations #максимальное количество итераций
    X_prev = np.array([x1, x2])
    X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])
    gd_iteration = 0
    gd_mem = 0
    start_time = datetime.now()
    final_point = [0, 0]
    arif_operation = 0
    while np.linalg.norm(X - X_prev) > eps:
        prevMem = psutil.virtual_memory()[3] / 1000000000
        X_prev = X.copy()
        i = i + 1
        #print(i, ":", X)
        X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])  # Формула
        points[iteration] = X
        arif_operation += 4
        gd_iteration += 1
        if iteration > max_iteration:
            break
        curMem = psutil.virtual_memory()[3] / 1000000000
        final_point = X
        if (curMem - prevMem > 0):
            gd_mem += (curMem - prevMem)
    end_time = datetime.now()
    print('GD acc = ' + str(calculate_actual_min(final_point)))
    print('GD time = ' + 'Duration: {}'.format(end_time - start_time))
    print("GD iteration:", gd_iteration)
    print('GD mem(KB) = ' + str(gd_mem * memcoef) + '\n')
    print('GD arifm = ' + str(arif_operation))
    return X
result = Gr_m(st[0], st[1])
# points_GD= points[~np.all(points == 0, axis=1)]
# #asdasd
# levels = np.arange(0, 3000, 125)
# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
# plt.plot(points_GD[:, 0], points_GD[:, 1], '-o', color='red', label='Gradient Descent', linewidth=1, markersize=3)
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with GD')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45GD' + nameFunc + '.png')
# #plt.show()
# #print(result)


# contour = plt.contour(X, Y, Z, levels=levels, colors=plt.cm.jet(np.linspace(0, 1, len(levels))), linewidths=0.5, )
# plt.colorbar(contour, label='Function Value')
#
#
# plt.plot(points_adam[:, 0], points_adam[:, 1], '-o', color='red', label='Adam', linewidth=1, markersize=3)
# plt.plot(points_nesterov[:, 0], points_nesterov[:, 1], '-o', color='blue', label='Nesterov', linewidth=1, markersize=3)
# plt.plot(points_momentum[:, 0], points_momentum[:, 1], '-o', color='green', label='Momentum', linewidth=1, markersize=3)
# plt.plot(points_adagrad[:, 0], points_adagrad[:, 1], '-o', color='black', label='Adagrad', linewidth=1, markersize=3)
# plt.plot(points_rmsprop[:, 0], points_rmsprop[:, 1], '-o', color='yellow', label='RMSprop', linewidth=1, markersize=3)
# plt.plot(points_GD[:, 0], points_GD[:, 1], '-o', color='purple', label='GD', linewidth=1, markersize=3)
# plt.scatter([initial_x], [initial_y], color='green', marker='o', s=50, label='Start', zorder=5)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Gradient Descent on 2D Quadratic Function with all')
# plt.grid(True)
# plt.xlim(-4, 30)  # Adjust the x-axis limits
# plt.ylim(-4, 14)  # Adjust the y-axis limits
# plt.savefig('metOptLab3Task45All' + nameFunc + '.png')
#plt.show()
