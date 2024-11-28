import numpy as np
import matplotlib.pyplot as plt


def grid(numbers):
    A = np.zeros((numbers, numbers))
    return A


def initial_grid(x, y, numbers):
    """
    :param x: x_range, such as [0, 1]
    :param y: y_range, such as [0, 1]
    :param numbers: number of points in every direction
    """
    A = grid(numbers)
    # set boundary conditions, others default to 0
    for k in range(numbers):
        A[0][k] = 0
        A[numbers-1][k] = 0
        A[k][numbers-1] = 0
    x_range = np.linspace(x[0], x[1], numbers)
    A[:, 0] = np.sin(np.pi * x_range)
    return A


def is_converged(A, new_A, tol=1e-9):
    n1, n2 = A.shape
    for i in range(n1):
        for j in range(n2):
            if abs(A[i][j] - new_A[i][j]) > tol:
                return False
    return True


def one_jacobi(A):
    n1, n2 = A.shape
    new_A = np.copy(A)
    new_A[1:n1-1, 1:n2-1] = (
        (A[0:n1-2, 1:n2-1] + A[2:n1, 1:n2-1] +
         A[1:n1-1, 0:n2-2] + A[1:n1-1, 2:n2]) / 4
    )
    return new_A


def jacobi(origin_A):
    A = np.copy(origin_A)
    iteration_time = 0
    while True:
        new_A = one_jacobi(A)
        iteration_time += 1
        if is_converged(A, new_A):
            break
        A = new_A
    return new_A, iteration_time


def one_gause_seidel(origin_A):
    A = np.copy(origin_A)
    n1, n2 = A.shape
    for i in range(1, n1-1):
        for j in range(1, n2 - 1):
            if (i+j) % 2 == 0:
                A[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4
    for i in range(1, n1 - 1):
        for j in range(1, n2 - 1):
            if (i+j) % 2 == 1:
                A[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4
    return A


def gause_seidel(origin_A):
    A = np.copy(origin_A)
    iteration_time = 0
    while True:
        new_A = one_gause_seidel(A)
        iteration_time += 1
        if is_converged(A, new_A):
            break
        A = new_A
    return new_A, iteration_time


def plot_hot(x, y, A, fun):
    converged_A, iteration_time = fun(A)
    plt.figure(figsize = (12, 8))
    plt.imshow(converged_A, cmap='hot', interpolation='nearest', extent=[x[0], x[-1], y[0], y[-1]])
    plt.colorbar()  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'solving Laplace equation using {fun.__name__} method, after {iteration_time} iterations')
    plt.show(block=False)


if __name__ == '__main__':
    x = [0, 1]
    y = [0, 1]
    numbers = 100
    A = initial_grid(x, y, numbers)
    plot_hot(x, y, A, jacobi)
    plot_hot(x, y, A, gause_seidel)
    plt.show()