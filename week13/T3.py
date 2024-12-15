import numpy as np
import matplotlib.pyplot as plt

def grid(num_of_points):
    A = np.zeros((num_of_points, num_of_points))
    return A

def initial_grid(x, y, numbers):


    space = grid(numbers+1)
    x_range = np.linspace(x[0], x[1], numbers+1)


    space[0, :] = np.sin(np.pi * x_range)

    space[numbers, :] = 0.0

    space[:, 0] = 0.0

    space[:, numbers] = 0.0

    return space


def get_A_b(x, y, numbers):
    """
    Construct the coefficient matrix A and right-hand side vector b for
    the interior points using a 5-point stencil for the Laplacian.
    Interior points: i=1,...,numbers-1; j=1,...,numbers-1.
    """
    dimension_A = (numbers - 1) ** 2
    A = np.zeros((dimension_A, dimension_A))

    for i_idx in range(dimension_A):
        A[i_idx, i_idx] = -4

        if i_idx % (numbers - 1) != 0:
            A[i_idx, i_idx - 1] = 1

        if (i_idx + 1) % (numbers - 1) != 0:
            A[i_idx, i_idx + 1] = 1

        if i_idx - (numbers - 1) >= 0:
            A[i_idx, i_idx - (numbers - 1)] = 1

        if i_idx + (numbers - 1) < dimension_A:
            A[i_idx, i_idx + (numbers - 1)] = 1

    b = np.zeros(dimension_A)
    x_range = np.linspace(x[0], x[1], numbers+1)
    sin_x = np.sin(np.pi * x_range)

    for i in range(1, numbers):
        idx = i - 1
        b[idx] -= sin_x[i]

    return A, b


def CG(A, b, x0=None, tol=1e-15):
    if x0 is None:
        x0 = np.zeros(len(b))
    iteration_time = 0
    x = x0
    r = b - A @ x
    p = r.copy()
    while True:
        q = A @ p
        alpha = (r @ r) / (p @ q)
        x = x + alpha * p
        iteration_time += 1
        r_new = r - alpha * q
        if np.linalg.norm(r_new, ord=2) < tol:
            r = r_new
            break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return x, iteration_time


def plot_hot(x, y, numbers):

    tol = 1e-9
    space = initial_grid(x, y, numbers)
    A, b = get_A_b(x, y, numbers)
    sol, iteration_time = CG(A, b, tol=tol)

    for i in range(1, numbers):
        for j in range(1, numbers):
            idx = (i - 1) * (numbers - 1) + (j - 1)
            space[i, j] = sol[idx]

    plt.figure(figsize=(8, 6))
    plt.imshow(space, cmap='hot', interpolation='nearest', 
               extent=[x[0], x[1], y[0], y[1]], origin='lower', aspect='equal')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Solving Laplace equation using CG method, {iteration_time} iterations ,\n {numbers}x{numbers} grid, tol={tol}')
    plt.show()


if __name__ == '__main__':

    x = [0, 1]
    y = [0, 1]
    numbers = 128  
    plot_hot(x, y, numbers)
