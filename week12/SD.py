import numpy as np

def steepest_descent(A, b, x0, tol=1e-16, max_iter = np.inf):
    """
    use steepest descent method to solve Ax = b, A is a symmetric positive matrix
    :param A: a symmetric positive matrix
    :param b: a vector
    :param x0: initial guess
    :return: x, the solution of Ax = b, according to the tol
    :return: history, the history of the iteration
    """
    n1, n2 = A.shape
    n3 = b.shape[0]
    should_zero = np.linalg.norm(A - A.T)
    if n1 != n2 or should_zero != 0 or n1 != n3:
        print("Error: invalid input")
        return None
    r = b - np.dot(A, x0)
    x = x0
    iteration_time = 0
    history = [x0]

    while iteration_time < max_iter:
        if np.linalg.norm(r) < tol:
            break
        Ark = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ark)
        x = x + alpha * r
        r = b - np.dot(A, x)
        iteration_time += 1
        history.append(x)

    return x, history

# main
A = np.array([[20, 0], [0, 1]])
b = np.array([0, 0]).T
x0 = np.array([1, 5]).T
iteration_time = 10
_ , history = steepest_descent(A, b, x0, max_iter= iteration_time)

# Plot
import matplotlib.pyplot as plt
history = np.array(history)
plt.figure(figsize=(12, 8))

# Plot the history points
for i, (x1, x2) in enumerate(history):
    plt.scatter(x1, x2, color="blue", s=35)
    plt.text(x1, x2, str(i), fontsize=12, color="blue", ha="left")

# Plot the true solution (0, 0)
plt.scatter(0, 0, color="blue", marker="x", s=70, label="True Solution (0, 0)")

# Configure the plot
plt.title(f"A special linear equations, with Steepest Descent Iteration {iteration_time} times")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid()
plt.legend()
plt.show()