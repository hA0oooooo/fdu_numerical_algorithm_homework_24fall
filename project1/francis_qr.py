import numpy as np
import matplotlib.pyplot as plt
import time

########### Francis_QR_Double_Shift Part 01: Calculation Function ###########

def house(x):
    """
    householder transformation to let x become a unit vector
    :param x: a vector
    :return: a matrix H such that Hx = ||x||e_1, and H has the corresponding size of x
    """
    n = len(x)
    e1 = np.zeros(n)
    e1[0] = 1
    v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
    v = v / np.linalg.norm(v)
    return np.eye(n) - 2 * np.outer(v, v)


def hessenberg(origin_A):
    """
    Use Householder transformation to transform a matrix to Hessenberg form
    :param A: a square matrix
    :return: Hessenberg matrix A, and the corresponding orthogonal matrix Q, Q A Q.T = Hessenberg matrix
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print('Input error')
        return None
    A = np.copy(origin_A)
    n = n1
    Q = np.eye(n)
    for k in range(n-2):
        x = A[k+1:n, k]
        H = house(x)
        A[k+1:n, k:n] = H @ A[k+1:n, k:n]
        A[0:n, k+1:n] = A[0:n, k+1:n] @ H
        Q[k+1:n, :] = H @ Q[k+1:n, :]

    return A, Q


def francis_one(H, Q_accum, i):
    """
    Fransic's double shift QR algorithm for Hessenberg matrix, iterate once
    :param H: a Hessenberg matrix
    :return: H hat, a Hessenberg matrix with the same eigenvalues as H
    """
    n = i
    m = n - 1
    # double shift
    s = H[m-1, m-1] + H[n-1, n-1]
    t = H[m-1, m-1] * H[n-1, n-1] - H[m-1, n-1] * H[n-1, m-1]
    x = H[0, 0] * H[0, 0] + H[0, 1] * H[1, 0] - s * H[0, 0] + t
    y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
    z = H[1, 0] * H[2, 1]

    # implicit QR
    for k in range(n-2):
        Q = house(np.array([x, y, z]))
        q = max(0, k-1)
        H[k:k+3, q:] = Q @ H[k:k+3, q:]
        H[:, k:k+3] = H[:, k:k+3] @ Q
        x = H[k+1, k]
        y = H[k+2, k]
        if k < n-3:
            z = H[k+3, k]
        # record the transformation matrix
        Q_accum[k:k+3, :] = Q @ Q_accum[k:k+3, :]

    Q = house(np.array([x, y]))
    H[n-2:n, n-3:] = Q @ H[n-2:n, n-3:]
    H[:, n-2:n] = H[:, n-2:n] @ Q
    # record the transformation matrix

    Q_accum[n-2:n, :] = Q @ Q_accum[n-2:n, :]

    return H, Q_accum


def hessenberg_to_schur_form(H, max_iter=np.inf, tol=1e-16):
    """
    turn a Hessenberg matrix to Schur form, from the right-bottom to the left-top
    """

    n = H.shape[0]
    Q_accum = np.eye(H.shape[0])
    if n <= 2:
        return H, Q_accum, 0
    m = n
    iteration_time = 0

    while m > 2 and iteration_time < max_iter:
        
        found_block = False
        for i in range(m-1, 0, -1):
            if abs(H[i, i-1]) < tol * (abs(H[i-1, i-1]) + abs(H[i, i])):
                H[i, i-1] = 0
                m = i
                found_block = True
                break

        if not found_block:

            H, Q_accum = francis_one(H, Q_accum, m)
        
        iteration_time += 1

    return H, Q_accum, iteration_time


def zeros_hessenberg(H):

    n = H.shape[0]
    for i in range(n-2):
        H[i+2:n, i] = 0

    return H


def generate_test_matrix(n, seed=None):

    if seed is not None:
        np.random.seed(seed) 

    A = np.random.randn(n, n) * 10  

    return A


def francis_double_shift_qr(A):
    """
    :return: T, the real schur form of A
    :return: Q, the orthogonalization transformation
    :return: iteration_time
    :return: running time
    Q T Q.T = A
    """
    start_time = time.time()
    H, Q1 = hessenberg(A)
    T, Q2, iteration_time = hessenberg_to_schur_form(H)
    T = zeros_hessenberg(T)
    end_time = time.time()
    running_time = end_time - start_time
    Q = Q1.T @ Q2.T

    return T, Q, iteration_time, running_time


########### Francis_QR_Double_Shift Part 02: Plot Figure Function ###########

def collect_information(dimensions, seed=None):
    """
    Collect metrics for various matrix dimensions.
    """
    information = {
        'dimension': [],
        'orthogonality_loss': [],
        'forward_abs_error': [],
        'forward_rel_error': [],
        'running_time': [],
        'iteration_time': [],
        'iterations_per_dimension': [],
        'iterations_per_time': []
    }

    for n in dimensions:

        A = generate_test_matrix(n, seed)
        norm_A = np.linalg.norm(A, ord='fro')
        T, Q, iteration_time, running_time = francis_double_shift_qr(A)

        orthogonality_loss = np.linalg.norm(Q.T @ Q - np.eye(n), ord='fro')
        forward_abs_error = np.linalg.norm(Q @ T @ Q.T - A, ord='fro')
        forward_rel_error = forward_abs_error / norm_A
        iterations_per_dimension = iteration_time / n
        iterations_per_time = iteration_time / running_time if running_time > 0 else np.inf

        information['dimension'].append(n)
        information['orthogonality_loss'].append(orthogonality_loss)
        information['forward_abs_error'].append(forward_abs_error)
        information['forward_rel_error'].append(forward_rel_error)
        information['running_time'].append(running_time)
        information['iteration_time'].append(iteration_time)
        information['iterations_per_dimension'].append(iterations_per_dimension)
        information['iterations_per_time'].append(iterations_per_time)
        
    return information


def plot_orthogonality_loss(dimensions, orthogonality_losses):
    """
    Plot the orthogonality loss ||Q.T Q - I||_F vs matrix dimension.
    Uses a logarithmic scale for the y-axis.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, orthogonality_losses, marker='o', linestyle='-', label='Orthogonality Loss')
    plt.title('Orthogonality Loss ||Q.T Q - I|| vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('||Q.T Q - I||_F')
    plt.yscale('log')  
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('orthogonality_loss.png') 
    plt.show(block=False)


def plot_forward_abs_error(dimensions, forward_abs_errors):
    """
    Plot the forward absolute error ||Q T Q.T - A||_F vs matrix dimension.
    Uses a logarithmic scale for the y-axis.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, forward_abs_errors, marker='o', linestyle='-', label='Forward Absolute Error')
    plt.title('Forward Absolute Error ||Q T Q.T - A|| vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('||Q T Q.T - A||_F')
    plt.yscale('log') 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('forward_abs_error.png') 
    plt.show(block=False)


def plot_forward_rel_error(dimensions, forward_rel_errors):
    """
    Plot the forward relative error ||Q T Q.T - A||_F / ||A||_F vs matrix dimension.
    Uses a logarithmic scale for the y-axis.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, forward_rel_errors, marker='o', linestyle='-', label='Forward Relative Error')
    plt.title('Forward Relative Error ||Q T Q.T - A|| / ||A|| vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('||Q T Q.T - A||_F / ||A||_F')
    plt.yscale('log') 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('forward_rel_error.png') 
    plt.show(block=False)


def plot_running_time(dimensions, running_times):
    """
    Plot the running time vs matrix dimension.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, running_times, marker='o', linestyle='-', label='Running Time')
    plt.title('Running Time vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Running Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('running_time.png') 
    plt.show(block=False)


def plot_iteration_time(dimensions, iteration_time):
    """
    Plot the number of iterations vs matrix dimension.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, iteration_time, marker='o', linestyle='-', label='Number of Iterations')
    plt.title('Number of Iterations vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Number of Iterations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iteration_time.png') 
    plt.show(block=False)


def plot_iterations_per_dimension(dimensions, iterations_per_dimension):
    """
    Plot the iterations per dimension vs matrix dimension.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, iterations_per_dimension, marker='o', linestyle='-', label='Iterations per Dimension')
    plt.title('Iterations Time per Dimension vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Iterations per Dimension ')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iterations_per_dimension.png') 
    plt.show(block=False)


def plot_iterations_per_time(dimensions, iterations_per_time):
    """
    Plot the iterations per running time vs matrix dimension.
    """
    plt.figure(figsize=(8,6))
    plt.plot(dimensions, iterations_per_time, marker='o', linestyle='-', label='Iterations per Second')
    plt.title('Iterations per Running Time vs Matrix Dimension')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Iterations per Second')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iterations_per_time.png') 
    plt.show(block=False)


########## Francis_QR_Double_Shift Part 03: Demonstration ##########
def one_test(dimension):
    """
    Test the Francis Double Shift QR algorithm on a randomly generated matrix.
    :param dimension: The dimension of the square matrix.
    """
    # Validation before wrapping the function: francis_double_shift_qr(A) 
    A = generate_test_matrix(dimension)
    print(f"dimension of Matrix A: {dimension}")
    # Q1 A Q1^T = H
    origin_H, Q1 = hessenberg(A)
    print(f"Norm of the residual ||A - Q1.T H Q1||: {np.linalg.norm(A - Q1.T @ origin_H @ Q1, ord='fro'):.5e}")
    print(f"Norm of the orthogonality ||Q1 Q1^T - I||: {np.linalg.norm(Q1 @ Q1.T - np.eye(dimension), ord='fro'):.5e}")

    # Q2 H Q2^T = T
    H = np.copy(origin_H)
    T, Q2, iteration_time = hessenberg_to_schur_form(H)
    T = zeros_hessenberg(T)
    print(f"Norm of the residual ||H - Q2.T T Q2||: {np.linalg.norm(origin_H - Q2.T @ T @ Q2, ord='fro'):.5e}")
    print(f"Norm of the orthogonality ||Q2 Q2.T - I||: {np.linalg.norm(Q2 @ Q2.T - np.eye(dimension), ord='fro'):.5e}")

    # A = Q1.T H Q1 = Q1.T Q2.T T Q2 Q1
    Q = Q1.T @ Q2.T
    final_A = Q @ T @ Q.T
    print(f"Norm of the residual ||A - Q T Q.T||: {np.linalg.norm(A - final_A, ord='fro'):.5e}")
    print(f"Norm of the orthogonality ||Q Q.T - I||: {np.linalg.norm(Q @ Q.T - np.eye(dimension), ord='fro'):.5e}")

    # Examine the eigenvalues of the real Schur form T and the origin matrix A
    eigenvalues = np.linalg.eigvals(A)
    # eigenvalue_diff = np.sort(np.linalg.eigvals(T)) - np.sort(np.linalg.eigvals(A))
    # print(f"Difference between eigenvalues of T and A: {eigenvalue_diff}")

    # Output to a .txt for convenient observation
    with open("output.txt", "w") as f:
        f.write("Eigenvalues of A:\n")
        for val in eigenvalues:
            f.write(f"{val:.3e}\n")
        f.write("\nMatrix T:\n")
        for row in T:
            f.write(" ".join([f"{elem:.3e}" for elem in row]) + "\n")
# end of one_test



def various_test():

    dimensions = range(10, 501, 1)
    information = collect_information(dimensions, seed=1)
    
    plot_orthogonality_loss(information['dimension'], information['orthogonality_loss'])
    plot_forward_abs_error(information['dimension'], information['forward_abs_error'])
    plot_forward_rel_error(information['dimension'], information['forward_rel_error'])
    plot_running_time(information['dimension'], information['running_time'])
    plot_iteration_time(information['dimension'], information['iteration_time'])
    plot_iterations_per_dimension(information['dimension'], information['iterations_per_dimension'])
    plot_iterations_per_time(information['dimension'], information['iterations_per_time'])
    plt.show()

if __name__ == "__main__":
   
   dimension = 20

   one_test(dimension)  

   # various_test()