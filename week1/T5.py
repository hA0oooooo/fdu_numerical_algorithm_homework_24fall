import numpy as np
import time
import matplotlib.pyplot as plt

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

def generate_test_matrix(n):
    A = np.random.randn(n, n)  
    b = np.random.randn(n)     
    return A, b

def measure_execution_time(n):
    A, b = generate_test_matrix(n)
    start_time = time.time()  
    gaussian_elimination(A, b) 
    end_time = time.time()     
    return end_time - start_time  

if __name__ == "__main__":
    dimensions = np.arange(10, 1001, 10)
    times = []

    for n in dimensions:
        exec_time = measure_execution_time(n)
        times.append(exec_time)

    log_dimensions = np.log10(dimensions)
    log_times = np.log10(times)

    plt.figure(figsize=(8, 6))
    plt.plot(log_dimensions, log_times, marker='o', linestyle='-')
    plt.xlabel('log(matrix dimension)', fontsize=12)
    plt.ylabel('log(execution time)', fontsize=12)
    plt.title('Log(matrix dimension)-log(execution time)', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.show()
