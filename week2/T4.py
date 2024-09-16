import numpy as np
import time
import matplotlib.pyplot as plt

def partial_pivoting_gaussion_elimination:



def complete_pivoting_gaussion_elimination:



def generate_test_matrix(n):
    A = np.random.randn(n,n)
    return A


def measure_execution_time(n, strategy):
    A = generate_test_matrix
    start = time.time()
    strategy(A)
	end = time.time()
    return end_time - start_time
    
def plot_log_log(strategy):
    dimention = np.arange(10, 1001, 10)


