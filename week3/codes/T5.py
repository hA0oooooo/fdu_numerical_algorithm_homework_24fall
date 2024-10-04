import numpy as np

def pivoting_gaussian_elimination(A_origin, b_origin):
    A = np.copy(A_origin)
    b = np.copy(b_origin)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if max_row != i:
            A[[i, max_row], :] = A[[max_row, i], :]
        max_col = np.argmax(np.abs(A[i, i:])) + i
        if max_col != i:
            A[:, [i, max_col]] = A[:, [max_col, i]]
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def partial_pivoting_gaussian_elimination(A_origin, b_origin):
    A = np.copy(A_origin)
    b = np.copy(b_origin)
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if max_row != i:
            A[[i, max_row], :] = A[[max_row, i], :]

        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x


def gaussian_elimination(A_origin, b_origin):
    A = np.copy(A_origin)
    b = np.copy(b_origin)
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


# matric one

n = 100

A1 = np.zeros((n, n), dtype=np.float32)
for i in range(n-1):
    A1[i, i] = 8
    A1[i, i+1] = 1
    A1[i+1, i] = 6

A1[n-1, n-1] = 8

b1 = np.zeros(n, dtype=np.float32)
for i in range(n):
    b1[i] = 15
b1[0] -= 6
b1[n-1] -= 1

without_sol1 = gaussian_elimination(A1, b1)

with_sol1 = pivoting_gaussian_elimination(A1, b1)

with_partial_sol1 = partial_pivoting_gaussian_elimination(A1, b1)


# matric two

A2 = np.zeros((n, n), dtype=np.float32)
for i in range(n-1):
    A2[i, i] = 6
    A2[i, i+1] = 1
    A2[i+1, i] = 8

A2[n-1, n-1] = 6

b2 = np.zeros(n, dtype=np.float32)
for i in range(n):
    b2[i] = 15
b2[0] -= 8
b2[n-1] -= 1

without_sol2 = gaussian_elimination(A2, b2)

with_sol2 = pivoting_gaussian_elimination(A2, b2)

with_partial_sol2 = partial_pivoting_gaussian_elimination(A2, b2)


# print

print(with_sol1)
print(with_partial_sol1)
print(without_sol1)

print(with_sol2)
print(with_partial_sol2)
print(without_sol2)