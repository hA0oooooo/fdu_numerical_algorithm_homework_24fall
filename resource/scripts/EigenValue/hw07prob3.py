import numpy as np



a = 4
b = 8
c = 7
e = 1e-6
aa = 4
A = np.array([[a, c], [e, b]])
B = A.copy()
I = np.array([[1., 0], [0, 1.]])
print("initial A: ")
print(A)
orthos = []
i = 1
while abs(A[1, 0]) > 1e-15 or A[0, 1] < 0 or i < 7:
    Q, R = np.linalg.qr(A - aa * I, mode="complete")
    orthos.append(Q)
    A = R @ Q + aa * I
    print(f"iterration {i}, element at c's place: {A[0, 1]}")
    i += 1
print("final A:")
print(A)
# final = I.copy()
# for i in range(len(orthos)):
#     final = final @ orthos[i]
# print(final.T @ B @ final)








