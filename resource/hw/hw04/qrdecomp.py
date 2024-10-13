import numpy as np
import math
from numba import jit
from matplotlib import pyplot as plt
from matplotlib import colors


def Cholesky(A):
    """
    Compute the Cholesky factorization of hermitian positive-definite matrix A.
    Return the upper triangular Cholesky factor of A. Will mutate A.
    """
    n = A.shape[0]
    for j in range(n):
        if A[j, j].real < 0:
            print(f"warning: {A[j, j]}")
        A[j, j] = math.sqrt(A[j, j].real)
        A[j, j + 1:] /= A[j, j]
        for k in range(j + 1, n):
            A[k, k:] -= A[j, k] * A[j, k:]
    return np.triu(A)


def CholeskyQR(A):
    """
    Use Cholesky factorization of A^*A to compute the QR factorization of A.
    :param A: general complex matrix
    :return Q, R: the QR factorization of A
    """
    R = Cholesky(A.conj().T @ A)
    Q = A @ np.linalg.inv(R)
    return Q, R


@jit(nopython=True)
def house_vec(x):
    sq_length = np.dot(x[1:].conj(), x[1:]).real
    if sq_length == 0:
        print("!!!")
        return x
    # x = x.copy()
    angle = x[0] / abs(x[0]) if abs(x[0]) != 0 else 1
    print(angle)
    length = math.sqrt((sq_length + abs(x[0]) ** 2))
    if angle.real > 0:
        x[0] = - angle * (sq_length / (length + abs(x[0])))
        x[1:] /= x[0]
        x[0] = length * angle  # part of R
    else:
        x[0] = angle * (abs(x[0]) + length)
        x[1:] /= x[0]
        x[0] = length * (-angle)  # part of R
    return x


@jit(nopython=True)
def house_left(A, h, inplace=False):
    if inplace:
        tmp = h.conj() @ A * (-2) / np.dot(h.conj(), h).real
        for i in range(h.shape[0]):
            for j in range(tmp.shape[0]):
                A[i, j] += h[i] * tmp[j]
    else:
        A -= 2 * np.outer(h, h.conj() @ A) / np.dot(h.conj(), h).real
    return A


def house_tri(A):
    """
    Use Householder reflection to unitary triangular complex matrix A. Will mutate A.
    :param A: general complex matrix
    :return A: R is stored in the upper triangular part of A,
    and the essential part of Householder vectors in the lower part of A.
    """
    for i in range(A.shape[1]):
        house_vec(A[i:, i])
        house_left(A[i:, i + 1:], np.concatenate([[1], A[i + 1:, i]]), False)
    return A


def house_Q(A):
    """
    Use backward accumulation to explicitly construct Q
    :param A: a complex matrix AFTER house_tri()
    :return Q: the unitary matrix.
    """
    n = A.shape[1]
    Q = np.eye(A.shape[0], dtype=A.dtype)
    for j in range(n - 1, -1, -1):
        h = np.concatenate([[1], A[j + 1:, j]])
        b = 2 / (1 + np.dot(A[j + 1:, j].conj(), A[j + 1:, j]).real)
        Q[j:, j:] -= b * np.outer(h, h.conj() @ Q[j:, j:])
    return Q

ortholossnorm = [[] for _ in range(2)]
ortholoss = [[] for _ in range(2)]
kappas = [10 ** i for i in range(10)]


def rand_matrix_by_kappa(m, n, kappa):
    singulars = np.random.random(n) * (1 - 1. / kappa) + 1. / kappa
    singulars.sort()
    singulars[0] = 1. / kappa
    singulars[-1] = 1.
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = singulars[i]
    return np.linalg.qr(np.random.random((m, n)) - 0.5)[0] @ D @ np.linalg.qr(np.random.random((n, n)) - 0.5)[0]


np.random.seed(0)
m, n = 200, 50
for kappa in kappas:
    A = rand_matrix_by_kappa(m, n, kappa)
    Q, _ = CholeskyQR(A.copy())
    ortholoss[0].append(Q @ Q.conj().T - np.eye(m))
    ortholossnorm[0].append(np.linalg.norm(ortholoss[0][-1]))
    Q = house_Q(house_tri(A.copy()))
    ortholoss[1].append(Q.conj().T @ Q - np.eye(m))
    ortholossnorm[1].append(np.linalg.norm(ortholoss[1][-1]))

scatters = []
for ortho in ortholossnorm:
    scatters.append(plt.scatter(range(len(kappas)), np.log10(ortho), marker='.'))
    plt.plot(range(len(kappas)), np.log10(ortho))
plt.legend(scatters, ['Cholesky', 'Householder'])
plt.xlabel('log10 of Condition Number $\kappa$')
plt.ylabel('log10 of Orthogonality Loss $\Vert Q^*Q - I\Vert_F$')
plt.savefig("hw04pic1")
plt.show()

datas = [np.abs(ortholoss[0][0]), np.abs(ortholoss[1][0])]
Nc = 2
fig, axs = plt.subplots(1, Nc)
fig.suptitle('Loss of Orthogonality ($\kappa = 1$)')
images = []
for j in range(Nc):
    # Generate data with a range that varies from one plot to the next.
    images.append(axs[j].imshow(datas[j]))
    axs[j].label_outer()
axs[0].set_title("Cholesky")
axs[1].set_title("Householder")

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


for im in images:
    im.callbacks.connect('changed', update)
fig.savefig("hw04pic2")
plt.show()
