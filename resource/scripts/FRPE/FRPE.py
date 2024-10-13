import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def display(xy, ax, n):
    ax.fill(xy[0], xy[1], facecolor="none", edgecolor="black")
    ax.scatter(xy[0], xy[1], color="r", s=5)
    if n == 0:
        ax.set(title="Initial Polygon")
    else:
        ax.set(title=f"After {n} Averagings")


def iter1(x, y, n, delta=None):
    if not delta:
        for k in range(n):
            x = (x + np.roll(x, -1)) / 2
            y = (y + np.roll(y, -1)) / 2
        return x, y
    center = (np.sum(x) / len(x), np.sum(y) / len(y))
    for k in range(n):
        x = (x + np.roll(x, -1)) / 2
        y = (y + np.roll(y, -1)) / 2
        srdiff = np.linalg.norm(x - center[0]) ** 2 + np.linalg.norm(y - center[1]) ** 2
        if srdiff <= delta ** 2:
            return k
    return x, y


def first_try():
    fig, axs = plt.subplots(1, 4, figsize=(9, 3),
                                    subplot_kw={'aspect': 'equal'})
    n = 20
    x = np.random.randn(n)
    y = np.random.randn(n)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    iters = (0, 10, 50, 200)
    for i in range(4):
        display(iter1(x, y, iters[i]), axs[i], iters[i])
    fig.suptitle(f"first try, {n} vertices")
    plt.savefig("first_try.png")
    plt.show()


def centroid_convergent_rate():
    sizes = np.array([10, 20, 40, 80])
    delta = 0.001
    data = pd.DataFrame(columns=["n", "Average $k_{\delta}$"])
    for _ in range(100):
        for i in range(4):
            n = sizes[i]
            x = np.random.randn(n)
            y = np.random.randn(n)
            x /= np.linalg.norm(x)
            y /= np.linalg.norm(y)
            data.loc[len(data.index)] = [n, iter1(x, y, 10000, delta=delta)]
    avg = data.groupby("n", as_index=False).mean()
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=avg.values, colLabels=avg.columns, loc='center')
    ax.set(title="100 trials, $\delta = 0.001$")
    fig.tight_layout()
    plt.savefig("rate_of_convergence.png")
    plt.show()


def iter2(x, y, n, recenter=False):
    if not recenter:
        for k in range(n):
            x = (x + np.roll(x, -1)) / 2
            y = (y + np.roll(y, -1)) / 2
            x /= np.linalg.norm(x)
            y /= np.linalg.norm(y)
    else:
        length = len(x)
        for k in range(n):
            x = (x + np.roll(x, -1)) / 2
            y = (y + np.roll(y, -1)) / 2
            x -= np.sum(x) / length
            y -= np.sum(y) / length
            x /= np.linalg.norm(x)
            y /= np.linalg.norm(y)
    return x, y


def second_try():
    fig, axs = plt.subplots(1, 4, figsize=(9, 3),
                            subplot_kw={'aspect': 'equal'})
    n = 20
    x = np.random.randn(n)
    y = np.random.randn(n)
    x -= np.sum(x) / n
    y -= np.sum(y) / n
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    iters = (0, 10, 50, 200)
    for i in range(4):
        display(iter2(x, y, iters[i]), axs[i], iters[i])
    fig.suptitle(f"second try, {n} vertices, with normalizations")
    plt.savefig("second_try.png")
    plt.show()


def third_exp():
    n = 12
    tau = np.array([x * 2 * np.pi / n for x in range(n)])
    c = np.cos(tau)
    s = np.sin(tau)
    tu = np.random.random() * 2 * np.pi
    tv = np.random.random() * 2 * np.pi
    x = c * np.cos(tu) + s * np.sin(tu)
    y = c * np.cos(tv) + s * np.sin(tv)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.fill(x, y, facecolor="none", edgecolor="grey", ls="-.")
    ax.scatter(x, y, color="r", s=25)
    x, y = iter2(x, y, 1)
    ax.fill(x, y, facecolor="none", edgecolor="grey", ls="-.")
    ax.scatter(x, y, color="b", s=25)

    fig.suptitle(f"third experiment, {n} vertices, even is red and odd is blue")
    plt.savefig("third_exp.png")
    plt.show()


if __name__ == "__main__":
    sns.set_theme()
    first_try()
    centroid_convergent_rate()
    second_try()
    third_exp()
