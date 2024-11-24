import numpy as np
import matplotlib.pyplot as plt

def generate_random_polygon(n):

    x = np.random.randn(n)
    y = np.random.randn(n)
    x -= np.mean(x)
    y -= np.mean(y)
    norm = np.sqrt(np.sum(x**2 + y**2))
    x /= norm
    y /= norm
    return np.array([x, y])

def iterate_ellipse(polygon, iterations):
    x, y = polygon
    flag = 0
    for _ in range(iterations):
        x_new = (x + np.roll(x, -1)) / 2
        y_new = (y + np.roll(y, -1)) / 2
        
        if flag % 100 == 0:
            center_x = np.mean(x_new)
            center_y = np.mean(y_new)
            x_new -= center_x
            y_new -= center_y
        
        norm = np.sqrt(np.sum(x_new**2 + y_new**2))
        x_new /= norm
        y_new /= norm
        x, y = x_new, y_new
        flag += 1

    return np.array([x, y])

# standard ellipse
def compute_c_s(n):

    tau = np.linspace(0, 2 * np.pi, n, endpoint=False) 
    c = np.sqrt(2 / n) * np.cos(tau)  
    s = np.sqrt(2 / n) * np.sin(tau)  

    return c, s


def calculate_theta(x, c, s):
    cos = np.dot(c, x)
    sin = np.dot(s, x)
    theta = np.arctan2(sin, cos)
    if theta < 0:
        theta += 2 * np.pi
    return theta


def calculate_a_b(theta_u, theta_v):

    a = (theta_u + theta_v) / 2
    b = (theta_v - theta_u) / 2

    return a, b


def calculate_matrix_A(a, b, n):

    U = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                  [np.sin(np.pi / 4),  np.cos(np.pi / 4)]])
    V = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]])
    Sigma = np.array([[np.sqrt(2) / np.sqrt(n) * np.cos(b), 0],
                      [0, np.sqrt(2) / np.sqrt(n) * np.sin(b)]]) 
    
    return U @ Sigma @ V.T


def generate_ellipse_points(A, num_points=1000):

    t = np.linspace(0, 2 * np.pi, num_points)
    parametric_points = np.array([np.cos(t), np.sin(t)])
    ellipse_points = A @ parametric_points

    return ellipse_points[0], ellipse_points[1]


n = 50
polygon = generate_random_polygon(n)

# iteration
iterations = 20000
new_polygon1 = iterate_ellipse(polygon, iterations)
new_polygon2 = iterate_ellipse(polygon, iterations+1)

# standard ellipse
c, s = compute_c_s(n)
theta_u = calculate_theta(polygon[0], c, s)
theta_v = calculate_theta(polygon[1], c, s)
a, b = calculate_a_b(theta_u, theta_v)
A = calculate_matrix_A(a, b, n)
u, v = generate_ellipse_points(A)

# plot
plt.figure(figsize=(12, 8))

plt.scatter(polygon[0], polygon[1], color='black', label="Random Polygon Points")
for i in range(n):
    plt.plot([polygon[0][i], polygon[0][(i + 1) % n]], [polygon[1][i], polygon[1][(i + 1) % n]], 'r-', alpha=0.5)

plt.scatter(new_polygon1[0], new_polygon1[1], color='blue', label="Ellipse points, after enough even iterations")
for i in range(n):
    plt.plot([new_polygon1[0][i], new_polygon1[0][(i + 1) % n]], [new_polygon1[1][i], new_polygon1[1][(i + 1) % n]], 'b-', alpha=0.5)

plt.scatter(new_polygon2[0], new_polygon2[1], color='green', label="Ellipse points, after enough odd iterations")
for i in range(n):
    plt.plot([new_polygon2[0][i], new_polygon2[0][(i + 1) % n]], [new_polygon2[1][i], new_polygon2[1][(i + 1) % n]], 'b-', alpha=0.5)

plt.plot(u, v, color='orange', label="Theoretical Ellipse, according to the paper")

plt.title(f"From Random Polygon to Ellipse after {iterations}(+0/+1) iterations")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.show()


