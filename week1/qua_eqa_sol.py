from math import sqrt

def determinant(a, b, c):
    delta = b**2 - 4*a*c
    return delta

def is_root_real(a, b, c):
    delta = determinant(a, b, c)
    if delta >= 0:
        return True
    else:
        return False

def quad_fun_solve(a, b, c):
    if is_root_real(a, b, c):
        delta = determinant(a, b, c)
        # 判断是否会发生消元误差
        if abs(delta - abs(b)) < 1e-2:
            if b > 0:
                x1 = (-b - sqrt(delta)) / (2*a)
                x2 = 2*c / (-b - sqrt(delta))
            else:
                x1 = (-b + sqrt(delta)) / (2*a)
                x2 = 2*c / (-b + sqrt(delta))
        else:
            x1 = (-b + sqrt(delta)) / (2*a)
            x2 = (-b - sqrt(delta)) / (2*a)
        return x1, x2
    else:
        return None

if __name__ == "__main__":
    a, b, c = map(float, input("请输入方程系数 a b c，例如 1 2 3 ").split())

    result = quad_fun_solve(a, b, c)
    if result:
        x1, x2 = result
        print(f"x1 = {x1}, x2 = {x2}")
    else:
        print("无实数根。")
