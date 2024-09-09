def sum_inverse_x(n):
    sum = 0
    for i in range(1, n + 1):
        sum += 1 / i
    return sum

if __name__ == "__main__":
    i = 1
    while True:
        if sum_inverse_x(i) == sum_inverse_x(i + 1):
            break
        else:
            i += 1

    print('The harmonic series converges on', i)