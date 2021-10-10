import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.sin(x) / (2 + x)


def dif1func(x):
    return ((x + 2) * np.cos(x) - np.sin(x)) / ((x + 2)**2)


def dif2func(x):
    return (-np.sin(x) / (x + 2)) + (2 * np.sin(x) / ((x + 2)**3)) - (2 * np.cos(x) / ((x + 2)**2))


def centraldif(prev, next, h):
    return (next - prev) / (2 * h)


def backwarddif(curr, prev, h):
    return (curr - prev) / h


def forwarddif(curr, next, h):
    return (next - curr) / h


def centraldif2(prev, curr, next, h):
    return (prev - 2 * curr + next) / (h**2)


def main1():
    N = 15
    a = -1.5
    b = 1.5
    h = (b - a) / (N - 1)
    xrange = np.arange(a, b + h, h)
    forwarddif1values = np.zeros(len(xrange))
    centraldif2values = np.zeros(len(xrange))
    centraldif1values = np.zeros(len(xrange))
    for i in range(1, N - 1):
        forwarddif1values[i] = forwarddif(func(xrange[i]), func(xrange[i + 1]), h)
        centraldif1values[i] = centraldif(func(xrange[i - 1]), func(xrange[i + 1]), h)
        centraldif2values[i] = centraldif2(func(xrange[i - 1]), func(xrange[i]), func(xrange[i + 1]), h)
    forwarddif1values[0] = forwarddif(func(xrange[0]), func(xrange[1]), h)
    #forwarddif1values[N - 1] = forwarddif1values[N - 2]
    #centraldif1values[N - 1] = centraldif1values[N - 2]
    #centraldif1values[0] = centraldif1values[1]

    truedif1values = dif1func(xrange)
    truedif2values = dif2func(xrange)

    plt.subplot(2, 3, 1)
    plt.title("Первая производная, правые разности")
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.grid()
    plt.plot(xrange[0:N - 1], forwarddif1values[0:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif1values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.xlabel("x")
    plt.ylabel("|Δy'|")
    plt.grid()
    plt.plot(xrange[0:N - 1], abs(truedif1values[0:N - 1] - forwarddif1values[0:N - 1]), color='k', label='Абсолютная погрешность')

    plt.subplot(2, 3, 2)
    plt.title("Первая производная, центральные разности")
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.grid()
    plt.plot(xrange[1:N - 1], centraldif1values[1:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif1values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.xlabel("x")
    plt.ylabel("|Δy'|")
    plt.grid()
    plt.plot(xrange[1:N - 1], abs(truedif1values[1:N - 1] - centraldif1values[1:N - 1]), color='k', label='Абсолютная погрешность')

    plt.subplot(2, 3, 3)
    plt.title("Вторая производная, центральные разности")
    plt.xlabel("x")
    plt.ylabel("y''")
    plt.grid()
    plt.plot(xrange[1:N - 1], centraldif2values[1:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif2values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.xlabel("x")
    plt.ylabel("|Δy''|")
    plt.grid()
    plt.plot(xrange[1:N - 1], abs(truedif2values[1:N - 1] - centraldif2values[1:N - 1]), color='k',
             label='Абсолютная погрешность')

    plt.show()

def main2():
    hmin = 0.001
    hstep = 0.001
    hmax = 0.1
    hrange = np.arange(hmin, hmax, hstep)
    a = -1.5
    b = 1.5

    error = np.zeros([len(hrange), 3])

    for j in range(len(hrange)):
        h = hrange[j]
        xrange = np.arange(a, b + h, h)
        forwarddif1values = np.zeros(len(xrange))
        centraldif2values = np.zeros(len(xrange))
        centraldif1values = np.zeros(len(xrange))
        for i in range(1, len(xrange) - 1):
            forwarddif1values[i] = forwarddif(func(xrange[i]), func(xrange[i + 1]), h)
            centraldif1values[i] = centraldif(func(xrange[i - 1]), func(xrange[i + 1]), h)
            centraldif2values[i] = centraldif2(func(xrange[i - 1]), func(xrange[i]), func(xrange[i + 1]), h)
        forwarddif1values[0] = forwarddif(func(xrange[0]), func(xrange[1]), h)
        # forwarddif1values[len(xrange) - 1] = forwarddif1values[len(xrange) - 2]
        # centraldif1values[len(xrange) - 1] = centraldif1values[len(xrange) - 2]
        # centraldif1values[0] = centraldif1values[1]

        truedif1values = dif1func(xrange)
        truedif2values = dif2func(xrange)
        error[j, 0] = max(abs(truedif1values - forwarddif1values)[:-1])
        error[j, 1] = max(abs(truedif1values - centraldif1values)[1:-1])
        error[j, 2] = max(abs(truedif2values - centraldif2values)[1:-1])

    error = np.log(error)
    hrange = np.log(hrange)

    plt.subplot(1, 3, 1)
    plt.title("Первая производная, правые разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy'|))")
    plt.grid()
    plt.plot(hrange, error[:, 0], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Первая производная, центральные разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy'|))")
    plt.grid()
    plt.plot(hrange, error[:, 1], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Вторая производная, центральные разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy''|))")
    plt.grid()
    plt.plot(hrange, error[:, 2], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.show()


main2()
