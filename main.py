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


def centraldif1order4(prev2, prev1, next1, next2, h):
    return (prev2 - 8 * prev1 + 8 * next1 - next2) / (12*h)


def centraldif2order4(prev2, prev1, curr, next1, next2, h):
    return (-prev2 + 16 * prev1 - 30 * curr + 16 * next1 - next2) / (12 * (h**2))


def main1():
    N = 20
    a = -1.5
    b = 1.5
    h = (b - a) / (N - 1)
    xrange = np.arange(a, b + h, h)
    forwarddif1values = np.zeros(len(xrange))
    centraldif2values = np.zeros(len(xrange))
    centraldif1values = np.zeros(len(xrange))
    centraldif2values4order = np.zeros(len(xrange))
    for i in range(1, N - 1):
        forwarddif1values[i] = forwarddif(func(xrange[i]), func(xrange[i + 1]), h)
        centraldif1values[i] = centraldif(func(xrange[i - 1]), func(xrange[i + 1]), h)
        centraldif2values[i] = centraldif2(func(xrange[i - 1]), func(xrange[i]), func(xrange[i + 1]), h)
    forwarddif1values[0] = forwarddif(func(xrange[0]), func(xrange[1]), h)

    for i in range(2, N - 2):
        centraldif2values4order[i] = centraldif2order4(func(xrange[i - 2]), func(xrange[i - 1]), func(xrange[i]), func(xrange[i + 1]), func(xrange[i + 2]), h)

    truedif1values = dif1func(xrange)
    truedif2values = dif2func(xrange)

    plt.subplot(2, 4, 1)
    plt.title("Первая производная,\n правые разности")
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.grid()
    plt.plot(xrange[0:N - 1], forwarddif1values[0:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif1values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.xlabel("x")
    plt.ylabel("|Δy'|")
    plt.grid()
    plt.plot(xrange[0:N - 1], abs(truedif1values[0:N - 1] - forwarddif1values[0:N - 1]), color='k', label='Абсолютная погрешность')

    plt.subplot(2, 4, 2)
    plt.title("Первая производная,\n центральные разности")
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.grid()
    plt.plot(xrange[1:N - 1], centraldif1values[1:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif1values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.xlabel("x")
    plt.ylabel("|Δy'|")
    plt.grid()
    plt.plot(xrange[1:N - 1], abs(truedif1values[1:N - 1] - centraldif1values[1:N - 1]), color='k', label='Абсолютная погрешность')

    plt.subplot(2, 4, 3)
    plt.title("Вторая производная,\n центральные разности")
    plt.xlabel("x")
    plt.ylabel("y''")
    plt.grid()
    plt.plot(xrange[1:N - 1], centraldif2values[1:N - 1], color='k', label='Численное значение')
    plt.plot(xrange, truedif2values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.xlabel("x")
    plt.ylabel("|Δy''|")
    plt.grid()
    plt.plot(xrange[1:N - 1], abs(truedif2values[1:N - 1] - centraldif2values[1:N - 1]), color='k',
             label='Абсолютная погрешность')

    plt.subplot(2, 4, 4)
    plt.title("Вторая производная,\n центральные разности, 4 порядка")
    plt.xlabel("x")
    plt.ylabel("y''")
    plt.grid()
    plt.plot(xrange[2:N - 2], centraldif2values4order[2:N - 2], color='k', label='Численное значение')
    plt.plot(xrange, truedif2values, ls='--', color='k', label='Аналитическое значение')
    plt.legend()

    plt.subplot(2, 4, 8)
    plt.xlabel("x")
    plt.ylabel("|Δy''|")
    plt.grid()
    plt.plot(xrange[2:N - 2], abs(truedif2values[2:N - 2] - centraldif2values4order[2:N - 2]), color='k',
             label='Абсолютная погрешность')

    plt.show()

def main2():
    hmin = 0.001
    hstep = 0.001
    hmax = 0.1
    hrange = np.arange(hmin, hmax, hstep)
    a = -1.5
    b = 1.5

    error = np.zeros([len(hrange), 4])

    for j in range(len(hrange)):
        h = hrange[j]
        xrange = np.arange(a, b + h, h)
        forwarddif1values = np.zeros(len(xrange))
        centraldif2values = np.zeros(len(xrange))
        centraldif1values = np.zeros(len(xrange))
        centraldif2values4order = np.zeros(len(xrange))
        for i in range(1, len(xrange) - 1):
            forwarddif1values[i] = forwarddif(func(xrange[i]), func(xrange[i + 1]), h)
            centraldif1values[i] = centraldif(func(xrange[i - 1]), func(xrange[i + 1]), h)
            centraldif2values[i] = centraldif2(func(xrange[i - 1]), func(xrange[i]), func(xrange[i + 1]), h)
        forwarddif1values[0] = forwarddif(func(xrange[0]), func(xrange[1]), h)
        # forwarddif1values[len(xrange) - 1] = forwarddif1values[len(xrange) - 2]
        # centraldif1values[len(xrange) - 1] = centraldif1values[len(xrange) - 2]
        # centraldif1values[0] = centraldif1values[1]
        for i in range(2, len(xrange) - 2):
            centraldif2values4order[i] = centraldif2order4(func(xrange[i - 2]), func(xrange[i - 1]), func(xrange[i]),
                                                           func(xrange[i + 1]), func(xrange[i + 2]), h)

        truedif1values = dif1func(xrange)
        truedif2values = dif2func(xrange)
        error[j, 0] = max(abs(truedif1values - forwarddif1values)[:-1])
        error[j, 1] = max(abs(truedif1values - centraldif1values)[1:-1])
        error[j, 2] = max(abs(truedif2values - centraldif2values)[1:-1])
        error[j, 3] = max(abs(truedif2values - centraldif2values4order)[2:-2])

    error = np.log(error)
    hrange = np.log(hrange)

    plt.subplot(1, 4, 1)
    plt.title("Первая производная,\n правые разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy'|))")
    plt.grid()
    plt.plot(hrange, error[:, 0], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.title("Первая производная,\n центральные разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy'|))")
    plt.grid()
    plt.plot(hrange, error[:, 1], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.title("Вторая производная,\n центральные разности")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy''|))")
    plt.grid()
    plt.plot(hrange, error[:, 2], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.title("Вторая производная,\n центральные разности, 4 порядка")
    plt.xlabel("log(h)")
    plt.ylabel("log(max(|Δy''|))")
    plt.grid()
    plt.plot(hrange, error[:, 3], color='k', label='Абсолютная погрешность')
    plt.legend()

    plt.show()


main1()
main2()
