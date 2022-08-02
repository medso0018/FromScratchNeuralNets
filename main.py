import numpy as np


x = np.random.rand(3, 10)

w = np.random.rand(3, 3)

b = np.array([
    [2],
    [3],
    [4]
])


print(x @ w + b)


print(x)
