import numpy as np

y_true = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
y_pred = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

# print(np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)))

print(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
