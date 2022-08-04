import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt


img = np.array([
    [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ],
    [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ],
    [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ],

])
print(img.shape)
imgplot = plt.imshow(img)
