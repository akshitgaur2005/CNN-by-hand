from backend import CNN
import numpy as np

data = np.arange(1 * 4 * 4).reshape(1, 4, 4)
print(data)

layer = CNN(2, 1, 2)
print(layer.weights)
out = layer.forward(data)
print(out)
