import numpy as np

m = np.zeros([2, 2])
m[1, 0] = 1
T = np.array([[0.42, 0.026],[0.58, 0.974]])
i = 0
while abs(m[0, 0] - m[1, 0]) > 0.0001:
    m[0] = m[1]
    m[1] = T.dot(m[0])
    i += 1
print("Timestep: {}, Probability: {}".format(i, m[1, 0]))
