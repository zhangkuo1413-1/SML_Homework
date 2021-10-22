import numpy as np
from matplotlib import pyplot as plt

m = np.zeros([200, 2])
m[0, 0] = 1
T = np.array([[0.42, 0.026],[0.58, 0.974]])
for i in range(199): m[i+1] = T.dot(m[i])
plt.plot(np.arange(200), m[:, 0], color='b')
plt.scatter(18, m[18, 0], color='r')
plt.text(18, m[18, 0], (18, round(m[18, 0], 4)), color='r', ha='right', va='bottom', fontsize=10)
plt.xlabel("Generation")
plt.ylabel("Probability")
plt.title("Will the mutation still be present?")
plt.xticks(np.arange(0, 199, 1))
plt.axis([0, 199, 0, 1.1])
plt.grid()
plt.savefig('3c1.png')