#! python3
"""
@author: b04902053
"""

import numpy as np

data = np.load('./data.npz')

eigh = np.zeros((200, 60))
for i in range(200):
    x = data[str(i)]
    val, vec = np.linalg.eigh(np.cov(x.T))
    eigh[i] = val[:60]

np.savetxt('./eigh.csv', eigh)
