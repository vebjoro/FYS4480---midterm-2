
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.rcParams.update({'font.size': 14})

#!/usr/bin/env python3

#diagonaliser H-matrisa
import numpy as np
import matplotlib.pyplot as plt


def set_up_H(g):
    H = np.ones([6, 6])
    H -= np.identity(6)
    H = -0.5*g*H
    #diagonal
    H += np.diag([2., 4., 6., 6., 8., 10.])
    H -= np.identity(6)*g
    #zeros
    H[0, 5] = 0
    H[5, 0] = 0
    H[1, 4] = 0
    H[4, 1] = 0
    H[2, 3] = 0
    H[3, 2] = 0
    return H

H = set_up_H(-1)
E, C = np.linalg.eig(H)
print(E)
#print('')
#print(C)

#sjekkar normalisering
# for i in range(6):
#     print(np.sum(C[:, i]**2))
def perturbation_energy_2(g, order = 2):
    E0 = 2
    DE1 = -g
    DE2 = -g**2*(1/2 + 1/4 + 1/4 + 1/6)*1/4
    return E0 + DE1 + DE2

n = 100
gs = np.linspace(-1, 1, n)
E0_exact = np.zeros(n)
E0_2order = np.zeros(n)
E_pert_2 = np.zeros(n)

for i in range(n):
    H = set_up_H(gs[i])
    E, C = np.linalg.eig(H)
    E0_exact[i] = np.min(E)
    E, C = np.linalg.eig(H[:-1, :-1])
    E0_2order[i] = np.min(E)
    E_pert_2[i] = perturbation_energy_2(gs[i])





fig, ax = plt.subplots()
ax.plot(gs, E0_exact, color = 'k', label = 'FCI - Exact')
ax.plot(gs, E0_2order, ls = '--', color = 'r', label = 'FCI - Approx')
ax.plot(gs, E_pert_2, ls = 'dotted', color = 'g', label = 'MBPT - 2. order')
ax.set_xlabel('g')
ax.set_ylabel('E')
plt.legend()
plt.savefig('task6.pdf')
plt.show()
