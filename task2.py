import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import sympy as symp


def H(g):
    #g variable is self explanatory


    matmat = np.zeros((len(g),6))
    minivec = np.zeros((len(g),5))
    for j in range(len(g)):
        Vones = np.ones((6,6))
        Vline = np.identity(6)
        for i in range(6):
            Vones[i,5-i]  = 0
        V = -0.5*g[j]*(Vline+Vones)

        hdiag = np.array([2,4,6,6,8,10])
        H0 = np.diag(hdiag)

        H1 = H0+V
        H1approx = H1[0:5,0:5]
        eigvalapprox =np.linalg.eig(H1approx)
        eigval = np.linalg.eig(H1)
        minivec[j,:]= np.sort(eigvalapprox[0])
        matmat[j,:] = np.sort(eigval[0])

    
    for elem1,elem2 in zip(minivec.T,matmat.T):
        plt.plot(g,elem1,"-+",label = "approx")
        plt.plot(g,elem2,"--", label = "exact")
    plt.ylabel("$\epsilon_0$")
    plt.xlabel("$g$")
    #plt.legend()
    plt.savefig("task23.pdf")
    plt.show()
    return 0

gvec = np.linspace(-1,1,200)

H(gvec)
"""
gg = symp.Symbol("g")

Vones = symp.ones(6,6)
Vline = symp.eye(6)
for i in range(6):
    Vones[i,5-i]  = 0
V = -0.5*gg*(Vline+Vones)

H0 = symp.diag(2,4,6,6,8,10)

H1 = H0+V

#print(d.subs[0]({g:1}))
#eigval = list(H1.eigenvals().keys())
print(H1.values())
print(eigval[1].subs({"g":1}))


"""