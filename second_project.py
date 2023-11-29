import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def H(xi = 1, g = -1, dims=6):
    H = np.zeros((dims,dims))
    H[0,0] = 2 * xi - g
    H[1,1] = 4 * xi - g
    H[2,2] = 6 * xi - g
    H[3,3] = 6 * xi - g
    H[4,4] = 8 * xi - g
    if dims == 6:
        H[5,5] = 10 - g
    off_diags = - g / 2
    H[0,1] = off_diags
    H[0,2] = off_diags
    H[0,3] = off_diags
    H[1,2] = off_diags
    H[1,3] = off_diags
    H[2,4] = off_diags
    H[3,4] = off_diags
    if dims == 6:
        H[1,5] = off_diags
        H[2,5] = off_diags
        H[3,5] = off_diags
        H[4,5] = off_diags

    H = H + H.T - np.diag(np.diag(H)) # Make the matrix symmetric
    
    
    return H



def task_5(xi=1, g=-1, dims=4, C=np.eye(4)):
    def h_hartreefock(C):
        """
        Calculates the matrix h^HF
        """
        h_HF = np.zeros((dims,dims))
        alfa = np.arange(1, dims, 1)
        h_HF[0,0] = 2 - g * 0.5
        for a in alfa:
            h_HF[a,a] = ( (2 - g) + sum((p) * delta_funky(a,p) for p in [0,1])
                            + sum((p) * delta_funky(a,p) for p in [2,3]) 
                                + 0.5 * g * sum(1 * delta_funky(a,p) for p in [0,1]) )
        return h_HF 
       
    # e_HF = np.zeros(dims)
    # e_HF_prev = np.ones(dims)
    # tol = 1e-4
    # i = 1
    # while any(abs(e_HF - e_HF_prev) > tol) and i < 1000:
    #     if i % 25 == 0:
    #         print(f"Running iteration number: {i}")
    #     h_HF = h_hartreefock(C)
    #     soleig, solvec = np.linalg.eigh(h_HF)
    #     for j, vec in enumerate(solvec):
    #         C[:,j] = vec

    #     e_HF_prev = e_HF
    #     e_HF  = soleig

    #     i += 1
    h_HF = h_hartreefock(C)
    e_HF, e_HF_vecs = np.linalg.eigh(h_HF)

    return e_HF[0], h_HF
    
        
     

def energy_prefactor(H0, i, a):
    eps_i = H0[i,i] 
    eps_a = H0[a,a]
    
    return 1 /  ( eps_i - eps_a)

def V_element(V, bra=0, ket=1):
    element = V[bra, ket]

    return element
 
def diagrammatic_factors(nh, nl, nep):
    from_nep = (1 / 2) ** nep
    from_nl = (-1) ** nl
    from_nh = (-1) ** nh
    
    return from_nh * from_nep * from_nl

def delta_funky(a, b):
    if a == b:
        return 1
    return 0

if __name__ == "__main__":
    g = np.linspace(-1,1,50)

    exercise_2 = True
    energy_FCI = []
    if exercise_2:
        eigvals = {}
        eigvecs = {}
        # print("Following GS - energies are for Hilbert space of dimension: 6")
        for g_val in g:
            eigvals[g_val], eigvecs[g_val] = np.linalg.eigh(H(xi=1, g=g_val, dims=6)) 
            # print(f"E_gs for g = {g_val} : {eigvals[g_val][0]}")
            energy_FCI.append(eigvals[g_val][0])
        # print("------------------------------------\n")
        
    
    exercise_3 = True
    energy_FCI_5dims = []
    if exercise_3:
        eigvals = {}
        eigvecs = {}
        print("Following GS - energies are for Hilbert space of dimension: 5")
        for g_val in g:
            eigvals[g_val], eigvecs[g_val] = np.linalg.eigh(H(xi=1, g=g_val, dims=5)) 
            print(f"E_gs for g = {g_val} : {eigvals[g_val][0]}")
            energy_FCI_5dims.append(eigvals[g_val][0])
        print("------------------------------------\n")

    exercise_4 = True
    if exercise_4:
        h_HF = {}
        e_HF_GS = {}
        # print("Following GS - energies for Hartree-Fock hamiltonian")
        for g_val in g:
            e_HF_GS[g_val], h_HF[g_val] = task_5(g=g_val)
            # print(f"E_gs for g = {g_val} : {e_HF_GS[g_val]}")
        
        
        lists = sorted(e_HF_GS.items())
        # plt.plot(*zip(*lists))
        # plt.show()
    
    sns.set_theme()
    # Task 5; FCI vs HF plot
    # plt.plot(*zip(*lists), "r--", g, energy_FCI, "k-")
    # plt.legend(["HF", "FCI (Exact)"])
    # plt.xlabel("g")
    # plt.ylabel(r"$E_{GS}$")
    # plt.title("Ground state energy estimate comparison: FCI vs HF")
    # plt.savefig("FCIvsHF.pdf")
    # plt.show()
    
    
    sums_3rdorder = []
    sums_4thorder = []
    for g_val in g:         
        # V_old, H0_old = H_sep(g=g_val)
        e, h_HF = task_5(g_val)
        V = - np.ones((4,4)) * 0.5 * g_val
        # np.fill_diagonal(V, - g_val)
        H0 = np.diag(np.array([0,2,4,6])) 
        # H0 = h_HF 
        diagram_0 = - g_val

        print(f"Calculating MBPT energy diagrams for g = {g_val}")
        diagram_1_3rd = diagrammatic_factors(nh=2,nl=2,nep=2) * (
            sum(sum(
                V_element(V=V, bra=i, ket=a) * V_element(V=V, bra=a, ket=i)
                    * energy_prefactor(H0=H0, i=i, a=a) 
                for a in [2,3]
            ) 
            for i in [0,1] )
        )
        
        # diagram_1_other = 0
        # for i in [0,1]:
        #     for a in [2,3]:
        #         diagram_1_other += V_element(V=V, bra=i, ket=a) * V_element(V=V, bra=a, ket=i) * energy_prefactor(H0=H0, i=i, a=a)
        
        
        # In diagram 2, ShavittBartlett claims nh=3 for some reason - needs to be questioned.
        diagram_4_3rd = diagrammatic_factors(nh=3, nep=3, nl=2) * (
            sum(sum(sum(
                V_element(V=V, bra=i, ket=a) * V_element(V=V, ket=b, bra=a) * V_element(V=V, ket=i, bra=b)
                    * energy_prefactor(H0=H0, i=i, a=a) * energy_prefactor(H0=H0, i=i, a=b) 
                    for a in [2,3]
            ) for b in [2,3]
            ) for i in [0,1] )
        )


        diagram_5_3rd = diagrammatic_factors(nh=4, nep=3, nl=2) * (
            sum(sum(sum(
                V_element(V=V, bra=i, ket=a) * V_element(V=V, bra=j, ket=i) * V_element(V=V, bra=a, ket=j)
                    * energy_prefactor(H0=H0, i=i, a=a) ** 2 for a in [2,3]
            ) for j in [0,1] ) for i in [0,1] )
        )

        diagram_8_3rd = diagrammatic_factors(nh=4, nep=1, nl=3) * (
            sum(sum(
                V_element(V=V, bra=i, ket=a) * V_element(V=V, bra=i, ket=i) * V_element(V=V, bra=a, ket=i)
                    * energy_prefactor(H0=H0, i=i, a=a) ** 2 for a in [2,3]
            ) for i in [0,1] )
        )
        diagram_sum = diagram_0 + diagram_1_3rd + diagram_4_3rd + diagram_5_3rd + diagram_8_3rd
        # print(f"Energy found for diagram 1: {diagram_1_3rd}")
        # print(f"Energy found for diagram 4: {diagram_4_3rd}")
        # print(f"Energy found for diagram 5: {diagram_5_3rd}")
        # print(f"Energy found for diagram 8: {diagram_8_3rd}")
        # print("\n")
        # print(rf"Sum of all perturbative energies (diagrams), $$\delta E$$: {diagram_sum}")
        # print(f"Energy estimate for E_GS in RS-MBPT: {H0[0,0] + diagram_sum}")
        # print("-------------------------" + "\n")
        sums_3rdorder.append(diagram_sum + H0[0,0] + H0[1,1])

        # The following code is for part 7 of the midterm. 
        # 2P2H to fourth order
        diagram_5 = diagrammatic_factors(nh=4, nl=2, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, a, b) * V_element(V, j, i) * V_element(V, b, j)
                * energy_prefactor(H0,i=i, a=a) * energy_prefactor(H0, i=j,a=b) * energy_prefactor(H0, i=i, a=b)

            for i in [0,1])
             for j in [0,1])
              for a in [2,3]) 
                for b in [2,3])
        )
        diagram_6 = diagrammatic_factors(nh=4, nl=2, nep=4) * (
            sum(sum(sum(sum(
                V_element(V,i,a) * V_element(V, j, i) * V_element(V, a, b) * V_element(V, b, j)
                 * energy_prefactor(H0, i=i, a=a) * energy_prefactor(H0, i=j, a=a) * energy_prefactor(H0, i=j, a=b)            
            for i in [0,1])
            for j in [0,1])
            for a in [2,3])
            for b in [2,3])
        )
        diagram_14 = diagrammatic_factors(nh=2, nl=2, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, a, b) * V_element(V, b, c) * V_element(V, c, i)
                * energy_prefactor(H0, i=i, a=a) * energy_prefactor(H0, i=i, a=b) * energy_prefactor(H0, i=i, a=c)
            for i in [0,1])
            for a in [2,3])
            for b in [2,3])
            for c in [2,3])
        )
        diagram_15 = diagrammatic_factors(nh=6, nl=2, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, j, i) * V_element(V, k, j) * V_element(V, a, k)
                * energy_prefactor(H0, i=i, a=a) * energy_prefactor(H0, i=j, a=a) * energy_prefactor(H0, i=k, a=a)
            for i in [0,1])
            for j in [0,1])
            for k in [0,1])
            for a in [2,3])
        )
        sum_2P2H_4th = diagram_5 + diagram_15 + diagram_14 + diagram_6
        
        # 4P4H to fourth order
        diagram_33 = diagrammatic_factors(nh=4, nl=4, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, j, b) * V_element(V, a, i) * V_element(V, b, j)
                * energy_prefactor(H0, i=i, a=a) * (energy_prefactor(H0, i=i, a=a) + energy_prefactor(H0, i=j, a=b)) * energy_prefactor(H0, i=j, a=b)
            for i in [0,1])
            for j in [0,1])
            for a in [2,3])
            for b in [2,3])
        )


        diagram_36 = diagrammatic_factors(nh=4, nl=0, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, j, b) * V_element(V, a, j) * V_element(V, b, i)
                * energy_prefactor(H0, i=i, a=a) * (energy_prefactor(H0, i=i, a=a) + energy_prefactor(H0, i=j, a=b)) * energy_prefactor(H0, i=i, a=b)
            for i in [0,1])
            for j in [0,1])
            for a in [2,3])
            for b in [2,3])
        )

        diagram_37 = diagrammatic_factors(nh=4, nl=0, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, j, b) * V_element(V, b, i) * V_element(V, a, j)
                * energy_prefactor(H0, i=i, a=a) * (energy_prefactor(H0, i=i, a=a) + energy_prefactor(H0, i=j, a=b)) * energy_prefactor(H0, i=j, a=a)
            for i in [0,1])
            for j in [0,1])
            for a in [2,3])
            for b in [2,3])
        )

        diagram_41 = diagrammatic_factors(nh=4, nl=4, nep=4) * (
            sum(sum(sum(sum(
                V_element(V, i, a) * V_element(V, j, b) * V_element(V, b, j) * V_element(V, a, i)
                * energy_prefactor(H0, i=i, a=a) * (energy_prefactor(H0, i=i, a=a) + energy_prefactor(H0, i=j, a=b)) * energy_prefactor(H0, i=i, a=a)
            for i in [0,1])
            for j in [0,1])
            for a in [2,3])
            for b in [2,3])
        )

        sum_4P4H_4th = diagram_36 + diagram_33 + diagram_37 + diagram_41

        sums_4thorder.append(sum_2P2H_4th + sum_4P4H_4th + diagram_sum + H0[0,0] + H0[1,1])
        
    

    # plt.plot(g, sums_3rdorder,"r--", g, energy_FCI, "k-", g, energy_FCI_5dims, "g--")
    # plt.legend(["3rd order MBPT", "FCI", "5D CI"])
    # plt.title("Groundstate energy comparison: 3rd order MBPT vs CI")
    # plt.xlabel("g")
    # plt.ylabel(r"$E_{GS}$")
    # plt.savefig("3rdVS5dCI.pdf")
    # plt.show()
    
    rel_diff_3 = np.abs(np.array(energy_FCI) - np.array(sums_3rdorder))
    rel_diff_4 = np.abs(np.array(energy_FCI) - np.array(sums_4thorder))
    # plt.plot(g, sums_4thorder, "r--", label="4th order MBPT")
    # plt.plot(g, energy_FCI, "k-", label="FCI")
    # plt.plot(g, energy_FCI_5dims, "g--", label="5D CI")
    # plt.xlabel("g")
    # plt.ylabel(r"$E_{GS}$")
    # plt.title("Grounstate energy comparison: 4th order MBPT vs CI")
    # plt.legend()
    # plt.savefig("4thVS5dCI.pdf")
    # plt.show()


    plt.plot(g, rel_diff_3, label="Deviance in 3rd order MBPT")
    plt.plot(g, rel_diff_4, label="Deviance in 4th order MBPT")
    plt.title("The deviance from the true ground state, for 3rd and 4th order MBPT")
    plt.xlabel("g")
    plt.ylabel(r"$E_{GS} - E_{GSapprox}$")
    plt.legend()
    plt.savefig("energyerror_comaprison.pdf")

    exit()

    plt.plot(g, sums)
    plt.show()
    breakpoint()

    list_of_gs_eigvals = [eigvals[g_val][0] for g_val in g]
    plt.plot(g, list_of_gs_eigvals, label=f"{g}")
    plt.title("Groundstate energy of H, as a function of pairing parameter g := [-1,1]")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.legend(loc="best")
    plt.show() 
    breakpoint()

    # """ Testrun of code yields the following energies
    # Remember, run code with and without diagram 3 (Pauli violating diagram) + whichever unlinked diagrams arise from 
    # 4th order PT (4P4H)
    
    # """