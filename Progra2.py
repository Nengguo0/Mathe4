import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, identity, csr_matrix
from scipy.sparse.linalg import spsolve

# =========================================================
# Aufgabe a) Upwind-Differenzenquotient
# =========================================================
def solve_conv_diff_upwind(n, eps, beta):
    h = 1/n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)

    N = (n-1)**2 # Anzahl der inneren Gitterpunkte

    alpha_x = np.cos(beta) # Upwind-Richtungen
    alpha_y = np.sin(beta)

    e = np.ones(n-1)
    Lap1D = diags([e, -2*e, e], [-1, 0, 1], shape=(n-1, n-1)) / h**2

    # Upwind Fallunterscheidung
    if alpha_x >= 0:
        Dx1D = diags([-e, e], [-1, 0], shape=(n-1, n-1)) / h
    else:
        Dx1D = diags([-e, e], [0, 1], shape=(n-1, n-1)) / h
    
    if alpha_y >= 0:
        Dy1D = diags([-e, e], [-1, 0], shape=(n-1, n-1)) / h
    else:
        Dy1D = diags([-e, e], [0, 1], shape=(n-1, n-1)) / h

    I = identity(n-1)

    # 2D Operatoren mit Kronecker-Produkt
    Lap2D = kron(I, Lap1D) + kron(Lap1D, I)
    Dx2D = kron(Dx1D, I)
    Dy2D = kron(I, Dy1D)

    A = -eps * Lap2D + alpha_x * Dx2D + alpha_y * Dy2D
    A = csr_matrix(A)

    b = np.ones(N) # f(x,y) = 1

    u_inner = spsolve(A, b)

    u = np.zeros((n+1, n+1))
    for i in range(1, n):
        for j in range(1, n):
            u[i,j] = u_inner[(i-1)*(n-1) + (j-1)]
    
    return x, y, u


# =========================================================
# Aufgabe b) Zentraler Differenzenquotient
# =========================================================
def solve_conv_diff_central(n, eps, beta):
    h = 1/n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)

    N = (n-1)**2

    alpha_x = np.cos(beta)
    alpha_y = np.sin(beta)

    e = np.ones(n-1)
    Lap1D = diags([e, -2*e, e], [-1, 0, 1], shape=(n-1, n-1)) / h**2
    D1D = diags([-e, e], [-1, 1], shape=(n-1, n-1)) / (2*h) # Zentrale Differenz

    I = identity(n-1)

    Lap2D = kron(I, Lap1D) + kron(Lap1D, I)
    Dx2D = kron(D1D, I)
    Dy2D = kron(I, D1D)

    A = -eps * Lap2D + alpha_x * Dx2D + alpha_y * Dy2D
    A = csr_matrix(A)

    b = np.ones(N)

    u_inner = spsolve(A, b)

    u = np.zeros((n+1, n+1))
    for i in range(1, n):
        for j in range(1, n):
            u[i, j] = u_inner[(i-1)*(n-1) + (j-1)]

    return x, y, u


# =========================================================
# Aufgabe c) Poisson-Problem
# =========================================================
def solve_poisson_2d(n):
    h = 1 / n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    
    N = (n-1)**2 

    e = np.ones(n-1)
    # [-1, 2, -1] entspricht der negativen zweiten Ableitung
    Lap1D = diags([-e, 2*e, -e], [-1, 0, 1], shape=(n-1, n-1)) / h**2
    I = identity(n-1)

    A = kron(I, Lap1D) + kron(Lap1D, I)
    A = csr_matrix(A)

    b = -4 * np.ones(N) # f(x,y) = -4

    # Randbedingungsfunktion g(x,y) = x^2 + y^2
    def g(x_val, y_val):
        return x_val**2 + y_val**2

    # Übertragung der Randbedingungen
    for i in range(1, n):       
        for j in range(1, n):   
            k = (i-1)*(n-1) + (j-1)  
            
            if i == 1:       b[k] += g(x[0], y[j]) / h**2
            if i == n-1:     b[k] += g(x[n], y[j]) / h**2
            if j == 1:       b[k] += g(x[i], y[0]) / h**2
            if j == n-1:     b[k] += g(x[i], y[n]) / h**2

    u_inner = spsolve(A, b)

    u = np.zeros((n+1, n+1))
    
    u[0, :] = g(x[0], y)  
    u[-1, :] = g(x[-1], y) 
    u[:, 0] = g(x, y[0])  
    u[:, -1] = g(x, y[-1]) 

    for i in range(1, n):
        for j in range(1, n):
            k = (i-1)*(n-1) + (j-1)
            u[i, j] = u_inner[k]

    return x, y, u


# =========================================================
# Ausführung und Visualisierung (Main)
# =========================================================
if __name__ == "__main__":
    n_ab = 60
    beta = (5 * np.pi) / 6
    eps_values = [1, 1e-2, 1e-4]

    # --- a) Plot für Upwind ---
    fig_a = plt.figure(figsize=(15, 5))
    for idx, eps_val in enumerate(eps_values):
        x, y, u = solve_conv_diff_upwind(n_ab, eps_val, beta)
        X, Y = np.meshgrid(x, y, indexing='ij')

        ax = fig_a.add_subplot(1, 3, idx+1, projection='3d')
        ax.plot_surface(X, Y, u, cmap='viridis')
        ax.set_title(f'Upwind: e = {eps_val:.0e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
    plt.suptitle('Aufgabe a): Konvektive Diffusionsgleichung (Upwind)')
    plt.tight_layout()
    plt.show()

    # --- b) Plot für Zentral vs Upwind ---
    fig_b = plt.figure(figsize=(12, 6))
    for idx, eps_val in enumerate(eps_values):
        # Upwind
        x, y, u_up = solve_conv_diff_upwind(n_ab, eps_val, beta)
        X, Y = np.meshgrid(x, y, indexing='ij')

        ax = fig_b.add_subplot(2, 3, idx+1, projection='3d')
        ax.plot_surface(X, Y, u_up, cmap='viridis')
        ax.set_title(f'Upwind: e = {eps_val:.0e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Central
        x, y, u_cent = solve_conv_diff_central(n_ab, eps_val, beta)
        X, Y = np.meshgrid(x, y, indexing='ij')

        ax = fig_b.add_subplot(2, 3, idx+4, projection='3d')
        ax.plot_surface(X, Y, u_cent, cmap='plasma')
        ax.set_title(f'Zentral: e = {eps_val:.0e}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.suptitle('Aufgabe b): Vergleich Upwind vs Zentraler Differenzenquotient')
    plt.tight_layout()
    plt.show()

    # --- c) Plot für Poisson ---
    n_c = 40 
    x_p, y_p, u_p = solve_poisson_2d(n_c)
    X_p, Y_p = np.meshgrid(x_p, y_p, indexing='ij')

    fig_c = plt.figure(figsize=(8, 6))
    ax = fig_c.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_p, Y_p, u_p, cmap='plasma', edgecolor='none')
    ax.set_title('Aufgabe c): Lösung des Poisson-Problems\nf(x,y)=-4, g(x,y)=x^2+y^2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    fig_c.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()