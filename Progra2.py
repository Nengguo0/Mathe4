import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, identity, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

def solve_diffusion_1d(n):
    h = 1 / n
    x = np.linspace(0, 1, n+1)
    b = np.ones(n-1)

    # Matrix A
    A = np.diag([-2] * (n-1)) + np.diag([1] * (n-2), 1) + np.diag([1] * (n-2), -1)
    A = -A / h**2

    u_inner = np.linalg.solve(A, b)
    u = np.zeros(n+1)
    u[1:-1] = u_inner

    # Analytische Lösung
    u_exact = -0.5 * (x - 0.5)**2 + 1/8

    # Fehlernormen
    l2_error = np.sqrt(np.sum((u - u_exact)**2) * h)
    linf_error = np.max(np.abs(u - u_exact))

    return l2_error, linf_error


# Fehler für verschiedene n
n_values = [4, 8, 16, 32, 64]
l2_upwind, linf_upwind = [], []
l2_central, linf_central = [], []

for n in n_values:
    l2_u, linf_u = solve_diffusion_1d(n)
    l2_upwind.append(l2_u)
    linf_upwind.append(linf_u)

# Log-Log Plot
plt.figure(figsize=(8, 5))
plt.semilogy(n_values, l2_upwind, 'o--', label='L2-Fehler (Upwind)')
plt.semilogy(n_values, linf_upwind, 's--', label='L∞-Fehler (Upwind)')
plt.xlabel('n (Anzahl Unterteilungen)')
plt.ylabel('Fehlernorm')
plt.title('Log-Log-Plot der Fehlernormen (1D Diffusion)')
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.show()

############################################################

def solve_triv_transport(n):
    h = 1 / n
    x = np.linspace(0, 1, n+1)
    u = np.zeros(n+1)
    u[0] = 0

    for i in range(1, n+1):
        u[i] = u[i-1] + h

    u_exact = x

    # Fehler
    l2_error = np.sqrt(np.sum((u - u_exact)**2) * h)
    linf_error = np.max(np.abs(u - u_exact))
    
    return l2_error, linf_error

n_values = [4, 8, 16, 32, 64]
l2_errors = []
linf_errors = []

for n in n_values:
    l2, linf = solve_triv_transport(n)
    l2_errors.append(l2)
    linf_errors.append(linf)

# Log-Log Plot
plt.figure(figsize=(8, 5))
plt.plot(n_values, l2_errors, 'o-', label='L2-Fehler')
plt.plot(n_values, linf_errors, 's-', label='L∞-Fehler')
plt.xlabel('n (Anzahl Unterteilungen)')
plt.ylabel('Fehlernorm')
plt.title('Log-Log-Plot der Fehlernormen (Triviale 1D Transportgleichung)')
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.show()

###############################################################

def solve_transport(n):
    a, b = 0, 2 * np.pi
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    
    u = np.zeros(n+1)
    u[0] = 1  # Randbedingung u(0) = 1

    # Upwind: rekursiv
    for i in range(1, n+1):
        u[i] = u[i-1] - h * np.sin(x[i])  # beachte Minuszeichen in -sin(x)

    # Analytische Lösung
    u_exact = np.cos(x)

    # Fehlernormen
    l2_error = np.sqrt(np.sum((u - u_exact)**2) * h)
    linf_error = np.max(np.abs(u - u_exact))
    return l2_error, linf_error

n_values = [4, 8, 16, 32, 64]
l2_errors = []
linf_errors = []

for n in n_values:
    l2, linf = solve_transport(n)
    l2_errors.append(l2)
    linf_errors.append(linf)

# Log-Log Plot
plt.figure(figsize=(8, 5))
plt.loglog(n_values, l2_errors, 'o-', label='L2-Fehler')
plt.loglog(n_values, linf_errors, 's-', label='L∞-Fehler')
plt.xlabel('n (Anzahl Unterteilungen)')
plt.ylabel('Fehlernorm')
plt.title('Log-Log-Plot der Fehlernormen (1D Transportgleichung)')
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.show()

###################################################



###############################################################
## a)

def solve_conv_diff_upwind(n, eps, beta):
    h = 1/n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    N = (n-1)**2 #Number innere points meshgrid

    alpha_x = np.cos(beta) # upwind directions
    alpha_y = np.sin(beta)

    e = np.ones(n-1)
    Lap1D = diags([e, -2*e, e], [-1, 0,1], shape=(n-1, n-1)) / h**2

    if alpha_x >=0:
        Dx1D = diags([-e, e], [-1, 0], shape=(n-1, n-1)) / h  # 改为 [-e, e]
    else:
        Dx1D = diags([-e, e], [0, 1], shape=(n-1, n-1)) / h   # 改为 [-e, e]
    
    if alpha_y >=0:
        Dy1D = diags([-e, e], [-1, 0], shape=(n-1, n-1)) / h  # 改为 [-e, e]
    else:
        Dy1D = diags([-e, e], [0, 1], shape=(n-1, n-1)) / h   # 改为 [-e, e]

    I = identity(n-1)

    Lap2D = kron(I, Lap1D) + kron(Lap1D, I)
    Dx2D = kron(Dx1D, I)
    Dy2D = kron(I, Dy1D)

    A = -eps * Lap2D + alpha_x * Dx2D + alpha_y * Dy2D
    A = csr_matrix(A)

    b = np.ones(N)

    u_inner = spsolve(A, b)

    u = np.zeros((n+1, n+1))
    for i in range(1, n):
        for j in range(1, n):
            u[i,j] = u_inner[(i-1)*(n-1) + (j-1)]
    
    return x, y, u

n = 60
beta = (5* np.pi) / 6
eps = [1, 1e-2, 1e-4]

fig = plt.figure(figsize=(15, 5))
for idx, eps_val in enumerate(eps):
    x, y, u = solve_conv_diff_upwind(n, eps_val, beta)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ax = fig.add_subplot(1, 3, idx+1, projection= '3d')
    ax.plot_surface(X, Y, u, cmap='viridis')
    ax.set_title(f'e = {eps_val:.0e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')

plt.suptitle('Lösung der konvektiven Diffusionsgleichung (Upwind)')
plt.tight_layout()
# plt.show()


###############################################################
## b)
def solve_conv_diff_central(n, eps, beta):
    h = 1/n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    N = (n-1)**2

    alpha_x = np.cos(beta)
    alpha_y = np.sin(beta)

    e = np.ones(n-1)
    Lap1D = diags([e, -2*e, e], [-1, 0, 1], shape=(n-1, n-1)) / h**2
    D1D = diags([-e, e], [-1, 1], shape=(n-1, n-1)) / (2*h)

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

# Visualisierung Upwind vs Central für Vergleich
fig = plt.figure(figsize=(12, 6))
for idx, eps_val in enumerate(eps):
    # Upwind
    x, y, u_up = solve_conv_diff_upwind(n, eps_val, beta)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ax = fig.add_subplot(2, 3, idx+1, projection='3d')
    ax.plot_surface(X, Y, u_up, cmap='viridis')
    ax.set_title(f'Upwind: e = {eps_val:.0e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')

    # Central
    x, y, u_cent = solve_conv_diff_central(n, eps_val, beta)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ax = fig.add_subplot(2, 3, idx+4, projection='3d')
    ax.plot_surface(X, Y, u_cent, cmap='plasma')
    ax.set_title(f'Zentral: e = {eps_val:.0e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')

plt.suptitle('Vergleich Upwind vs Zentraler Differenzenquotient')
plt.tight_layout()
#plt.show()
def solve_poisson_2d(n):
    h = 1 / n
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    
    # Anzahl der inneren Punkte
    N = (n-1)**2 

    # 1. Konstruktion des 1D Laplace-Operators (negativer Laplace-Operator -Delta)
    e = np.ones(n-1)
    # [-1, 2, -1] entspricht der negativen zweiten Ableitung
    Lap1D = diags([-e, 2*e, -e], [-1, 0, 1], shape=(n-1, n-1)) / h**2
    I = identity(n-1)

    # 2. Konstruktion der 2D Laplace-Matrix A (A = -Delta)
    # Verwendung des Kronecker-Produkts zur Erweiterung auf 2D
    A = kron(I, Lap1D) + kron(Lap1D, I)
    A = csr_matrix(A)

    # 3. Konstruktion der rechten Seite b (f(x,y) = -4)
    b = -4 * np.ones(N)

    # Randbedingungsfunktion g(x,y) = x^2 + y^2
    def g(x_val, y_val):
        return x_val**2 + y_val**2

    # 4. Übertragung der Randbedingungen auf die rechte Seite b
    # Da A * u_inner = f + boundary_terms
    for i in range(1, n):       # x-Richtung (innere Punkte)
        for j in range(1, n):   # y-Richtung (innere Punkte)
            k = (i-1)*(n-1) + (j-1)  # Flacher Index
            
            # Linker Rand (i=1 bei x=0)
            if i == 1:       
                b[k] += g(x[0], y[j]) / h**2
            # Rechter Rand (i=n-1 bei x=1)
            if i == n-1:     
                b[k] += g(x[n], y[j]) / h**2
            # Unterer Rand (j=1 bei y=0)
            if j == 1:       
                b[k] += g(x[i], y[0]) / h**2
            # Oberer Rand (j=n-1 bei y=1)
            if j == n-1:     
                b[k] += g(x[i], y[n]) / h**2

    # 5. Lösen des linearen Gleichungssystems
    u_inner = spsolve(A, b)

    # 6. Rücktransformation in die 2D-Matrix und Einfügen der Randwerte
    u = np.zeros((n+1, n+1))
    
    # Ränder setzen
    u[0, :] = g(x[0], y)  # Links
    u[-1, :] = g(x[-1], y) # Rechts
    u[:, 0] = g(x, y[0])  # Unten
    u[:, -1] = g(x, y[-1]) # Oben

    # Innere Punkte einfügen
    for i in range(1, n):
        for j in range(1, n):
            k = (i-1)*(n-1) + (j-1)
            u[i, j] = u_inner[k]

    return x, y, u


# ==========================================
# Ausführung und 3D-Plot für Aufgabe c)
# ==========================================
n_poisson = 40 
x_p, y_p, u_p = solve_poisson_2d(n_poisson)
X_p, Y_p = np.meshgrid(x_p, y_p, indexing='ij')

# Berechnung der exakten Lösung zur Validierung
u_exact = X_p**2 + Y_p**2
max_error = np.max(np.abs(u_p - u_exact))
print(f"Maximaler absoluter Fehler in c): {max_error:.2e}") 

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_p, Y_p, u_p, cmap='plasma', edgecolor='none')
ax.set_title(f'Lösung des Poisson-Problems (c)\nf(x,y)=-4, g(x,y)=x^2+y^2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()