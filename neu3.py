import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, eye, tril, triu
from scipy.sparse.linalg import splu
from time import perf_counter


# ============================================================
# Gegeben:
# -Δu = f  in Ω = (0,1)^2
# u = g    auf dem Rand
# f(x,y) = -4
# g(x,y) = x^2 + y^2
# ============================================================

def f(x, y):
    return -4.0 + 0.0 * (np.asarray(x) + np.asarray(y))


def g(x, y):
    return x**2 + y**2


# ============================================================
# Diskretisierung des Poisson-Problems
#
# Wir bauen das unskalierte System:
#
# (4u_ij - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}) / h^2 = f_ij
#
# Dadurch passt die Abbruchbedingung:
#
# ||A u - b||_∞ < 10^-4
#
# direkt zur Aufgabenstellung.
# ============================================================

def build_system(n):
    h = 1.0 / n
    m = n - 1

    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    e = np.ones(m)

    # 1D-Matrix für -d²/dx² ohne h^-2
    T = diags(
        diagonals=[-e, 2 * e, -e],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="csr"
    )

    I = eye(m, format="csr")

    # Matrix für -Δ_h mit Faktor 1/h²
    A = (kron(I, T, format="csr") + kron(T, I, format="csr")) / h**2

    xi = x[1:-1]
    yi = y[1:-1]

    X_inner, Y_inner = np.meshgrid(xi, yi, indexing="xy")

    # Rechte Seite f
    B = f(X_inner, Y_inner)

    # Randwerte in die rechte Seite einbauen
    B[:, 0] += g(0.0, yi) / h**2       # linker Rand x=0
    B[:, -1] += g(1.0, yi) / h**2      # rechter Rand x=1
    B[0, :] += g(xi, 0.0) / h**2       # unterer Rand y=0
    B[-1, :] += g(xi, 1.0) / h**2      # oberer Rand y=1

    b = B.reshape(m * m)

    return A, b, x, y


# ============================================================
# Teil a)
# Jacobi-Verfahren
#
# Methode:
# u^{k+1} = u^k + D^{-1}(b - A u^k)
#
# Kosten pro Iteration:
# O(N), weil A dünnbesetzt ist.
# ============================================================

def Jacobi(A, b, u0, tol=1e-4, max_steps=1_000_000):
    u = u0.astype(float).copy()

    D = A.diagonal()

    steps = 0

    r = b - A @ u
    residual = np.linalg.norm(r, ord=np.inf)

    while residual >= tol and steps < max_steps:
        u = u + r / D

        steps += 1

        r = b - A @ u
        residual = np.linalg.norm(r, ord=np.inf)

    return u, steps


# ============================================================
# Teil a)
# Klassische Gauss-Seidel-Version
#
# Diese Version ist gut, falls du im Testat zeigen musst,
# wie Gauss-Seidel wirklich komponentenweise arbeitet.
#
# Sie ist mathematisch korrekt, aber langsamer wegen Python-Schleifen.
# ============================================================

def GaussSeidel_classic(A, b, u0, tol=1e-4, max_steps=1_000_000):
    A = A.tocsr()

    u = u0.astype(float).copy()
    N = len(b)
    D = A.diagonal()

    steps = 0

    residual = np.linalg.norm(A @ u - b, ord=np.inf)

    while residual >= tol and steps < max_steps:

        for i in range(N):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]

            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            sigma = vals @ u[cols] - D[i] * u[i]

            u[i] = (b[i] - sigma) / D[i]

        steps += 1

        residual = np.linalg.norm(A @ u - b, ord=np.inf)

    return u, steps


# ============================================================
# Teil a)
# Optimierte Gauss-Seidel-Version
#
# Mathematisch bleibt es Gauss-Seidel:
#
# A = L + D + U
#
# (L + D) u^{k+1} = b - U u^k
#
# Änderung:
# L + D wird einmal vorbereitet.
# Danach wird in jeder Iteration nur noch solve_LD(rhs) benutzt.
#
# Die Methode bleibt Gauss-Seidel, aber die Implementierung ist schneller.
# ============================================================

def GaussSeidel(A, b, u0, tol=1e-4, max_steps=1_000_000):
    A = A.tocsr()

    # Unterer Dreiecksteil inklusive Diagonale
    LD = tril(A, format="csc")

    # Strikt oberer Dreiecksteil
    U = triu(A, k=1, format="csr")

    # Einmalige Faktorisierung von L + D
    solve_LD = splu(
        LD,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0
    ).solve

    u = u0.astype(float).copy()

    steps = 0

    residual = np.linalg.norm(A @ u - b, ord=np.inf)

    while residual >= tol and steps < max_steps:
        rhs = b - U @ u

        # Gauss-Seidel-Schritt:
        # (L+D) u_neu = b - U u_alt
        u = solve_LD(rhs)

        steps += 1

        residual = np.linalg.norm(A @ u - b, ord=np.inf)

    return u, steps


# ============================================================
# Hilfsfunktion für Teil b)
# Aus dem inneren Lösungsvektor wird wieder eine volle 2D-Lösung.
# ============================================================

def reconstruct_full_grid(u_inner, n, x, y):
    m = n - 1

    X, Y = np.meshgrid(x, y, indexing="xy")

    U = np.zeros_like(X)

    # Randwerte
    U[:, 0] = g(0.0, y)
    U[:, -1] = g(1.0, y)
    U[0, :] = g(x, 0.0)
    U[-1, :] = g(x, 1.0)

    # Innere Werte
    U[1:-1, 1:-1] = u_inner.reshape((m, m))

    return X, Y, U


# ============================================================
# Teil b)
# Lösung für n = 50 berechnen und Surface-Plot erstellen.
# ============================================================

def solve_and_plot_surface(n=50):
    A, b, x, y = build_system(n)

    u0 = np.zeros_like(b)

    start = perf_counter()
    u, steps = GaussSeidel(A, b, u0, tol=1e-4)
    end = perf_counter()

    print("Teil b)")
    print("n =", n)
    print("Anzahl der Unbekannten N =", len(b))
    print("Gauss-Seidel Iterationen =", steps)
    print("Gauss-Seidel Zeit =", end - start, "Sekunden")

    X, Y, U = reconstruct_full_grid(u, n, x, y)

    U_exact = X**2 + Y**2
    error = np.max(np.abs(U - U_exact))

    print("Maximaler Fehler gegen x^2 + y^2 =", error)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, U, shade=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("Poisson-Lösung für n = 50")

    # plt.show()

    print()
    print("Interpretation Teil b:")
    print("Die Randbedingung ist g(x,y)=x^2+y^2.")
    print("Sie legt die Werte der Lösung auf dem Rand des Einheitsquadrats fest.")
    print("Die rechte Seite f(x,y)=-4 bestimmt die Krümmung der Lösung im Inneren.")
    print("Da für u(x,y)=x^2+y^2 gilt: -Δu=-4, ist die exakte Lösung u=x^2+y^2.")


# ============================================================
# Teil c)
# Vergleich von Jacobi und Gauss-Seidel
#
# Für verschiedene n werden gemessen:
# - Rechenzeit
# - Anzahl der Iterationen
# - durchschnittliche Zeit pro Iteration
# ============================================================

def compare_methods():
    ns = range(10, 101, 10)

    N_values = []

    jacobi_times = []
    gs_times = []

    jacobi_steps = []
    gs_steps = []

    jacobi_time_per_step = []
    gs_time_per_step = []

    for n in ns:
        print("Berechne n =", n)

        A, b, x, y = build_system(n)

        u0 = np.zeros_like(b)

        N = len(b)
        N_values.append(N)

        # ------------------------
        # Jacobi messen
        # ------------------------
        start = perf_counter()
        u_jacobi, steps_jacobi = Jacobi(A, b, u0, tol=1e-4)
        end = perf_counter()

        time_jacobi = end - start

        jacobi_times.append(time_jacobi)
        jacobi_steps.append(steps_jacobi)
        jacobi_time_per_step.append(time_jacobi / max(steps_jacobi, 1))

        # ------------------------
        # Gauss-Seidel messen
        # ------------------------
        start = perf_counter()
        u_gs, steps_gs = GaussSeidel(A, b, u0, tol=1e-4)
        end = perf_counter()

        time_gs = end - start

        gs_times.append(time_gs)
        gs_steps.append(steps_gs)
        gs_time_per_step.append(time_gs / max(steps_gs, 1))

        print("N =", N)
        print("Jacobi:       steps =", steps_jacobi, ", time =", time_jacobi)
        print("Gauss-Seidel: steps =", steps_gs, ", time =", time_gs)
        print()

    # ------------------------
    # Plot 1: Rechenzeit
    # ------------------------
    plt.figure()
    plt.loglog(N_values, jacobi_times, marker="o", label="Jacobi")
    plt.loglog(N_values, gs_times, marker="o", label="Gauss-Seidel")
    plt.xlabel("Anzahl der Unbekannten N")
    plt.ylabel("Rechenzeit in Sekunden")
    plt.title("Rechenzeit vs. Anzahl der Unbekannten")
    plt.legend()
    plt.grid(True)
    # plt.show()

    # ------------------------
    # Plot 2: Iterationen
    # ------------------------
    plt.figure()
    plt.loglog(N_values, jacobi_steps, marker="o", label="Jacobi")
    plt.loglog(N_values, gs_steps, marker="o", label="Gauss-Seidel")
    plt.xlabel("Anzahl der Unbekannten N")
    plt.ylabel("Anzahl der Iterationen")
    plt.title("Iterationen vs. Anzahl der Unbekannten")
    plt.legend()
    plt.grid(True)
    # plt.show()

    # ------------------------
    # Plot 3: Zeit pro Iteration
    # ------------------------
    plt.figure()
    plt.loglog(N_values, jacobi_time_per_step, marker="o", label="Jacobi")
    plt.loglog(N_values, gs_time_per_step, marker="o", label="Gauss-Seidel")
    plt.xlabel("Anzahl der Unbekannten N")
    plt.ylabel("Durchschnittliche Zeit pro Iteration")
    plt.title("Zeit pro Iteration vs. Anzahl der Unbekannten")
    plt.legend()
    plt.grid(True)
    # plt.show()


# ============================================================
# Teil d) Bonus
# Maximales n suchen, das noch in sinnvoller Zeit lösbar ist.
# ============================================================

def bonus_max_n(time_limit_seconds=30.0, n_max=300):
    last_good_n = None

    for n in range(10, n_max + 1, 10):
        print("Teste n =", n)

        A, b, x, y = build_system(n)
        u0 = np.zeros_like(b)

        start = perf_counter()
        u, steps = GaussSeidel(A, b, u0, tol=1e-4)
        end = perf_counter()

        elapsed = end - start

        print("Zeit =", elapsed, "Sekunden")
        print("Iterationen =", steps)

        if elapsed <= time_limit_seconds:
            last_good_n = n
        else:
            print("Abbruch: n =", n, "dauert länger als sinnvoll.")
            break

    print("Maximales sinnvolles n ungefähr:", last_good_n)


# ============================================================
# Zusatz: theoretische Ordnung für Präsentation/Testat
# ============================================================

def print_theory_summary():
    print()
    print("Theoretische Einordnung:")
    print("Anzahl der Unbekannten: N = (n-1)^2 = O(n^2)")
    print("Nichtnull-Einträge der Matrix: O(N)")
    print("Kosten pro Jacobi-Iteration: O(N)")
    print("Kosten pro Gauss-Seidel-Iteration: O(N)")
    print("Gemessene Iterationszahl wächst ungefähr wie O(N)")
    print("Gesamtlaufzeit daher ungefähr: O(N^2) = O(n^4)")
    print("Speicherbedarf: O(N) = O(n^2)")
    print("Konsistenzordnung der zentralen Differenzen: O(h^2)")


# ============================================================
# Hauptprogramm
# ============================================================

if __name__ == "__main__":

    # Teil b)
    solve_and_plot_surface(n=50)

    # Teil c)
    compare_methods()

    # Theorie-Zusammenfassung
    print_theory_summary()
    plt.show()

    # Teil d) Bonus
    # Achtung: Kann länger dauern.
    # bonus_max_n(time_limit_seconds=30.0, n_max=300)