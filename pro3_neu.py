import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, eye
from scipy.sparse import tril, triu                  # GEÄNDERT: für schnelle Gauss-Seidel-Zerlegung
from scipy.sparse.linalg import spsolve_triangular   # GEÄNDERT: schnelleres Lösen des unteren Dreieckssystems
from time import perf_counter


# ============================================================
# Gegeben aus der Aufgabenstellung:
# -Δu = f  in Ω = (0,1)^2
# u = g    auf dem Rand ∂Ω
# f(x,y) = -4
# g(x,y) = x^2 + y^2
# ============================================================

def f(x, y):
    # GEÄNDERT:
    # Alte Version war:
    # return -4.0 + 0.0 * x
    #
    # Neue Version funktioniert sicher für Skalare und Arrays.
    return -4.0 + 0.0 * (np.asarray(x) + np.asarray(y))


def g(x, y):
    return x**2 + y**2


# ============================================================
# Diskretisierung des Poisson-Problems
# Wir bauen A_h u_h = b_h.
#
# GEÄNDERT:
# Jetzt bauen wir das UNskalierte System:
#
# (4u_ij - Nachbarn) / h^2 = f_ij
#
# statt:
#
# 4u_ij - Nachbarn = h^2 f_ij
#
# Dadurch passt die Toleranz 10^-4 direkt zur Aufgabe.
# ============================================================

def build_system(n):
    """
    Baut das lineare Gleichungssystem A u = b
    für das 2D-Poisson-Problem.

    Nur die inneren Gitterpunkte sind Unbekannte.
    Die Randwerte g werden direkt in b eingebaut.

    GEÄNDERT:
    A enthält jetzt den Faktor 1/h^2.
    b enthält f ohne Multiplikation mit h^2.
    Randwerte werden mit 1/h^2 in b eingebaut.
    """

    # Gitterweite h = 1/n
    h = 1.0 / n

    # Anzahl der inneren Punkte pro Richtung
    m = n - 1

    # Gitterpunkte
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    # Vektor für Diagonalen
    e = np.ones(m)

    # 1D-Matrix für den Operator -d²/dx² ohne h^-2
    T = diags(
        diagonals=[-e, 2 * e, -e],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="csr"
    )

    # Einheitsmatrix
    I = eye(m, format="csr")

    # GEÄNDERT:
    # Alte Version:
    # A = kron(I, T, format="csr") + kron(T, I, format="csr")
    #
    # Neue Version:
    # Wir teilen durch h^2, damit A wirklich den Operator -Δ_h darstellt.
    A = (kron(I, T, format="csr") + kron(T, I, format="csr")) / h**2

    # Innere Gitterpunkte
    xi = x[1:-1]
    yi = y[1:-1]

    # 2D-Gitter der inneren Punkte
    X_inner, Y_inner = np.meshgrid(xi, yi, indexing="xy")

    # GEÄNDERT:
    # Alte Version:
    # B = h**2 * f(X_inner, Y_inner)
    #
    # Neue Version:
    # Weil A bereits durch h^2 geteilt ist, steht rechts direkt f.
    B = f(X_inner, Y_inner)

    # GEÄNDERT:
    # Alte Version:
    # B[:, 0] += g(0.0, yi)
    #
    # Neue Version:
    # Randwerte kommen durch die Formel mit Faktor 1/h^2 auf die rechte Seite.
    B[:, 0] += g(0.0, yi) / h**2       # linker Rand x=0
    B[:, -1] += g(1.0, yi) / h**2      # rechter Rand x=1
    B[0, :] += g(xi, 0.0) / h**2       # unterer Rand y=0
    B[-1, :] += g(xi, 1.0) / h**2      # oberer Rand y=1

    # Aus 2D-Array wird Vektor
    b = B.reshape(m * m)

    return A, b, x, y


# ============================================================
# Teil a)
# Jacobi(A, b, u0)
#
# GEÄNDERT:
# Die Jacobi-Iteration wird jetzt über das Residuum geschrieben:
#
# r^k = b - A u^k
# u^{k+1} = u^k + D^{-1} r^k
#
# Das ist mathematisch dieselbe Jacobi-Methode,
# aber man braucht die Restmatrix R nicht extra zu bauen.
# ============================================================

def Jacobi(A, b, u0, tol=1e-4, max_steps=1_000_000):
    """
    Löst A u = b mit dem Jacobi-Verfahren.

    Stoppt, wenn ||A u^k - b||_∞ < 10^-4.
    Gibt zurück:
    - u: Näherungslösung
    - steps: Anzahl der Iterationen
    """

    u = u0.astype(float).copy()

    # Diagonale von A
    D = A.diagonal()

    steps = 0

    # GEÄNDERT:
    # Wir verwenden r = b - A u.
    # Die Norm von r ist gleich der Norm von A u - b.
    r = b - A @ u
    residual = np.linalg.norm(r, ord=np.inf)

    while residual >= tol and steps < max_steps:

        # GEÄNDERT:
        # Alte Version:
        # u = (b - R @ u) / D
        #
        # Neue Version:
        # u = u + r / D
        #
        # Das ist dieselbe Jacobi-Formel, aber effizienter.
        u = u + r / D

        steps += 1

        # Residuum neu berechnen
        r = b - A @ u
        residual = np.linalg.norm(r, ord=np.inf)

    return u, steps


# ============================================================
# Teil a)
# GaussSeidel(A, b, u0)
#
# GEÄNDERT:
# Die alte Version hat eine Python-Schleife über alle Unbekannten benutzt.
# Das war mathematisch korrekt, aber langsam.
#
# Neue Version benutzt:
#
# A = L + D + U
#
# Dann gilt bei Gauss-Seidel:
#
# (L + D) u^{k+1} = b - U u^k
#
# Dieses Dreieckssystem lösen wir effizient mit scipy.
# ============================================================

def GaussSeidel(A, b, u0, tol=1e-4, max_steps=1_000_000):
    """
    Löst A u = b mit dem Gauss-Seidel-Verfahren.

    Stoppt, wenn ||A u^k - b||_∞ < 10^-4.
    Gibt zurück:
    - u: Näherungslösung
    - steps: Anzahl der Iterationen
    """

    A = A.tocsr()

    # GEÄNDERT:
    # Zerlegung von A in unteren Teil L+D und oberen Teil U.
    LD = tril(A, format="csr")          # unterer Dreiecksteil inklusive Diagonale
    U = triu(A, k=1, format="csr")      # strikt oberer Dreiecksteil

    u = u0.astype(float).copy()

    steps = 0

    residual = np.linalg.norm(A @ u - b, ord=np.inf)

    while residual >= tol and steps < max_steps:

        # GEÄNDERT:
        # Alte Version:
        # for i in range(N):
        #     u[i] = ...
        #
        # Neue Version:
        # Wir lösen das Dreieckssystem:
        # (L+D) u_neu = b - U u_alt
        rhs = b - U @ u
        u = spsolve_triangular(LD, rhs, lower=True)

        steps += 1

        residual = np.linalg.norm(A @ u - b, ord=np.inf)

    return u, steps


# ============================================================
# Hilfsfunktion für Teil b)
# Aus dem inneren Lösungsvektor wird wieder eine 2D-Lösung.
# ============================================================

def reconstruct_full_grid(u_inner, n, x, y):
    """
    Baut aus dem Vektor der inneren Werte eine volle Gitterlösung U.
    Randwerte werden mit g(x,y) gesetzt.
    """

    m = n - 1

    X, Y = np.meshgrid(x, y, indexing="xy")

    U = np.zeros_like(X)

    # Randwerte einsetzen
    U[:, 0] = g(0.0, y)       # linker Rand
    U[:, -1] = g(1.0, y)      # rechter Rand
    U[0, :] = g(x, 0.0)       # unterer Rand
    U[-1, :] = g(x, 1.0)      # oberer Rand

    # Innere Werte einsetzen
    U[1:-1, 1:-1] = u_inner.reshape((m, m))

    return X, Y, U


# ============================================================
# Teil b)
# Lösung für n = 50 berechnen und als Surface-Plot darstellen.
# ============================================================

def solve_and_plot_surface(n=50):
    """
    Löst das Poisson-Problem für n=50
    und erzeugt einen Surface-Plot.
    """

    A, b, x, y = build_system(n)

    # Startvektor u0 = Nullvektor
    u0 = np.zeros_like(b)

    # Mit Gauss-Seidel lösen
    u, steps = GaussSeidel(A, b, u0, tol=1e-4)

    print("Teil b)")
    print("n =", n)
    print("Anzahl der Unbekannten N =", len(b))
    print("Gauss-Seidel Iterationen =", steps)

    X, Y, U = reconstruct_full_grid(u, n, x, y)

    # Exakte Lösung zum Vergleich:
    # u(x,y)=x^2+y^2
    U_exact = X**2 + Y**2
    error = np.max(np.abs(U - U_exact))

    print("Maximaler Fehler gegen x^2 + y^2 =", error)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # GEÄNDERT:
    # Alte Version:
    # ax.plot_surface(X, Y, U)
    #
    # Neue Version:
    # shade=False verhindert die Matplotlib-Warnung beim 3D-Plot.
    ax.plot_surface(X, Y, U, shade=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("Poisson-Lösung für n = 50")

    plt.show()

    # Textliche Antwort für Teil b)
    print()
    print("Interpretation Teil b:")
    print("Die Randbedingung ist g(x,y)=x^2+y^2.")
    print("Sie legt die Werte der Lösung auf dem Rand des Einheitsquadrats fest.")
    print("Die rechte Seite f(x,y)=-4 bestimmt die Krümmung der Lösung im Inneren.")
    print("Da für u(x,y)=x^2+y^2 gilt: -Δu=-4, ist die exakte Lösung u=x^2+y^2.")


# ============================================================
# Teil c)
# Vergleich für verschiedene n:
# - Rechenzeit
# - Iterationsanzahl
# - durchschnittliche Zeit pro Iteration
# ============================================================

def compare_methods():
    """
    Vergleicht Jacobi und Gauss-Seidel für n = 10,20,...,100.
    Erstellt drei log-log Plots.
    """

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

        # Sicher gegen Division durch 0
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

        # Sicher gegen Division durch 0
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
    plt.show()

    # ------------------------
    # Plot 2: Iterationsanzahl
    # ------------------------
    plt.figure()
    plt.loglog(N_values, jacobi_steps, marker="o", label="Jacobi")
    plt.loglog(N_values, gs_steps, marker="o", label="Gauss-Seidel")
    plt.xlabel("Anzahl der Unbekannten N")
    plt.ylabel("Anzahl der Iterationen")
    plt.title("Iterationen vs. Anzahl der Unbekannten")
    plt.legend()
    plt.grid(True)
    plt.show()

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
    plt.show()


# ============================================================
# Teil d) Bonus
# Suche nach maximalem n, das noch in sinnvoller Zeit lösbar ist.
# ============================================================

def bonus_max_n(time_limit_seconds=30.0, n_max=300):
    """
    Testet, bis zu welchem n das Problem noch unter einer Zeitgrenze lösbar ist.

    Beispiel:
    sinnvoll = weniger als time_limit_seconds Sekunden.
    """

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
# Hauptprogramm
# ============================================================

if __name__ == "__main__":

    # Teil b)
    solve_and_plot_surface(n=50)

    # Teil c)
    compare_methods()

    # Teil d) Bonus
    # Achtung: Kann länger dauern.
    # bonus_max_n(time_limit_seconds=30.0, n_max=300)