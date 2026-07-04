import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, kron, eye
from time import perf_counter


# ============================================================
# Gegeben aus der Aufgabenstellung:
# -Δu = f  in Ω = (0,1)^2
# u = g    auf dem Rand ∂Ω
# f(x,y) = -4
# g(x,y) = x^2 + y^2
# ============================================================

def f(x, y):
    return -4.0 + 0.0 * x


def g(x, y):
    return x**2 + y**2


# ============================================================
# Diskretisierung des Poisson-Problems
# Dieser Teil gehört zur Einleitung vor a), b), c).
# Wir bauen A_h u_h = b_h.
# ============================================================

def build_system(n):
    """
    Baut das lineare Gleichungssystem A u = b
    für das 2D-Poisson-Problem auf.

    Wir benutzen nur die inneren Gitterpunkte als Unbekannte.
    Die Randwerte g werden direkt in b eingebaut.
    """

    # Gitterweite h = 1/n aus der Aufgabenstellung
    h = 1.0 / n

    # Anzahl der inneren Punkte pro Richtung:
    # Gesamtgitter: 0, 1, ..., n  -> n+1 Punkte
    # Innere Punkte: 1, ..., n-1  -> n-1 Punkte
    m = n - 1

    # Gitterpunkte in x- und y-Richtung
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    # Einheitsvektor für die Diagonalen
    e = np.ones(m)

    # 1D-Matrix für den Operator -d²/dx²:
    # [ 2 -1  0 ... ]
    # [-1  2 -1 ... ]
    # [ 0 -1  2 ... ]
    T = diags(
        diagonals=[-e, 2 * e, -e],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="csr"
    )

    # Einheitsmatrix der Größe m x m
    I = eye(m, format="csr")

    # 2D-Matrix für -Δ:
    # A = I ⊗ T + T ⊗ I
    #
    # Dadurch entsteht für jeden inneren Punkt:
    # 4*u_ij - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}
    A = kron(I, T, format="csr") + kron(T, I, format="csr")

    # Innere x- und y-Werte
    xi = x[1:-1]
    yi = y[1:-1]

    # 2D-Gitter der inneren Punkte
    X_inner, Y_inner = np.meshgrid(xi, yi, indexing="xy")

    # Rechte Seite:
    # Wir benutzen die mit h^2 multiplizierte Form:
    #
    # 4u_ij - Nachbarn = h^2 f_ij + Randbeiträge
    B = h**2 * f(X_inner, Y_inner)

    # Linker Rand: x = 0
    # Falls ein innerer Punkt direkt neben dem linken Rand liegt,
    # kommt g(0,y_j) auf die rechte Seite.
    B[:, 0] += g(0.0, yi)

    # Rechter Rand: x = 1
    B[:, -1] += g(1.0, yi)

    # Unterer Rand: y = 0
    B[0, :] += g(xi, 0.0)

    # Oberer Rand: y = 1
    B[-1, :] += g(xi, 1.0)

    # Aus der 2D-Rechten-Seite machen wir einen Vektor b
    b = B.reshape(m * m)

    return A, b, x, y


# ============================================================
# Teil a)
# Jacobi(A, b, u0)
# ============================================================

def Jacobi(A, b, u0, tol=1e-4, max_steps=1_000_000):
    """
    Löst A u = b mit dem Jacobi-Verfahren.

    Stoppt, wenn ||A u^k - b||_∞ < 10^-4.
    Gibt zurück:
    - u: Näherungslösung
    - steps: Anzahl der Iterationen
    """

    # Sicherheitskopie, damit der Startvektor u0 nicht verändert wird
    u = u0.astype(float).copy()

    # Diagonale von A
    D = A.diagonal()

    # R enthält alle Matrixeinträge außer der Diagonale
    R = A.copy()
    R.setdiag(0.0)
    R.eliminate_zeros()

    steps = 0

    # Anfangsresiduum ||A u^0 - b||_∞
    residual = np.linalg.norm(A @ u - b, ord=np.inf)

    # Iteration läuft, bis die Bedingung aus der Aufgabe erfüllt ist
    while residual >= tol and steps < max_steps:

        # Jacobi-Formel:
        # u^{k+1} = D^{-1} (b - R u^k)
        u = (b - R @ u) / D

        # Ein Iterationsschritt wurde durchgeführt
        steps += 1

        # Neues Residuum berechnen
        residual = np.linalg.norm(A @ u - b, ord=np.inf)

    return u, steps


# ============================================================
# Teil a)
# GaussSeidel(A, b, u0)
# ============================================================

def GaussSeidel(A, b, u0, tol=1e-4, max_steps=1_000_000):
    """
    Löst A u = b mit dem Gauss-Seidel-Verfahren.

    Stoppt, wenn ||A u^k - b||_∞ < 10^-4.
    Gibt zurück:
    - u: Näherungslösung
    - steps: Anzahl der Iterationen
    """

    # Wir benutzen CSR-Format, damit Zeilen effizient gelesen werden können
    A = A.tocsr()

    # Kopie des Startvektors
    u = u0.astype(float).copy()

    # Anzahl der Unbekannten
    N = len(b)

    # Diagonaleinträge von A
    D = A.diagonal()

    steps = 0

    # Anfangsresiduum
    residual = np.linalg.norm(A @ u - b, ord=np.inf)

    # Iteration bis zur gewünschten Genauigkeit
    while residual >= tol and steps < max_steps:

        # Bei Gauss-Seidel werden die Komponenten nacheinander aktualisiert
        for i in range(N):

            # Zugriff auf Zeile i der Matrix A
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]

            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            # Berechne Summe a_ij u_j über alle j,
            # aber ohne den Diagonaleintrag a_ii u_i
            sigma = vals @ u[cols] - D[i] * u[i]

            # Gauss-Seidel-Update:
            # u_i^{neu} = (b_i - Summe_{j != i} a_ij u_j) / a_ii
            #
            # Wichtig:
            # Für j < i sind u_j schon neue Werte.
            # Für j > i sind u_j noch alte Werte.
            u[i] = (b[i] - sigma) / D[i]

        steps += 1

        # Residuum nach einer vollständigen Iteration
        residual = np.linalg.norm(A @ u - b, ord=np.inf)

    return u, steps


# ============================================================
# Hilfsfunktion:
# Aus dem inneren Lösungsvektor wieder eine 2D-Lösung machen
# Das brauchen wir für Teil b), den Surface-Plot.
# ============================================================

def reconstruct_full_grid(u_inner, n, x, y):
    """
    Baut aus dem Vektor der inneren Werte eine volle Gitterlösung U.
    Randwerte werden mit g(x,y) gesetzt.
    """

    m = n - 1

    X, Y = np.meshgrid(x, y, indexing="xy")

    # Volle Lösungsmatrix
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
# Lösung für n = 50 berechnen und als Surface-Plot darstellen
# ============================================================

def solve_and_plot_surface(n=50):
    """
    Löst das Poisson-Problem für n=50
    und erzeugt einen Surface-Plot.
    """

    # System A u = b bauen
    A, b, x, y = build_system(n)

    # Startvektor u0 = Nullvektor, wie in der Aufgabe verlangt
    u0 = np.zeros_like(b)

    # Wir lösen hier mit Gauss-Seidel.
    # Jacobi wäre auch möglich, aber Gauss-Seidel braucht oft weniger Iterationen.
    u, steps = GaussSeidel(A, b, u0)

    print("Teil b)")
    print("n =", n)
    print("Anzahl der Unbekannten N =", len(b))
    print("Gauss-Seidel Iterationen =", steps)

    # Lösung auf vollem Gitter rekonstruieren
    X, Y, U = reconstruct_full_grid(u, n, x, y)

    # Exakte Lösung zum Vergleich:
    # Weil -Δ(x^2+y^2) = -4 und g=x^2+y^2 ist.
    U_exact = X**2 + Y**2
    error = np.max(np.abs(U - U_exact))

    print("Maximaler Fehler gegen x^2 + y^2 =", error)

    # Surface-Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, U)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    ax.set_title("Poisson-Lösung für n = 50")

    plt.show()


# ============================================================
# Teil c)
# Für verschiedene n messen:
# - Rechenzeit
# - Anzahl der Iterationen
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
        u_jacobi, steps_jacobi = Jacobi(A, b, u0)
        end = perf_counter()

        time_jacobi = end - start

        jacobi_times.append(time_jacobi)
        jacobi_steps.append(steps_jacobi)
        jacobi_time_per_step.append(time_jacobi / steps_jacobi)

        # ------------------------
        # Gauss-Seidel messen
        # ------------------------
        start = perf_counter()
        u_gs, steps_gs = GaussSeidel(A, b, u0)
        end = perf_counter()

        time_gs = end - start

        gs_times.append(time_gs)
        gs_steps.append(steps_gs)
        gs_time_per_step.append(time_gs / steps_gs)

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

    Hier nehmen wir als Beispiel:
    sinnvoll = weniger als time_limit_seconds Sekunden.
    """

    last_good_n = None

    for n in range(10, n_max + 1, 10):
        print("Teste n =", n)

        A, b, x, y = build_system(n)
        u0 = np.zeros_like(b)

        start = perf_counter()
        u, steps = GaussSeidel(A, b, u0)
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
# Hier kannst du entscheiden, was ausgeführt wird.
# ============================================================

if __name__ == "__main__":

    # Teil b)
    solve_and_plot_surface(n=50)

    # Teil c)
    compare_methods()

    # Teil d) Bonus
    # Achtung: Kann länger dauern.
    # bonus_max_n(time_limit_seconds=30.0, n_max=300)