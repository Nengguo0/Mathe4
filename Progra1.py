import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Stützstellen(N, M): 
    j = np.arange(N)
    k = np.arange(M)
    x_j = (2 * j + 1) / (2 * N) * np.pi
    y_k = (2 * k + 1) / (2 * M) * np.pi
    return x_j, y_k

def c(n):
    c = np.ones(n)
    c[0] = 0.5
    return c

# def DCT(F, x, y):
#     N, M = F.shape
#     d = np.zeros((N, M))
#     for j in range(N):
#         for k in range(M):
#             total = 0
#             for n in range(N):
#                 for m in range(M):
#                     total += F[n, m] * np.cos(j * x[n]) * np.cos(k * y[m])
#             d[j, k] = (4 / (M * N)) * total
#     return d

# def DCT(F, x, y):
#     N, M = F.shape
#     cos_jx = np.cos(np.outer(np.arange(N), x))   # N × N
#     cos_ky = np.cos(np.outer(np.arange(M), y))   # M × M
#     d = (4 / (M * N)) * cos_jx @ F @ cos_ky.T # N × N @ N × M @ M × M → N × M
#     return d

def DCT(F):

    N, M = F.shape
    x_nodes, y_nodes = Stützstellen(N, M)

    # cos_mat_x[j, n] = cos(j * x_n)
    cos_mat_x = np.cos(np.outer(np.arange(N), x_nodes))
    cos_mat_y = np.cos(np.outer(np.arange(M), y_nodes))
    
    # D = (4/NM) * C_x * F * C_y.T
    D = (4.0 / (N * M)) * (cos_mat_x @ F @ cos_mat_y.T)
    return D


# def TDCT(D, x, y):
#     N, M = D.shape
#     P, Q = len(x), len(y)
#     A = np.zeros((P, Q))

#     for p in range(P):
#         for q in range(Q):
#             total = 0
#             for j in range(N):
#                 for k in range(M):
#                     total += D[j, k] * c(j) * c(k) * np.cos(j * x[p]) * np.cos(k * y[q])
#             A[p, q] = total
#     return A

def TDCT(D, x_eval, y_eval):
    #T = sum(d * c_j * c_k * cos * cos)
    N, M = D.shape
    
    C_j = c(N)
    C_k = c(M)
    # D_scaled[j, k] = d_jk * c_j * c_k
    D_scaled = D * np.outer(C_j, C_k)
    
    cos_x = np.cos(np.outer(np.arange(N), x_eval)) # N x P
    cos_y = np.cos(np.outer(np.arange(M), y_eval)) # M x Q
    
    # A = cos_x.T @ D_scaled @ cos_y
    return cos_x.T @ D_scaled @ cos_y

def f(x, y):
    f_points = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            f_points[i, j] = (np.cos(2 * x[i]) + np.cos(3 * y[j]))
    return f_points


def f2(x, y):
    f_points = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            f_points[i, j] = (x[i] - (np.pi / 2)) ** 2 + (y[j] - (np.pi / 2)) ** 2
    return f_points


def Auswert(P):
    xp = []
    yq = []
    h = np.pi / P
    for p in range(1, P):
        xp.append(p * h)
        yq.append(p * h)
    return xp, yq


def fehler(xp, yq):
    e = []
    for N in range(1, 41):
        xj, yk = Stützstellen(N, N)
        F = f2(xj, yk)
        D = DCT(F)
        val = np.max(np.abs(f2(xp, yq) - TDCT(D, xp, yq)))
        e.append(val)
    return e


# def JPEG(F):
#     block_size = 8
#     n_blocks_y = F.shape[0] // block_size
#     n_blocks_x = F.shape[1] // block_size

#     A = np.zeros_like(F)
#     r_nonzero = 0

#     sigma = np.array([
#         [10, 15, 25, 37, 51, 66, 82, 100],
#         [15, 19, 28, 39, 52, 67, 83, 101],
#         [25, 28, 35, 45, 58, 72, 88, 105],
#         [37, 39, 45, 54, 66, 79, 94, 111],
#         [51, 52, 58, 66, 76, 89, 103, 119],
#         [66, 67, 72, 79, 89, 101, 114, 130],
#         [82, 83, 88, 94, 103, 114, 127, 142],
#         [100, 101, 105, 111, 119, 130, 142, 156]
#     ])

#     x = np.array([(2 * j + 1) / (2 * block_size) * np.pi for j in range(block_size)])  # Spalten (x)
#     y = np.array([(2 * k + 1) / (2 * block_size) * np.pi for k in range(block_size)])  # Zeilen (y)

#     for i in range(n_blocks_y):
#         for j in range(n_blocks_x):
#             block = F[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] - 128

#             D = DCT(block, x, y)
#             Dq = np.round(D / sigma)
#             r_nonzero += np.count_nonzero(Dq)

#             Dq_star = Dq * sigma

#             A_block = TDCT(Dq_star, x, y)
#             A[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = A_block

#     A = np.round(A + 128)
#     A = np.clip(A, 0, 255)

#     total_pixels = F.shape[0] * F.shape[1]
#     r = r_nonzero / total_pixels
#     print(f"Anteil der nicht Null-Werte: {r:.4}")
#     return A

def JPEG(F):
    N_total, M_total = F.shape
    A = np.zeros_like(F)
    nonzero_count = 0
    
    sigma = np.array([
        [10, 15, 25, 37, 51, 66, 82, 100],
        [15, 19, 28, 39, 52, 67, 83, 101],
        [25, 28, 35, 45, 58, 72, 88, 105],
        [37, 39, 45, 54, 66, 79, 94, 111],
        [51, 52, 58, 66, 76, 89, 103, 119],
        [66, 67, 72, 79, 89, 101, 114, 130],
        [82, 83, 88, 94, 103, 114, 127, 142],
        [100, 101, 105, 111, 119, 130, 142, 156]
    ])
    for i in range(0, N_total, 8):
        for j in range(0, M_total, 8):
            block = F[i:i+8, j:j+8]
            

            D = DCT(block)
            

            D_tilde = np.round(D / sigma)
            nonzero_count += np.count_nonzero(D_tilde)
            
            D_star = D_tilde * sigma
            
            x_nodes, y_nodes = Stützstellen(8, 8)
            reconstructed_block = TDCT(D_star, x_nodes, y_nodes)
            
            A[i:i+8, j:j+8] = reconstructed_block
            
    r = 1 - (nonzero_count / (N_total * M_total))
    return A, r

# test for 1.a
N = 6
M = 5
x_j, y_k = Stützstellen(N, M)
F = f(x_j, y_k)
D = DCT(F)
A = TDCT(D, x_j, y_k)
print(F)
print(A)

P = 100
xp, yq = Auswert(P)
errors = fehler(xp, yq)

plt.figure()
plt.semilogy(range(1, 41), errors, marker='o')
plt.xlabel('N (Anzahl der Stützstellen)')
plt.ylabel('Maximalfehler')
plt.title('Fehler zwischen f2 und TDCT-Approximation')
plt.grid(True)

# test for 2
img = Image.open('lena.jpg').convert('L').resize((256, 256))
img_array = np.array(img).astype(np.float64)

h, w = img_array.shape 
img_array = img_array[:h - h % 8, :w - w % 8]  # Ensure dimensions are multiples of 8
reconstructed_image, rate = JPEG(img_array)
print(f"JPEG rate r = {rate:.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title("JPEG rekonstruiert")
plt.axis("off")

plt.tight_layout()
plt.show()