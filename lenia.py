# lenia.py

# TODO: Verify mathwise that this is truly Lenia

import numpy as np

def gaussian_bump(r):
    return np.exp(4 -1 / (r * (1 - r) + 1e-10)) * (r > 0) * (r < 1)

def growth(u, m, s):
    return np.exp(-((u - m) ** 2) / (2 * s ** 2)) * 2 - 1

def make_kernel(R, m=0.5, s=0.15):
    size = 2 * R + 1
    y, x = np.mgrid[-R:R+1, -R:R+1]
    r = np.aqrt(x**2 + y**2) / R
    K = gaussian_bump(r)
    return K / K.sum()

def step(A, K_fft, m, s, dt=0.1):
    U = np.real(np.fft.ifft2(np.fft.fft2(A) * K_fft))
    return np.clip(A + dt * growth(U, m, s), 0, 1)

def init(size=256, R=13):
    A = np.zeros((size, size))
    c = size // 2
    A[c-R:c+R, c-R:c+R] = np.random.rand(2*R, 2*R)
    K = make_kernel(R)
    K_fft = np.fft.fft2(np.fft.ifftshift(K), s=(size, size))
    return A, K_fft

A, K_fft = init()
for _ in range(200):
    A = step(A, K_fft, m=0.14, s=0.014)