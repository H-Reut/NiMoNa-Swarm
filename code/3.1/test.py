import numpy as np

α = 1.0
β = 0.5
k = 3
M = np.random.rand(k,2,2)
A = M.copy()
B = M.copy()

N = np.array([[1,2],[3,4]])
X = N
print(N)
X[0,0] = 10
print(N)
print(X)

print(1.0 / np.inf)

def morse4(x_t):
    result = x_t[:,1,:].copy()
    for i in range(k):
        result[i] *= np.linalg.norm(x_t[i,1,:], axis=-1)
    return x_t[:,1,:]


def morse5(x_t):
    result = x_t[:,1,:]#.copy()
    for i in range(k):
        result[i] *= np.linalg.norm(x_t[i,1,:], axis=-1)
    return x_t[:,1,:]

print(morse4(A))
print("\n")
print(morse5(B))

