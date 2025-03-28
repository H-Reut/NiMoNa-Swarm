import numpy as np
import matplotlib.pyplot as plt
from visualization import visualize_2d
from visualization_ffmpeg import visualize_2d_ffmpeg

img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'


def rk4(t, y, Δt, f):
    k1 = f(t, y)
    k2 = f(t+(Δt/2), y + (Δt*k1)/2)
    k3 = f(t+(Δt/2), y + (Δt*k2)/2)
    k4 = f(t+Δt, y + Δt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6


# Different repulsion functions:

# radii for swarm function.
# Repulsion:   from r[0] to r[1],
# orientation: from r[1] to r[2],
# attraction:  from r[2] to r[3]
r = [0, 1.5, 4, 6]
# scaling factors for repulsion and attraction
factor = [1, 1]
def sigma_swarm(x):
    return np.piecewise(x, [(r[0] < x) & (x <= r[1]),
                            (r[2] < x) & (x < r[3])],
                        [lambda x: factor[0] * (1/((x-r[0])/(r[1]-r[0]))+((x-r[0])/(r[1]-r[0]))-2),
                         lambda x: factor[1] * (-((x-(r[2]+r[3])/2)*2/(r[3]-r[2]))**4 + 2*((x-(r[2]+r[3])/2)*2/(r[3]-r[2]))**2 - 1)])

def sigma_gravity(x):
    return np.where(x == 0, 0, -1/x**2)

D = 2
a = 3
r_e = 0.5
def sigma_morse(x):
    return D*(1 - np.exp(-a*(x-r_e)))**2-D


# Model parameters
k = 24
n = 200
T = 5
dt = T/n

sigma = sigma_morse


def interaction(t, p_t):
    w = np.zeros((k, 2))
    for i in range(k):
        for j in range(k):
            if i != j:
                w[i] += sigma(np.linalg.norm(p_t[i] - p_t[j]))*(p_t[i] - p_t[j])/np.linalg.norm(p_t[i] - p_t[j])      #f_ij = sigma(||pi- pj||) (pi-pj)
    return w

def generalmov(t, p_t):
    return np.zeros((k, 2))

def behavior(t, p_t):
    return interaction(t, p_t) + generalmov(t, p_t)


p = np.zeros((n+1, k, 2))  # k is agent, n is time
t = np.linspace(0, T, n+1)

p[0] = np.random.rand(k, 2) * 8 # initial condition: spread agents over [0,8]x[0,8] square

# solving
print("Solving ODE...")
for i in range(n):
    p[i+1] = p[i] + dt*rk4(t[i], p[i], dt, behavior)
    #plt.scatter(p[i, :, 0], p[i, :, 1])
    #plt.show()

# visualizations
visualize_2d(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, save_anim=True)
#visualize_2d_ffmpeg(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=img_save_path, fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)

print("Program finished!")
