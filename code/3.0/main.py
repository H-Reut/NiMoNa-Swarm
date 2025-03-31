import numpy as np
import matplotlib.pyplot as plt
from visualization import visualize_2d
from visualization_ffmpeg import visualize_2d_ffmpeg

path = 'H:/dump/'

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
#k = 2 #24
#n = 10000
#T = 50
#sigma = sigma_gravity
#p_0 = np.array([[-.25, 0],[.25, 0]])
#v_0 = np.array([[0,-1],[0, 1]])

k = 24
n = 500
T = 5
sigma = sigma_morse
p_0 = np.random.rand(k, 2) * 8 # initial condition: spread agents over [0,8]x[0,8] square # morse/sigma
v_0 = np.zeros((k, 2))


dt = T/n
t_ = np.linspace(0, T, n+1)



def interaction(t, x_t):
    w = np.zeros((k, 2))
    for i in range(k):
        for j in range(k):
            if i != j:
                w[i] += sigma(np.linalg.norm(x_t[i] - x_t[j]))*(x_t[i] - x_t[j])/np.linalg.norm(x_t[i] - x_t[j])      #f_ij = sigma(||pi- pj||) (pi-pj)
    return w

def generalmov(t, x_t):
    return np.zeros((k, 2))

def behavior(t, x_t):
    return interaction(t, x_t) + generalmov(t, x_t)

def yNaive(t, x_t):
    return np.stack((x_t[:,1,:], behavior(t, x_t[:,0,:])), axis=1)




xNaive = np.zeros((n+1, k, 2, 2))  # timesteps; agents; pos,vel; x,y
xNaive[0,:,0,:] = p_0
xNaive[0,:,1,:] = v_0
# solving Naive
for i in range(n):
    print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
    xNaive[i+1] = xNaive[i] + dt*rk4(t_[i], xNaive[i], dt, yNaive)
    #plt.scatter(p[i, :, 0], p[i, :, 1])
    #plt.show()
print()



def yTriangle_GeneralForm(t, x_t, currtime):
    func = behavior(t, x_t[:,0,:])
    return np.stack(((currtime-t)*func, func), axis=1)

xTriangle = np.zeros((n+1, k, 2, 2))  # timesteps; agents; pos,vel; x,y
xTriangle[0,:,0,:] = p_0
xTriangle[0,:,1,:] = v_0
# solving Triangle
for i in range(n):
    print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
    def yTriangle(t, x_t):
        return yTriangle_GeneralForm(t, x_t, t_[i+1])
    xTriangle[i+1] = xTriangle[i] + dt*np.stack((xTriangle[i,:,1,:],np.zeros((k, 2))), axis=1) + dt*rk4(t_[i], xTriangle[i], dt, yTriangle)
    #plt.scatter(p[i, :, 0], p[i, :, 1])
    #plt.show()
print()

#print(x.shape)
#print(x2.shape)
xStacked = np.concatenate((xNaive, xTriangle), axis=1)
#print(xStacked.shape)
colors_for_xStacked = np.array([(['r']*k + ['c']*k) for i in range(n+1)])
slice = 1

# visualizations
#visualize_2d(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, save_anim=True)
#visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=path, fps=10, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)
#visualize_2d_ffmpeg(pos=x2[:,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=path, fps=10, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)
visualize_2d_ffmpeg(pos=xStacked[::slice,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=path, fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=colors_for_xStacked[::slice])

print("Program finished!")

##TODO#
## Implement proper frame slicing, or array indexing with arrays