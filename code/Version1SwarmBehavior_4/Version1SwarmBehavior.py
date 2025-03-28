import numpy as np
import matplotlib.pyplot as plt
import time

from visualization_ffmpeg import visualize_2d_ffmpeg
from visualization import visualize_2d

img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'

#import visualization

def euler(t, y, dt, f):
    return f(t,y)

# Exercise 1: Heun method:
def heun(t, y, dt, f):
    k1 = f(t, y)
    k2 = f(t + dt, y + dt*k1)
    return (k1+k2)/2

# Exercise 2: Runge-Kutta 4 method:
def rk4(t, y, dt, f): 
    k1= f(t, y)
    k2= f(t+(dt/2), y+ (dt*k1)/2)
    k3= f(t+(dt/2), y+ (dt*k2)/2)
    k4= f(t+dt, y+ dt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6


#p0 = np.array([[1. ,0.],
#              [2. ,0.],
#              [1.5,2.],
#              [2. ,1.],
#              [4. ,2.]])

#p0 = np.array([[-2., 0.], [2., 0.]])


k=16
#k=p0.shape[0]
T=50#250
dt=0.01#0.005
n = int(T / dt)
assert n*dt == T

print(f"Model Parameters:\n\t  k = {k}\n\t  T = {T}\n\t dt = {dt}\n\t  n = {n}")

fps = max(int(0.1/dt),60)
duration_m, duration_s = divmod(n/fps, 60)
print(f'Animation Parameters:\n\tfps = {fps}\n\tdur = {duration_m:2.0f}:{duration_s:02.0f}')


# Repulsion function sigma:
# radii for repulsion function. Repulsion: from r[0] to r[1], orientation: from r[1] to r[2], attraction: from r[1] to r[2]
r = [0,1.5,4,6]
#r = [0,2,4,8] 
factor = [1,1]
#factor = [0.5,2] # scaling factors for repulsion and attraction
def sigma_swarm(x):
    return np.piecewise(x, [(r[0]<x)&(x<=r[1]), 
                            (r[2]<x)&(x<r[3])], 
                        [lambda x: factor[0]* (1/( (x-r[0])/(r[1]-r[0]) )+( (x-r[0])/(r[1]-r[0]) )-2), 
                         lambda x: factor[1]* (-( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**4 + 2*( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**2 - 1)])


def sigma_linear(x): 
    return 4-x

def sigma_gravity(x): 
    return np.where(x==0, 0,-1/x**2)

D   = 1
a   = 1
r_e = 1
def sigma_morse(x):
    return D*(1 - np.exp(-a*(x-r_e)) )**2-D

sigma = sigma_morse

# plotting repulsion function:
show_repulsion_function = False
if show_repulsion_function:
    t = np.arange(0, r[3]+1.5, 0.01)
    s = sigma(t)
    #plt.axis((min(-0.5,r[0]-0.5), r[3]+1.5, -factor[1]-0.5, 5))
    fig, ax = plt.subplots()
    #fig.set_size_inches(19.20,10.80)
    ax.set(xlim=(min(-0.5,r[0]-0.5), r[3]+1.5), ylim=(-factor[1]-0.5, 5))  # constant reference frame
    #ax.set(xlim=(min(-0.5,r[0]-0.5), r[3]+1.5), ylim=(-5, 0.5)) 
    ax.set_aspect(1)  # spacing the same for x and y direction
    plt.axhline(y=0, color='k', linewidth=.5)
    plt.axvline(x=0, color='k', linewidth=.5)
    plt.axvspan(0, r[0], facecolor='dimgray', alpha=0.2)
    plt.axvspan(r[0], r[1], facecolor='red', alpha=0.2)
    plt.axvspan(r[1], r[2], facecolor='yellow', alpha=0.2)
    plt.axvspan(r[2], r[3], facecolor='green', alpha=0.2)
    for radius in r:
        plt.axvline(radius, color='gray', linestyle='--', linewidth=.5)
    plt.hlines(y = -factor[1], xmin=r[2], xmax=r[3], color='gray', linestyle='--', linewidth=.5)
    plt.plot(t,s)
    #fig.set_size_inches(19.20,10.80)
    #plt.savefig('plot.svg',bbox_inches='tight')
    plt.savefig('plot.png',bbox_inches='tight')
    plt.show()


def interaction(t, p_t):
    w=np.zeros((k, 2))
    for i in range(k):
        for j in range(k):
            if i!=j: 
                w[i]+= sigma(np.linalg.norm(p_t[i] - p_t[j])) * ((p_t[i] - p_t[j])) / np.linalg.norm(p_t[i] - p_t[j])     #f_ij = sigma(||pi- pj||) (pi-pj)
    return w



def generalmov(t, p_t):
    return np.array([-p_t[i]/np.linalg.norm(p_t[i])**3 if np.linalg.norm(p_t[i]) != 0 else np.array([0., 0.]) for i in range(k)])

#def generalmov(t, p_t):
#    return np.zeros((k, 2))
    
def behavior(t, p_t):
    return interaction(t, p_t) + generalmov(t, p_t)


p=np.zeros((n+1, k, 2))  #k is agent, n is time
v=np.zeros((n+1, k, 2))
t=np.linspace(0, T, n+1)

p[0] = np.random.rand(k, 2) #spreads agents across unit square

p[0]= 2*np.random.rand(k, 2)-np.ones((k,2)) # spreads agents across [-1,1]x[-1,1] square
for K in range(k):
    p[0,K]= p[0,K]/np.linalg.norm(p[0,K]) * (1+np.linalg.norm(p[0,K]))
    v[0,K,:]= 0.5*np.array([-p[0,K,1], p[0,K,0]])
v[0]+= 0.1*(2*np.random.rand(k, 2)-np.ones((k,2)))

#p[0,0] = [1,0]
#v[0,0] = [0,4]
#c = ['white'] + 15*['cyan']

#p[0,:,0] *= 2 #stretch p[0] in y-direction
#p[1]= p[0] + np.array([[0.01 * l, 0.01*l] for l in range(-5, 5)])


#assert False
for i in range(1,n+1):
    print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
    m = i
    # implement in one vector
    p[i] = p[i-1] + dt * v[i-1] + dt * rk4(t[i-1], p[i-1], dt, lambda s, p_t: (t[i]-s) * behavior(s, p_t) )
    v[i] = v[i-1]               + dt * rk4(t[i-1], p[i-1], dt, behavior)
print()

#print(p)
model_xmin_xmax_ymin_ymax = np.vstack((np.min(np.min(p, 1), 0), np.max(np.max(p, 1), 0))).reshape(4, order='F')
fixed_xmin_xmax_ymin_ymax  = 4 * np.array([-1., 1., -1., 1.])
xmin_xmax_ymin_ymax = np.array([max(model_xmin_xmax_ymin_ymax[0], fixed_xmin_xmax_ymin_ymax[0]),
                                min(model_xmin_xmax_ymin_ymax[1], fixed_xmin_xmax_ymin_ymax[1]),
                                max(model_xmin_xmax_ymin_ymax[2], fixed_xmin_xmax_ymin_ymax[2]),
                                min(model_xmin_xmax_ymin_ymax[3], fixed_xmin_xmax_ymin_ymax[3])])
#xmin_xmax_ymin_ymax += [-.1, .1, -.1, .1]
print(f"Axis:\n\tmodel values:\t{model_xmin_xmax_ymin_ymax}\n\tmaximum bounds:\t{fixed_xmin_xmax_ymin_ymax}\n\tfinal axis:\t{xmin_xmax_ymin_ymax}")

#assert False
visualize_2d_ffmpeg(p, v,    *fixed_xmin_xmax_ymin_ymax, True, T_max=T, dt=dt, inverse_arrow_scale_factor=10, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, draw_rectangle=[-1,1,-1,1], origin_color=None)
visualize_2d_ffmpeg(p, None, *fixed_xmin_xmax_ymin_ymax, True, T_max=T, dt=dt, inverse_arrow_scale_factor=10, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, draw_rectangle=[-1,1,-1,1], origin_color=None)
# visualize_2d(p, None, *xmin_xmax_ymin_ymax, True)

