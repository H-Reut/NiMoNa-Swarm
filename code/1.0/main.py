import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_2d
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


p0 = np.array([[1. ,0.],
              [2. ,0.],
              [1.5,2.],
              [2. ,1.],
              [4. ,2.]])

#p0 = np.array([[-2., 0.], [2., 0.]])


k=5
k=p0.shape[0]
n=200
T=10
dt=T/n



# Repulsion function sigma:
# radii for repulsion function. Repulsion: from r[0] to r[1], orientation: from r[1] to r[2], attraction: from r[1] to r[2]
r = [0.5,2.5,4,8]
r = [0,2,4,8] 
factor = [0.5,2] # scaling factors for repulsion and attraction
def sigma(x):
    return np.piecewise(x, [(r[0]<x)&(x<=r[1]), 
                            (r[2]<x)&(x<r[3])], 
                        [lambda x: factor[0]* (1/( (x-r[0])/(r[1]-r[0]) )+( (x-r[0])/(r[1]-r[0]) )-2), 
                         lambda x: factor[1]* (-( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**4 + 2*( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**2 - 1)])


# def sigma(x): return 4-x

# def sigma(x): return -1/x**2

# plotting repulsion function:
t = np.arange(0, r[3]+1.5, 0.01)
s = sigma(t)
plt.axis((min(-0.5,r[0]-0.5), r[3]+1.5, -factor[1]-0.5, 5))
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
plt.show()


def interaction(t, p_t):
    w=np.zeros((k, 2))
    for i in range(k):
        for j in range(k):
            if i!=j: 
                w[i]+= sigma(np.linalg.norm(p_t[i] - p_t[j]))*((p_t[i] - p_t[j]))#/np.linalg.norm(p_t[i] - p_t[j]))     #f_ij = sigma(||pi- pj||) (pi-pj)
    return w


global m
m = 0

def generalmov(t, p_t):
    global m
    return p_t-p[m-1]
    
def behavior(t, p_t):
    return interaction(t, p_t) + generalmov(t, p_t)


k = 10
p=np.zeros((n+1, k, 2))  #k is agent, n is time
t=np.linspace(0, T, n+1)

p[0]= np.random.rand(k, 2)
p[1]= p[0] + np.array([[0.01 * l, 0.01*l] for l in range(-5, 5)])


for i in range(1,n):
    m = i
    if i%100 == 0: print(i, n)
    p[i+1] = p[i] + dt * rk4(t[i], p[i], dt, behavior)

#print(p)
xmin_xmax_ymin_ymax = np.vstack((np.min(np.min(p, 1), 0), np.max(np.max(p, 1), 0))).reshape(4, order='F')
#print("\naxis:\t",xmin_xmax_ymin_ymax)

visualize_2d(p, None, *(xmin_xmax_ymin_ymax+[-0.1, 0.1, -0.1, 0.1]), True)

'''for i in range(n):
    plt.axis(xmin_xmax_ymin_ymax) # sets x- and y- axis to minimum and maximum
    plt.scatter(p[i, :, 0], p[i, :, 1])
    plt.show()#'''

