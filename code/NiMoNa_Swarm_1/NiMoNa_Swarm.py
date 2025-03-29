import numpy as np
import matplotlib.pyplot as plt
show_plots = False # should every plot be shown?
show_σ = True # should repulsion function σ be shown?
img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'


# Repulsion function σ:
# radii for repulsion function. Repulsion: from r[0] to r[1], orientation: from r[1] to r[2], attraction: from r[1] to r[2]
r = [0.5,2.5,4,8]
r = [0,2,4,8] 
factor = [0.5,2] # scaling factors for repulsion and attraction
def σ(x):
    return np.piecewise(x, [(r[0]<x)&(x<=r[1]), 
                            (r[2]<x)&(x<r[3])], 
                        [lambda x: factor[0]* (1/( (x-r[0])/(r[1]-r[0]) )+( (x-r[0])/(r[1]-r[0]) )-2), 
                         lambda x: factor[1]* (-( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**4 + 2*( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**2 - 1)
                         ])

# plotting repulsion function:
if show_σ:
    t = np.arange(0, r[3]+1.5, 0.01)
    s = σ(t)
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


# plotting:
filename_counter = -1
def plotOrSave():
    global filename_counter
    #global colors
    for i in range(n):
        plt.scatter(*p[i,:],c=colors[i])
    #plt.plot(p[:,0], p[:,1], 'bo')
    plt.axis((-2.5, 7.5, -2.5, 7.5))
    plt.savefig(img_save_path+f'file_{(filename_counter:=filename_counter+1):04}.png', bbox_inches='tight')
    plt.show() if show_plots else plt.close()


p = np.array([[1. ,0.],
              [2. ,0.],
              [1.5,2.],
              [2. ,1.],
              [4. ,2.]])
              #[0. ,6.]])
colors = ['r', 'g', 'b', 'y', 'k', 'gray']
n = p.shape[0]
#print(n)
v = np.zeros(p.shape)
#print(a);print(b)

assert p.shape == v.shape
assert len(colors) >= n


Δt = 0.05
T  = 100
times = np.arange(0,T+Δt,Δt)
K = len(times)
#print("times:\n",times)

plotOrSave()

for k in range(1,len(times)):
    t = times[k]
    if k%10==0: print(f'\tprogress: {k}/{K-1} =\t{k/(K-1):.0%}')
    # update velocity vector:
    for i in range(n):
        f_i = np.zeros(p.shape)
        for j in range(n):
            if i != j:
                d = np.linalg.norm(p[i] - p[j], 2)
                f_i[j] = (p[i] - p[j])/d * σ(d)
            if i == j:
                # global movement
                True
        v[i] += sum(f_i)
        #v[i] = [v[i,0]+0.2, (p[i,1]-1.5)]

    # update positions:
    for i in range(n):
        p[i] += Δt * v[i]

    #print(a)
    plotOrSave()