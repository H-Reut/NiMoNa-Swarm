from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import jsonpickle

show_plots = False # should every plot be shown?
show_σ = True # should repulsion function σ be shown?
img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'

k=24
n=5000
T=10
Δt=T/n
print(f"Model Parameters:\n\t  k = {k}\n\t  T = {T}\n\t dt = {Δt}\n\t  n = {n}")

def rk4(t, y, dt, f): 
    k1= f(t, y)
    k2= f(t+(dt/2), y+ (dt*k1)/2)
    k3= f(t+(dt/2), y+ (dt*k2)/2)
    k4= f(t+dt, y+ dt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6

# Repulsion function σ:
# radii for repulsion function. Repulsion: from r[0] to r[1], orientation: from r[1] to r[2], attraction: from r[1] to r[2]
#r = [0.5,2.5,4,8]
r = [0,1,1,5] 
factor = [1,1] # scaling factors for repulsion and attraction
def sigma(x):
    return np.piecewise(x, [(r[0]<x)&(x<=r[1]), 
                            (r[2]<x)&(x<r[3])], 
                        [lambda x: factor[0]* (1/( (x-r[0])/(r[1]-r[0]) )+( (x-r[0])/(r[1]-r[0]) )-2), 
                         lambda x: factor[1]* (-( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**4 + 2*( (x-(r[2]+r[3])/2)*2/(r[3]-r[2]) )**2 - 1)
                         ])

def sigma_linear(x): 
    return 4-x

def sigma_gravity(x): 
    return np.where(x==0, 0,-1/x**2)

D   = 2
a   = 3
r_e = 0.5
def sigma_morse(x):
    return D*(1 - np.exp(-a*(x-r_e)) )**2-D

sigma = sigma_morse

# plotting repulsion function:
show_repulsion_function = False
if show_repulsion_function:
    t = np.arange(0, 10, 0.01)
    s = sigma(t)
    #plt.axis((min(-0.5,r[0]-0.5), r[3]+1.5, -factor[1]-0.5, 5))
    fig, ax = plt.subplots()
    #fig.set_size_inches(19.20,10.80)
    ax.set(xlim=(-0.0, 5.))#, ylim=(-2, 4))  # constant reference frame
    ax.set_aspect(1)  # spacing the same for x and y direction
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    plt.plot(t,s)
    #fig.set_size_inches(19.20,10.80)
    #plt.savefig('plot.png',bbox_inches='tight')
    plt.show()





def interaction(t, p_t):
    w=np.zeros((k, 2))
    for i in range(k):
        for j in range(k):
            if i!=j: 
                w[i]+= sigma(np.linalg.norm(p_t[i] - p_t[j]))*(p_t[i] - p_t[j])/np.linalg.norm(p_t[i] - p_t[j])
    #print(w)
    return w


def generalmov(t, p_t):
    return np.zeros((k, 2))
    
def behavior(t, p_t):
    return interaction(t, p_t) + generalmov(t, p_t)




# plotting:
filename_counter = -1
def plotOrSave(t, axis):
    global filename_counter
    #global colors
    #for i in range(k):
    #    plt.scatter(*p[t,i,:],c=colors[i])
    #plt.plot(p[:,0], p[:,1], 'bo')
    fig, ax = plt.subplots()
    ax.set(xlim=(axis[0], axis[1]), ylim=(axis[2], axis[3]))  # constant reference frame
    ax.set_aspect(1)
    plt.scatter(p[t,:,0], p[t,:,1], c=np.arange(k), cmap='tab10')
    if (draw_rectangle != [0.,0.,0.,0.]).all:
        plt.vlines(x=draw_rectangle[0:2], 
                    ymin=draw_rectangle[2], ymax=draw_rectangle[3], 
                    color='gray', linestyle='--', linewidth=1)
        plt.hlines(y=draw_rectangle[2:4], 
                    xmin=draw_rectangle[0], xmax=draw_rectangle[1], 
                    color='gray', linestyle='--', linewidth=1)
    plt.savefig(img_save_path+f'file_{(filename_counter:=filename_counter+1):04}.png', bbox_inches='tight')
    plt.show() if show_plots else plt.close()

    


p=np.zeros((n+1, k, 2))  #k is agent, n is time
t=np.linspace(0, T, n+1)

with open("example.json", "r+") as f:
    written_instance = f.read()
    p[0] = jsonpickle.decode(written_instance)

'''with open("example2.json", "w+") as f:
    array = p[0].tolist()
    exp = 8
    for i in range(len(array)):
        array[i] = [int(array[i][0]*2**exp)/(2**exp), int(array[i][1]*2**exp)/(2**exp)]
    encoded_instance = jsonpickle.encode(array)
    f.write(encoded_instance)


with open("example2.json", "r+") as f:
    written_instance = f.read()
    array = jsonpickle.decode(written_instance)
    p0_decoded = np.array(array)
    print("success")
#print(p0_decoded)
p[0] = p0_decoded'''

p[0] *= 8


#p[0]= np.random.rand(k, 2)
draw_rectangle = np.array([0.,1.,0.,1.])*0
#print(p[0])

for i in range(n):
    print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
    p[i+1] = p[i] + Δt*rk4(t[i], p[i], Δt, behavior)
    #plt.plot(p[:,0], p[:,1], 'bo')
print()

model_xmin_xmax_ymin_ymax = np.vstack((np.min(np.min(p, 1), 0), np.max(np.max(p, 1), 0))).reshape(4, order='F')
print(model_xmin_xmax_ymin_ymax)
model_xmin_xmax_ymin_ymax += [-.1, .1, -.1, .1]
model_xmin_xmax_ymin_ymax = [-1,1.1,-1,1.1]
model_xmin_xmax_ymin_ymax = [0,8,0,8]

plotOrSave(0, model_xmin_xmax_ymin_ymax)
for i in range(n):
    print(f'generating images\t{str(i).rjust(len(str(i)))} / {n}\t({i/n:.0%})', end="\r")
    plotOrSave(i, model_xmin_xmax_ymin_ymax)
print()






