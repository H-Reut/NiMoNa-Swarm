import numpy as np
import matplotlib.pyplot as plt
import jsonpickle
from visualization import visualize_2d
from visualization_ffmpeg import visualize_2d_ffmpeg
import visualization_ffmpeg_multi
import time
import scipy.spatial.distance as sciDist
from scipy.optimize import curve_fit
import runtimes

img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'


def rk4(t, y, Δt, f):
    k1 = f(t, y)
    k2 = f(t+(Δt/2), y + (Δt*k1)/2)
    k3 = f(t+(Δt/2), y + (Δt*k2)/2)
    k4 = f(t+Δt, y + Δt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6


def axisfinder(p):
    # format: xmin, xmax, ymin, ymax, zmin, zmax
    def model(p):
        return np.vstack((np.min(np.min(p, 1), 0), np.max(np.max(p, 1), 0))).reshape(6, order='F')

    def greatest(ax1, ax2):
        return np.array([min(ax1[0], ax2[0]),
                         max(ax1[1], ax2[1]),
                         min(ax1[2], ax2[2]),
                         max(ax1[3], ax2[3]),
                         min(ax1[4], ax2[4]),
                         max(ax1[5], ax2[5])])

    def smallest(ax1, ax2):
        return np.array([max(ax1[0], ax2[0]),
                         min(ax1[1], ax2[1]),
                         max(ax1[2], ax2[2]),
                         min(ax1[3], ax2[3]),
                         max(ax1[4], ax2[4]),
                         min(ax1[5], ax2[5])])

    def centerMin(ax):
        return np.array([-np.min(np.abs(ax[ :2])),
                          np.min(np.abs(ax[ :2])),
                         -np.min(np.abs(ax[2:4])),
                          np.min(np.abs(ax[2:4])),
                         -np.min(np.abs(ax[4: ])),
                          np.min(np.abs(ax[4: ]))])

    def centerMax(ax):
        return np.array([-np.max(np.abs(ax[ :2])),
                          np.max(np.abs(ax[ :2])),
                         -np.max(np.abs(ax[2:4])),
                          np.max(np.abs(ax[2:4])),
                         -np.max(np.abs(ax[4: ])),
                          np.max(np.abs(ax[4: ]))])

    averages = np.average(p, axis=1, keepdims=True)
    avg_final = averages[-1]
    avg_start = averages[ 0]
    print("\n\n\n", avg_start, avg_final)
    p_translated = p - averages
    T1 = n_timesteps//T
    model_early = model(p_translated[:T1])
    model_mid = model(p_translated[T1:-T1])
    model_late = model(p_translated[-T1:])
    default_ax  = np.array([-1., 1., -1., 1., -1., 1.])
    unit_ax  = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
    outer_ax = 2*np.array([-100.,100.,-100.,100.,-100.,100.])
    #ax = greatest(model_late, greatest(centerMin(model_mid), centerMin(model_early)))
    #return greatest(smallest(outer_ax, ax), unit_ax)
    ax_trans = smallest(outer_ax, centerMax(model_late) )
    return np.array([[ax_trans[0]+averages[t,:,0], ax_trans[1]+averages[t,:,0], ax_trans[2]+averages[t,:,1], ax_trans[3]+averages[t,:,1], ax_trans[4]+averages[t,:,2], ax_trans[5]+averages[t,:,2]] for t in range(p.shape[0])])
    # operands could not be broadcast together with shapes (6,) (4001,1,3)


'''def morse7(t, x_t):
    result = (α - β * np.linalg.norm(x_t[...,:,1,:])**2) * x_t[...,:,1,:]
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[...,i,0,:]-x_t[...,j,0,:]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[...,:,1,:], result), axis=1)'''

def U_morse(pos, C_r, l_r, C_a=1.0, l_a=1.0):
    '''
        returns a numpy array containing the total morse potential U
    '''
    print(pos.shape)
    k_timesteps = pos.shape[-3]
    N_agents    = pos.shape[-2]
    assert   2 == pos.shape[-1]
    result = np.zeros((k_timesteps))
    for t in range(k_timesteps):
        result[t] = (C_a - C_r)*N_agents
        print(f'calculating U timestep\t{str(t).rjust(len(str(k_timesteps)))} / {k_timesteps}\t({t/k_timesteps:.0%})', end="\r")
        for i in range(N_agents):
            norm_diff_ij = np.linalg.norm(pos[t,i] - pos[t,:], axis=-1)
            result[t]   +=   np.sum(C_r * np.exp(- norm_diff_ij / l_r)   - C_a * np.exp(- norm_diff_ij / l_a))
    return result

def morseBogus(t, x_t):
    result = np.zeros((k_agents, 2))
    for i in range(k_agents):
        result[i] = (a - b * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k_agents):
            if i != j:
                GradU_r = -(x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0])/l_r * np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_r)
                GradU_a = -(x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0])/l_a * np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_a)
                result[i] -= C_r * GradU_r - C_a * GradU_a
                #print((C_r * GradU_r - C_a * GradU_a).shape)
                #print(result[i])
    return np.stack((x_t[:,1,:], result), axis=1)

def morse2(t, x_t):
    result = np.zeros((k_agents, 2))
    for i in range(k_agents):
        result[i] = (a - b * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k_agents):
            if i != j:
                GradU_r = np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_r)
                GradU_a = np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_a)
                result[i] += (x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0]) * (C_r/l_r * GradU_r - C_a/l_a * GradU_a)
                #print((C_r * GradU_r - C_a * GradU_a).shape)
                #print(result[i])
    return np.stack((x_t[:,1,:], result), axis=1)

def morse3(t, x_t):
    result = np.zeros((k_agents, 2))
    for i in range(k_agents):
        result[i] = (a - b * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k_agents):
            if j != i:
                GradU_r = np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_r)
                GradU_a = np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_a)
                result[i] += (x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0]) * (C_r/l_r * GradU_r - C_a/l_a * GradU_a)
                #print((C_r * GradU_r - C_a * GradU_a).shape)
                #print(result[i])
    return np.stack((x_t[:,1,:], result), axis=1)

def morseNaive(t, x_t):
    result = np.zeros((k_agents, 2))
    for i in range(k_agents):
        result[i] = (a - b*np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k_agents):
            if j!=i:
                result[i] += (
                   (x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0])
                  *( C_r/l_r * np.exp(-np.linalg.norm(x_t[i,0]-x_t[j,0])/l_r) 
                    -C_a/l_a * np.exp(-np.linalg.norm(x_t[i,0]-x_t[j,0])/l_a) )
                )
    return np.stack((x_t[:,1,:], result), axis=1)

def morse2loops(t, x_t):
    result = np.zeros((k_agents, 2))
    for i in range(k_agents):
        result[i] = (a - b*np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k_agents):
            if j!=i:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += (diff/norm
                  * ( C_r/l_r * np.exp(-norm/l_r) 
                    - C_a/l_a * np.exp(-norm/l_a) )
                )
    return np.stack((x_t[:,1,:], result), axis=1)

def morse1loop(t, x_t):
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    for i in range(k_agents):
        # Movements caused by other agents.
        diffs = x_t[i,0]-x_t[:,0] # for i-th entry we have: diffs[i]=0.0
        norms = np.linalg.norm(diffs, axis=-1) # for i-th entry we have norms[i]=0.0
        norms[i] = np.inf
        factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a)) / norms # for i-th entry we have: factors[i]=0.0
        resultVel[i] += factors @ diffs # The effect of agent i on itself is ignored because (factors@diffs)[i]=[0.0, 0.0]
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morseSP1(t, x_t):
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1)) #np.subtract.outer(x_t[:,0], np.transpose(x_t[:,0], axes=(0,1)))
    #print(diffs.shape)
    #print(diffs)
    norms = np.linalg.norm(diffs, axis=-1)
    #print(norms.shape)
    #print(norms)
    np.fill_diagonal(norms, np.inf)
    #print(norms)
    factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a))
    #print(factors)
    #print(norms)
    factors /= norms
    #print(factors)
    #print(factors.shape)
    #print(diffs.shape)
    #print((factors @ diffs).shape)
    #print(resultVel.shape)
    #print(np.diagonal(factors @ diffs).shape)
    #print((np.transpose(factors)[:,:,np.newaxis] * diffs).shape)
    #resultVel += np.transpose(np.diagonal(factors @ diffs))#[:,np.newaxis]
    resultVel += (np.transpose(factors)[:,:,np.newaxis] * diffs).sum(1)
    #print(resultVel)
    #assert False
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morseSP2(t, x_t):
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1)) #np.subtract.outer(x_t[:,0], np.transpose(x_t[:,0], axes=(0,1)))
    norms = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(norms, np.inf)
    factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a))
    factors /= norms
    resultVel += (factors[:,:,np.newaxis] * diffs).sum(1)
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morse0loops(t, x_t):
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], 
                                         axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1)) #np.subtract.outer(x_t[:,0], np.transpose(x_t[:,0], axes=(0,1)))
    norms = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(norms, np.inf)
    factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a)) / norms
    resultVel += (factors[:,:,np.newaxis] * diffs).sum(1)
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morseSP4(t, x_t):
    assert C_a == 1.0 and l_a == 1.0
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1))
    norms = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(norms, np.inf)
    factors = (C_r/l_r * np.exp(-norms/l_r) - np.exp(-norms)) / norms
    resultVel += (np.transpose(factors)[:,:,np.newaxis] * diffs).sum(1)
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morseSP5(t, x_t):
    # General movements:
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1))
    norms = sciDist.pdist(x_t[:,0], metric='euclidean') #creates 1d distance vector
    #np.fill_diagonal(norms, np.inf)
    factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a)) / norms
    factors = sciDist.squareform(factors) # returns the factor to 2d
    #print(factors.shape, np.transpose(factors).shape, diffs.shape)
    #print(factors)
    #assert False
    #resultVel += (np.transpose(factors)[:,:,np.newaxis] * diffs).sum(1)
    resultVel += (factors[:,:,np.newaxis] * diffs).sum(1)
    return np.stack((x_t[:,1,:], resultVel), axis=1)

def morseScipy(t, x_t):
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1))
    norms = sciDist.pdist(x_t[:,0], metric='euclidean') #creates 1d distance vector
    factors = (C_r/l_r * np.exp(-norms/l_r) 
              -C_a/l_a * np.exp(-norms/l_a)) / norms
    resultVel += (sciDist.squareform(factors)[:,:,np.newaxis] 
                  * diffs).sum(axis=1) # returns the factor to 2d
    return np.stack((x_t[:,1,:], resultVel), axis=1)





#n = 100
#T = 10

#n=30
#T=3

################ TODO: RUN! ##################

#              [ α ,  β , C_a, C_r, l_a, l_r, name,             k,    n,  T]
'''clumps =       [1. ,  .5, 1. ,  .6, 1. ,  .5, " Clumps",        40, 300, 60]#3000, 60]
ringclumping = [1. ,  .5, 1. ,  .6, 1. , 1.2, " Ring clumping", 40, 300, 60]#3000, 60]
rings =        [1. ,  .5, 1. ,  .5, 1. ,  .5, " Rings",         40, 300, 60]#3000, 60]
catastrophic = [1.6,  .5,  .5, 1. , 2. ,  .5, " chaotic",       40,  50, 10]# 500, 10]
allparams = [clumps, ringclumping, rings, catastrophic]'''


# 1 value
#Cs = np.arange(0.8, .9, 1.)
#ls = np.arange(0.8, .9, 1.)

# 2x2
#Cs = np.arange(0.6, 1.25, 0.6)
#ls = np.arange(0.6, 1.25, 0.6)

# 3x2
#Cs = np.arange(0.6, 1.1, 0.4)
#ls = np.arange(0.6, 1.5, 0.4)

# 4x3
#Cs = np.arange(0.6, 1.7, 0.4)
#ls = np.arange(0.6, 1.9, 0.4)

# 4x4
#Cs = np.arange(0.6, 1.2, 0.2)
#ls = np.arange(0.6, 1.2, 0.2)

# 5x3
#Cs = np.arange(0.6, 1.9, 0.4)
#ls = np.arange(0.6, 2.3, 0.4)

# 10x8
#Cs = np.arange(0.6, 2.3, 0.2)
#ls = np.arange(0.6, 2.3, 0.2)
#Cs = np.arange(1.0, 2.3, 0.2)

# 8x5
#Cs = np.arange(0.5, 2.0, 0.2)
#ls = np.arange(1.3, 0.49, -0.2)

# 4x3
#Cs = np.arange(0.5, 2.0, 0.4)
#ls = np.arange(1.3, 0.49,-0.4)

# 3x2
Cs = np.arange(0.5, 1.35, 0.4)
ls = np.arange(1.2, 0.79,-0.4)

print(Cs, ls)


# Model parameters
'''k_agents = 250'''
n_timesteps = 1000#250#0#10000
T = 5#0#20#0#50
fps = 30#60
dim = 2

allparams = [[[1.0, 0.5, 1.0, C_r , 1.0, l_r] for C_r in Cs] for l_r in ls]
plottitles = [[f'C={C_r:.1f}, l={l_r:.1f}' for C_r in Cs] for l_r in ls]
#print(len(allparams), len(allparams[0]))

time_total_start = time.time()

startfunc = time.time()
agentnumbers_1 = [20,40,      100                            ]
agentnumbers_2 = [20,40,60,   100,        160                ]
agentnumbers_3 = [20,40,60,   100,        160,        250    ]
agentnumbers_4 = [20,40,60,80,100,120,140,160,180,200,250,400]
func_name_k = [[morseNaive,  'naive, redundant calcs',  agentnumbers_2],
               [morse2loops, 'improved, 2 for-loops',   agentnumbers_2],
               [morse1loop,  'improved, 1 for-loop',    agentnumbers_3],
               [morse0loops, 'improved, 0 for-loops',   agentnumbers_4],
               [morseScipy,  'scipy.spatial.distance',  agentnumbers_4]]
greatest_agentnumbers = agentnumbers_4

'''#func_name_k = [#[morseNaive,  'unimproved, redundant calcs', agentnumbers_2],
               #[morse2loops, 'improved, 2 for-loops',   agentnumbers_1],
               [morse1loop,  'improved, 1 for-loop',    agentnumbers_1],
               [morse0loops, 'improved, 0 for-loops',       agentnumbers_2],
               [morseScipy,  'scipy.spatial.distance',      agentnumbers_2]]
greatest_agentnumbers = agentnumbers_2'''

times = [[0.0]*len(func_name_k[fi][2]) for fi in range(len(func_name_k))]
'''print(times)
for fi in range(len(func_name_k)):
    f = func_name_k[fi][0]
    f_name = func_name_k[fi][1]
    agentnumbers = func_name_k[fi][2]
    print(f.__name__, f_name)
    for k_index in range(len(agentnumbers)):
        k_agents = agentnumbers[k_index]
        times_Parameters = np.zeros((len(ls), len(Cs)))
        #print(times_Parameters)
        for img_y in range(len(ls)):
            for img_x in range(len(Cs)):
                [a, b, C_a, C_r, l_a, l_r] = allparams[img_y][img_x]
                dt = T/n_timesteps

                #print(f'\timg_y={img_y}, img_x={img_x},\t{allparams[img_y][img_x]}, dt={dt}')

                t_ = np.linspace(0, T, n_timesteps+1)
                x = np.zeros((n_timesteps+1, k_agents, 2, dim))  # timesteps; agents; pos,vel; x,y

                # initial condition: spread agents over [0,8]x[0,8] square

                #with open("positions2.json", "r+") as f:
                #    jp_loaded = f.read()
                #    p_0 = np.array(jsonpickle.decode(jp_loaded))
                
                p_0 = -0.5*np.ones((k_agents, dim)) + np.random.rand(k_agents, dim)# *1 
                x[0,:,0,:] = p_0.copy()
                #x[0,:,1,:] = v_0



                # solving
                #print("solving PDE")
                time_solve_start = time.time()
                for i in range(n_timesteps):
                    #print(f'solving PDE. timestep\t{str(i).rjust(len(str(n_timesteps)))} / {n_timesteps}\t({i/n_timesteps:.0%})', end="\r")
                    #print()
                    x[i+1,:,:,:] = x[i,:,:,:] + dt*rk4(t_[i], x[i,:,:,:], dt, f)
                time_solve_end = time.time()
                time_solve_duration = time_solve_end - time_solve_start
                #print(f'\t solving ({k_agents}ag, {plottitles[img_y][img_x]}) with {f.__name__}("{f_name}") \ttook {time_solve_duration:.2f} seconds.')
                times_Parameters[img_y,img_x] = time_solve_duration
                #assert False

                
                if np.isfinite(x[-1,...]).all():
                    pass
                else:
                    print(f"invalid values encountered: {x[-1,0,0,:]}")

                #if i != 0:
                #    print("maximum abs difference\n",np.max(np.abs(xs-x0)))
        #print(f'agents: {k_agents}    time_Parameters:\n{times_Parameters}')
        times[fi][k_index] = np.average(times_Parameters).item()
        print(f'\t{f.__name__}("{f_name}"), {k_agents} ag,   average time: {times[fi][k_index]} sec')'''
time_total_end = time.time()
print(f"total time {time_total_end-time_total_start:.1f} seconds")
print(f"\n\nfunc_name_k=\n{func_name_k}\n\ntimes=\n{times}")

times = runtimes.times

polys = [None] * len(func_name_k)
for fi in range(len(func_name_k)):
    coeff = np.polyfit(func_name_k[fi][2], times[fi], deg=2)
    print(func_name_k[fi][1], coeff)
    x_poly = np.arange(func_name_k[fi][2][0], func_name_k[fi][2][-1], 1.0)
    polys[fi] = np.polyval(np.poly1d(coeff), x_poly)

### SCIPY
#def a function
def f(x, a, b):
    return a*x**2 + b

'''for fi in range(len(func_name_k)):
    xs = np.array(func_name_k[fi][2])
    ys = np.array(times[fi])
    popt, pcov = curve_fit(f, xs, ys)
    print(func_name_k[fi][1], *popt)
    x_poly = np.arange(func_name_k[fi][2][0], func_name_k[fi][2][-1], 1.0)
    polys[fi] = f(x_poly, *popt)'''


for plotter in [plt.plot, plt.semilogy, plt.loglog]:
    plt.xticks(greatest_agentnumbers)
    plt.ylabel('runtime (sec)')
    plt.xlabel('number of agents')
    for fi in range(len(func_name_k)):
        x_poly = np.arange(func_name_k[fi][2][0], func_name_k[fi][2][-1], 1.0)
        plt.plot(x_poly, polys[fi], '-')
    ax = plt.gca()
    ax.set_prop_cycle(None)
    labels = ax.get_xticklabels()
    labels[4] = ''
    labels[6] = ''
    labels[8] = ''
    ax.set_xticklabels(labels)
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    #print(plotter.__name__, xlim, ylim)
    for fi in range(len(func_name_k)):
        plotter(func_name_k[fi][2], times[fi], 'o', label=func_name_k[fi][1])
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.show()