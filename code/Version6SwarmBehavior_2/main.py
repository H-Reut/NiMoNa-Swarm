import numpy as np
import matplotlib.pyplot as plt
import jsonpickle
from visualization import visualize_2d
#from visualization_ffmpeg import visualize_2d_ffmpeg
import visualization_ffmpeg_multi
import time
import scipy.spatial.distance as sciDist
import os

img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'

np.set_printoptions(linewidth=125)

def rk4(t, y, Δt, f):
    k1 = f(t, y)
    k2 = f(t+(Δt/2), y + (Δt*k1)/2)
    k3 = f(t+(Δt/2), y + (Δt*k2)/2)
    k4 = f(t+Δt, y + Δt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6


def axisfinder(p):
    # format: xmin, xmax, ymin, ymax, zmin, zmax
    # format: np.array([[xmin, ymin, zmin],
    #                   [xmax, ymax, zmax]]) (z is excluded in 2d)
    def model(p):
        return np.vstack((np.min(p, (0,1)),
                          np.max(p, (0,1))))

    def greatest(ax1, ax2):
        return np.vstack((np.minimum(ax1[...,0,:], ax2[...,0,:]), 
                          np.maximum(ax1[...,1,:], ax2[...,1,:])))

    def smallest(ax1, ax2):
        return np.vstack((np.maximum(ax1[...,0,:], ax2[...,0,:]), 
                          np.minimum(ax1[...,1,:], ax2[...,1,:])))

    def square_inwards(ax):
        result = np.min(np.abs(ax), axis=-2)
        return np.vstack((-result, result))

    def square_outwards(ax):
        result = np.max(np.abs(ax), axis=-2)
        return np.vstack((-result, result))

    averages = np.average(p, axis=1, keepdims=True)
    avg_final = averages[-1]
    avg_start = averages[ 0]
    #print(f"\naverages:\n\tstart: {avg_start}\n\tfinal: {avg_final}")
    p_translated = p - averages
    T1 = n_timesteps//T
    model_early = model(p_translated[:T1])
    model_mid = model(p_translated[T1:-T1])
    model_late = model(p_translated[-T1:])
    unit_ax = np.array([[-1., -1., -1.],
                        [ 1.,  1.,  1.]])
    if dim == 2:
        unit_ax = unit_ax[:,:2]
    start_ax = 0.5 * unit_ax
    outer_ax = 200 * unit_ax
    #ax = greatest(model_late, greatest(centerMin(model_mid), centerMin(model_early)))
    #return greatest(smallest(outer_ax, ax), unit_ax)
    ax_translated = smallest(outer_ax, square_outwards(model_late) )
    overall_average = np.average(averages, axis=0)
    wobble_factor = np.max(np.linalg.norm(averages[:,:] - overall_average, axis=1)) / np.linalg.norm(ax_translated[0])
    #print(f'wobble_factor={wobble_factor:.2f}')
    if wobble_factor >= 0.1:
        return ax_translated[np.newaxis,:,:] + averages[:,:,:]
    else:
        return ax_translated[:,:] + overall_average

def regimeFinder(C, l, tol=0.01):
    assert tol >= 0.0
    if C <= 1. and l <= 1.: # I, II, III
        diff = C-l
        if tol < diff:
            return 1
        elif -tol <= diff <= tol:
            return 2
        elif diff < -tol:
            return 3
        else:
            return 0 # error
    elif C <=1. and l > 1.: # IV
        return 4
    elif C > 1. and l > 1.: # V
        return 5
    elif C > 1. and l <= 1.: # VI, VII
        if C * l**2 > 1.0:
            return 6
        else:
            return 7
    else:
        return 0 # error




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

def morseScipy(t, x_t):
    resultVel = (a - b * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    diffs = x_t[:,0,np.newaxis] - np.transpose(x_t[:,0], axes=(0,1))
    norms = sciDist.pdist(x_t[:,0], metric='euclidean') #creates 1d distance vector
    factors = (C_r/l_r * np.exp(-norms/l_r) 
              -C_a/l_a * np.exp(-norms/l_a)) / norms
    resultVel += (sciDist.squareform(factors)[:,:,np.newaxis] 
                  * diffs).sum(axis=1) # returns the factor to 2d
    return np.stack((x_t[:,1,:], resultVel), axis=1)



# Model parameters
k_agents = 125
n_timesteps = 4000##############10000
T = 40#############500
fps = 60
dim = 3#############2
regimeColors=['gray', 'b', 'k' if dim == 3 else 'w', 'r', 'm', 'y', 'g', 'c']


# 1 value
#Cs = np.arange(0.8, .9, 1.)
#ls = np.arange(0.8, .9, 1.)

# 3x2
Cs = np.arange(0.6, 1.85, 0.6)
ls = np.arange(1.2, 0.55, -0.6)

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
#Cs = np.arange(0.5, 1.7, .4)
#ls = np.arange(1.3, .5, -.4)

# 3x2
#Cs = np.arange(0.5, 1.35, 0.4)
#ls = np.arange(1.2, 0.79,-0.4)

# 7x5 = 35
#Cs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
#ls = [0.9, 0.7, 0.5, 0.3, 0.1]

# 8x6 = 48
#Cs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
#ls = [1.1, 0.9, 0.7, 0.5, 0.3, 0.1]

# 6x4
#Cs = np.arange(0.25, 1.7501, 0.3)
#ls = np.arange(1.15, 0.2499,-0.3)


#####ls = [1.2, 0.8]

print(f'Cs = {Cs}\nls = {ls}')

allparams = [[[1.0, 0.5, 1.0, C_r , 1.0, l_r, k_agents, n_timesteps, T] for C_r in Cs] for l_r in ls]
############allparams = [[[a, .5, 1.0, 2.0, 1.0, l_r, k_agents, n_timesteps, T] for a in [1.0, 0.1, 0.01] ] for l_r in ls]
print(allparams)
plottitles = [[f'C={C_r:.2f}, l={l_r:.2f}' for C_r in Cs] for l_r in ls]
#################plottitles = [[f'C=2.0,  l={l_r:.1f},  \u03b1={a}' for a in [1.0, 0.1, 0.01] ] for l_r in ls]
borderColors = [[regimeColors[regimeFinder(C_r, l_r)] for C_r in Cs] for l_r in ls]
print(np.array(plottitles))
print(np.array(borderColors))
#print(len(allparams), len(allparams[0]))

xs = np.zeros((len(ls), len(Cs), n_timesteps+1, k_agents, 2, dim))
print(xs.shape)
p_0 = -0.5*np.ones((k_agents, dim)) + np.random.rand(k_agents, dim)# *1 


#jp_p_0 = jsonpickle.encode(p_0.tolist(), indent=4)
#with open("positions3.json", "w+") as f:
#    f.write(jp_p_0)
#v_0 = np.random.rand(k, 2) *0.1

#                     [ α ,  β , C_a, C_r, l_a, l_r,  k,   n,  T]
params_clumps =       [1. ,  .5, 1. ,  .6, 1. ,  .5, 40, 1500, 30, "clumping"       ]
params_ringclumping = [1. ,  .5, 1. ,  .6, 1. , 1.2, 40, 3000, 60, "ringClmp"]
params_rings =        [1. ,  .5, 1. ,  .5, 1. ,  .5, 40, 3000, 60, "circular"        ]
params_catastrophic = [1.6,  .5,  .5, 1. , 2. ,  .5, 40, 3000, 12, "catastr." ]
paperparams = [params_clumps, params_ringclumping, params_rings, params_catastrophic]
paperparams = [params_ringclumping, params_rings, params_catastrophic]

#      [animname  , edg,  k, n_ts,  T, C_r, l_r]
cls = [['clumping', 1.0, 40, 1500, 30, 0.6, 0.5]]
cls = [['circular', 1.0, 40, 1500, 30, 0.5, 0.5]]
cls = [['ringClmp', 0.5, 40, 3000, 60, 0.6, 1.2]]
#cls = [['ringClmp', 0.2,100, 5000, 10, 0.6, 1.2]]
cls = [['region 3', 0.5, 100, 3000, 40, 0.5, 0.6]]
cls = [['region 3', 1.0,  40, 3000, 60, 0.5, 1.0]]
cls = [['region 3', 1.0,  40,10000,200, 0.5, 1.2]]
cls = [['region 3',0.25, 250,10000,200, 0.5, 1.2]]
cls = [['region 3',0.25,1000,10000,200, 0.5, 0.6]]
cls = [['region 3',0.25,1000,10000,200, 0.5, 1.2]]
cls = [['region 3', 0.5,  40, 5000,500, 0.5, 1.2],
       ['region 3', 0.5,  40, 5000,500, 0.5, 0.6]]

'''cls = [['region 5', 1.0, 40, 1500, 30, 1.2, 1.2],
       ['region 5',10.0, 40, 1500, 30, 1.2, 1.2],
       ['region 5', 1.0, 40, 1500,300, 1.2, 1.2],
       ['region 5',10.0, 40, 1500,300, 1.2, 1.2],
       ['region 5', 1.0,100, 1500, 30, 1.2, 1.2],
       ['region 5',10.0,100, 1500, 30, 1.2, 1.2],
       ['region 5', 1.0,100, 1500,300, 1.2, 1.2],
       ['region 5',10.0,100, 1500,300, 1.2, 1.2]]'''

cls = [['region 5',10.0,100, 1500, 15, 1.2, 1.2]]

cls = [['VI, VII', 8.0,  40, 5000,50, 1.75, 0.4],
       ['VI, VII', 8.0, 100, 5000,50, 1.75, 0.4],
       ['VI, VII', 8.0,  40, 5000,50, 1.75, 0.6],
       ['VI, VII', 8.0, 100, 5000,50, 1.75, 0.6],
       ['VI, VII', 8.0,  40, 5000,50, 1.75, 0.8],
       ['VI, VII', 8.0, 100, 5000,50, 1.75, 0.8]]


cls = [['VII',   10.0, 64, 5000,250,10.0 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,10.0 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250, 5.  , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250, 5.  , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250, 2.0 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250, 2.0 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250, 1.0 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250, 1.0 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .5 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .5 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .2 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .2 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .1 , 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .1 , 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .05, 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .05, 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .02, 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .02, 0.5, 2.0, 0.5],
       ['VII',   10.0, 64, 5000,250,  .01, 0.5, 2.0, 0.5],
       ['VII',  100.0, 64, 5000,250,  .01, 0.5, 2.0, 0.5]]



cls = [['VII',   10.0,250, 1500,150, 1.0 , 0.5, 2.0, 0.4],
       ['VII',    6.0,250, 1500,150, 1.0 , 0.5, 2.0, 0.4],
       ['VII',    4.0,250, 1500,150, 1.0 , 0.5, 2.0, 0.4],
       ['VII',    2.5,250, 1500,150, 1.0 , 0.5, 2.0, 0.4],
       ['VII',   10.0,250, 1500,150, 1.0 , 0.5, 1.6, 0.5],
       ['VII',    6.0,250, 1500,150, 1.0 , 0.5, 1.6, 0.5],
       ['VII',    4.0,250, 1500,150, 1.0 , 0.5, 1.6, 0.5],
       ['VII',    2.5,250, 1500,150, 1.0 , 0.5, 1.6, 0.5]]
#[0.5, 0.1, 10.0]:#, 0.4, 2.5, 100.0]
# Visualize as gif
dim = 2
for params in cls:
    #[ a ,  b , C_a, C_r, l_a, l_r,  k_agents,   n_timesteps,  T, animname] = params
    [animname, edge, k_agents, n_timesteps, T, a, b, C_r, l_r] = params
    C_a, l_a = 1.0, 1.0
    fps = 50
    duration = n_timesteps // fps
    if not os.path.exists(f'anim/{animname}'):
        os.makedirs(f'anim/{animname}') 
    filename = f'anim/{animname}/a{a:.2f}, b{b:.2f}, C{C_r:.2f}, l{l_r:.2f}, T{T}, {k_agents}ag {edge:.2f}ax {duration:d}s'
    print(filename)
    t = np.linspace(0, T, n_timesteps+1)
    x = np.zeros((n_timesteps+1, k_agents, 2, dim))
    p_0 = -0.5*np.ones((k_agents, dim)) + np.random.rand(k_agents, dim)# *1 
    x[0,:,0] = p_0
    dt = T/n_timesteps
    time_solve_start = time.time()
    print(    f'\tsolved timesteps: {str(0).rjust(len(str(n_timesteps+1)))} / {n_timesteps}\t({0/n_timesteps:.0%})', end="\r")
    for i in range(n_timesteps):
        x[i+1,:,:,:] = x[i,:,:,:] + dt*rk4(t[i], x[i,:,:,:], dt, morseScipy)
        print(f'\tsolved timesteps: {str(i).rjust(len(str(n_timesteps+1)))} / {n_timesteps}\t({i/n_timesteps:.0%})', end="\r")
    time_solve_end = time.time()
    time_solve_duration = time_solve_end - time_solve_start
    print(f'\n\tsolving with {morseScipy.__name__} took {time_solve_duration:.2f} seconds')
    #print("\tcreating .gif")
    #visualize_2d(x[:,:,0,:], vel=None, xmin=-edge, xmax=edge, ymin=-edge, ymax=edge, save_anim=True, filename=filename, interval=20, format='.gif')
    time_mp4_start = time.time()
    print("\tcreating .mp4")
    permtitle = f'  \u03b1={a:.2f}  \u03b2={b:.2f}  C={C_r:.1f}  l={l_r:.1f}  T={T}  {k_agents}agents'
    permtitle = f'  C={C_r:.1f}  l={l_r:.1f}  T={T}  {k_agents}agents'
    visualize_2d(x[:,:,0,:], vel=None, xmin=-edge, xmax=edge, ymin=-edge, ymax=edge, save_anim=True, filename=filename, interval=20, format='.mp4', draw_rectangle=[-0.5, 0.5, -0.5, 0.5], titleAppendix=permtitle)
    time_mp4_end = time.time()
    time_mp4_duration = time_mp4_end - time_mp4_start
    print(f'\tcreating .mp4 file took {time_mp4_duration:.1f} seconds')
        

assert False




# visualize as frames and use FFMpeg
start = time.time()
for img_y in range(len(ls)):
    for img_x in range(len(Cs)):
        [a, b, C_a, C_r, l_a, l_r, k_agents, n_timesteps, T] = allparams[img_y][img_x]
        dt = T/n_timesteps

        print(f'solving ODE for plot ({img_y}, {img_x})\n\tplottitle={plottitles[img_y][img_x]}, dt={dt}')

        t_ = np.linspace(0, T, n_timesteps+1)
        x = np.zeros((n_timesteps+1, k_agents, 2, dim))  # timesteps; agents; pos,vel; x,y

        # initial condition: spread agents over [0,8]x[0,8] square

        #with open("positions2.json", "r+") as f:
        #    jp_loaded = f.read()
        #    p_0 = np.array(jsonpickle.decode(jp_loaded))

        x[0,:,0,:] = p_0.copy()
        #x[0,:,1,:] = v_0

        # solving
        time_solve_start = time.time()
        print(    f'\tsolved timesteps: {str(0).rjust(len(str(n_timesteps+1)))} / {n_timesteps}\t({0/n_timesteps:.0%})', end="\r")
        for i in range(n_timesteps):
            x[i+1,:,:,:] = x[i,:,:,:] + dt*rk4(t_[i], x[i,:,:,:], dt, morseScipy)
            print(f'\tsolved timesteps: {str(i).rjust(len(str(n_timesteps+1)))} / {n_timesteps}\t({i/n_timesteps:.0%})', end="\r")
        time_solve_end = time.time()
        time_solve_duration = time_solve_end - time_solve_start
        print(f'\tsolving with {morseScipy.__name__} took {time_solve_duration:.2f} seconds')
        #assert False

        '''def behavior2(t, x_t, tindex):
            result = np.stack(((t_[tindex]-t)*(interaction(t, x_t[:,0,:]) + generalmov(t, x_t[:,0,:])) , interaction(t, x_t[:,0,:]) + generalmov(t, x_t[:,0,:])), axis=1)
            return result

        x2 = np.zeros((n+1, k, 2, 2))  # timesteps; agents; pos,vel; x,y
        x2[0,:,0,:] = p_0
        #x[0,:,1,:] = v_0

        # solving
        for i in range(n):
            print(f'solving ODE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
            x2[i+1] = x2[i] + dt*np.stack((x2[i,:,1,:],np.zeros((k, 2))), axis=1) + dt*rk4(t_[i], x2[i], dt, lambda t, x2_t: behavior2(t, x2_t, i))
            #plt.scatter(p[i, :, 0], p[i, :, 1])
            #plt.show()
        print()

        #print(x.shape)
        #print(x2.shape)
        xStacked = np.concatenate((x, x2), axis=1)
        #print(xStacked.shape)
        colors_for_xStacked = np.array([(['r']*k + ['c']*k) for i in range(n+1)])'''



        '''stabtime_srt  = time.time()
        h_stab  =  U_morse(x[:,:,1,:], C_r, l_r) / (-N)
        stabtime_end  = time.time()
        print(f'\n\ttime for morse potential = {stabtime_end -stabtime_srt :.2f} sec')

        plt.plot(t_, h_stab)
        plt.axhline(y=0, color='k', linewidth=1)
        #plt.axvline(x=0, color='k', linewidth=1)
        plt.savefig('H:/_dump/parameter studies/2024-12-31/'+plottitles[img_y][img_x]+'.png')
        plt.close()'''

        if np.isfinite(x[-1,...]).all():
            pass
            '''axis = axisfinder(x[:,:,0,:])
            print(axis)
            order = [k-i for i in range(k+1)]

            #timeVisStart = time.time()
            #visualize_2d_ffmpeg(pos=x[:,:,0,:], vel=None, *axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier=name)
            timeVisCP = time.time()
            #visualize_2d_ffmpeg(pos=x[:,:,0,:], vel=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier=plottitles[img_x][img_y])
            unitSq = np.array([-1., 1., -1., 1.])
            squarestack = np.stack((np.stack((x[:,:,0,:], x[:,:,0,:]), axis=0), np.stack((x[:,:,0,:], x[:,:,0,:]), axis=0)), axis=0)
            squareaxis = [[unitSq, 10*unitSq], [100*unitSq, 1000*unitSq]]
            squareplottitles = [["zoom 1", "zoom 10"], ["zoom 100", "zoom 1000"]]
            ###visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss=squarestack, vels=None, axis=squareaxis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier=plottitles[img_x][img_y], plottitles=squareplottitles, res=(19.20, 10.80), order=order)
            timeVisEnd = time.time()
            #print(f"time for H: {timeVisCP-timeVisStart:.1f} seconds")
            print(f"time for X: {timeVisEnd-timeVisCP:.1f}")'''
        else:
            print(f"invalid values encountered: {x[-1,0,0,:]}")
            
        xs[img_y, img_x] = x

#print(results[fi][:,:, 0,:4,:,:])
#print(results[fi][:,:, 1,:4,:,:])
#print(f"xs[..., 1,-3, :, :]={xs[:,:, 1,-3,:,:]}\n\nxs[...,-1,-3, :, :]={xs[:,:,-1,-3,:,:]}")
#endfunc = time.time()
#durationfunc = endfunc-startfunc
#print(f"the computation took {durationfunc:.1f} seconds")
axis = [[axisfinder(xs[img_y,img_x,:,:,0,:]) for img_y in range(len(ls))] for img_x in range(len(Cs))]
#print(axis)
###############axis = [[np.array([[-100, -100],[100, 100]]) for img_y in range(len(ls))] for img_x in range(len(Cs))]


order = [n_timesteps-i for i in range(n_timesteps+1)]
#print(n, len(order))
#print(order)
assert len(order)==n_timesteps+1
#visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=[-0.5, 1.5, -0.5, 1.5], T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=False, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, file_identifier="", plottitles=plottitles, res=(38.40, 21.60))
    
origin = None
if dim == 3:
    origin = 'k'
timeVisStart = time.time()
visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axes=axis, borderColors=borderColors, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=origin, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
timeVisCP = time.time()
#visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
#timeVisEnd = time.time()
#print(f"time for H: {timeVisCP-timeVisStart:.1f} seconds")
#print(f"time for X: {timeVisEnd-timeVisCP:.1f}")

        # visualizations
        #visualize_2d(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, save_anim=True)
        #visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, name=name)
        #visualize_2d_ffmpeg(pos=x2[:,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)
        #visualize_2d_ffmpeg(pos=xStacked[:,:,0,:], vel=None, xmin=-4, xmax=5, ymin=-4, ymax=5, T_max=T, dt=dt, img_save_path=img_save_path, fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=colors_for_xStacked)
end = time.time()
print(f"total time {end-start:.1f} seconds")
