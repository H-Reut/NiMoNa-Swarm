import numpy as np
import matplotlib.pyplot as plt
import jsonpickle
from visualization import visualize_2d
from visualization_ffmpeg import visualize_2d_ffmpeg
import visualization_ffmpeg_multi
import time

img_save_path = 'H:/dump/' # Path for saving the images. Example: 'C:/dump/'


def rk4(t, y, Δt, f):
    k1 = f(t, y)
    k2 = f(t+(Δt/2), y + (Δt*k1)/2)
    k3 = f(t+(Δt/2), y + (Δt*k2)/2)
    k4 = f(t+Δt, y + Δt*k3)
    return k1/6 + k2/3 + k3/3 + k4/6


def axisfinder(p):
    # format: xmin, xmax, ymin, ymax
    def model(p):
        return np.vstack((np.min(np.min(p, 1), 0), np.max(np.max(p, 1), 0))).reshape(4, order='F')
    def greatest(ax1, ax2):
        return np.array([min(ax1[0], ax2[0]),
                         max(ax1[1], ax2[1]),
                         min(ax1[2], ax2[2]),
                         max(ax1[3], ax2[3])])

    def smallest(ax1, ax2):
        return np.array([max(ax1[0], ax2[0]),
                         min(ax1[1], ax2[1]),
                         max(ax1[2], ax2[2]),
                         min(ax1[3], ax2[3])])

    def toCenter(ax):
        return np.array([-np.min(np.abs(ax[:2])),
                          np.min(np.abs(ax[:2])),
                         -np.min(np.abs(ax[2:])),
                          np.min(np.abs(ax[2:]))])
    T1 = k//T
    model_early = model(p[:T1])
    model_mid = model(p[T1:-T1])
    model_late = model(p[-T1:])
    default_ax  = np.array([-1., 1., -1., 1.])
    unit_ax  = np.array([-0.5, 0.5, -0.5, 0.5])
    outer_ax = 2*np.array([-100.,100.,-100.,100.])
    ax = greatest(model_late, greatest(toCenter(model_mid), toCenter(model_early)))
    return greatest(smallest(outer_ax, ax), unit_ax)


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

def morse9(t, x_t):
    # General movements:
    resultVel = (α - β * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    for i in range(N):
        # Movements caused by other agents.
        diffs = x_t[i,0]-x_t[:,0] # for i-th entry we have: diffs[i]=0.0
        norms = np.linalg.norm(diffs, axis=-1) # for i-th entry we have norms[i]=0.0
        norms[i] = np.inf
        factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a)) / norms # for i-th entry we have: factors[i]=0.0
        resultVel[i] += factors @ diffs # The effect of agent i on itself is ignored because (factors@diffs)[i]=[0.0, 0.0]
    return np.stack((x_t[:,1,:], resultVel), axis=1)


# Model parameters
N = 250
k = 120#2500
T = 3#100
fps = 30

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
Cs = np.arange(0.8, .9, 1.)
ls = np.arange(0.8, .9, 1.)

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
Cs = np.arange(0.5, 2.0, 0.2)
ls = np.arange(1.3, 0.49, -0.2)

# 4x3
#Cs = np.arange(0.5, 2.0, 0.4)
#ls = np.arange(1.3, 0.49,-0.4)

# 3x2
#Cs = np.arange(0.5, 1.35, 0.4)
#ls = np.arange(1.2, 0.79,-0.4)

print(Cs, ls)


allparams = [[[1.0, 0.5, 1.0, C_r , 1.0, l_r, N, k, T] for C_r in Cs] for l_r in ls]
plottitles = [[f'C={C_r:.1f}, l={l_r:.1f}' for C_r in Cs] for l_r in ls]
#print(len(allparams), len(allparams[0]))

xs = np.zeros((len(ls), len(Cs), k+1, N, 2, 2))
print(xs.shape)
p_0 = -0.5*np.ones((N, 2)) + np.random.rand(N, 2) *1 


jp_p_0 = jsonpickle.encode(p_0.tolist(), indent=4)
with open("positions3.json", "w+") as f:
    f.write(jp_p_0)
#v_0 = np.random.rand(k, 2) *0.1


start = time.time()


startfunc = time.time()
for img_y in range(len(ls)):
    for img_x in range(len(Cs)):
        [α, β, C_a, C_r, l_a, l_r, N, k, T] = allparams[img_y][img_x]
        dt = T/k

        print(f'img_y={img_y}, img_x={img_x},\t{allparams[img_y][img_x]}, dt={dt}')

        t_ = np.linspace(0, T, k+1)
        x = np.zeros((k+1, N, 2, 2))  # timesteps; agents; pos,vel; x,y

        # initial condition: spread agents over [0,8]x[0,8] square

        #with open("positions2.json", "r+") as f:
        #    jp_loaded = f.read()
        #    p_0 = np.array(jsonpickle.decode(jp_loaded))

        x[0,:,0,:] = p_0.copy()
        #x[0,:,1,:] = v_0



        # solving
        for i in range(k):
            print(f'solving PDE. timestep\t{str(i).rjust(len(str(k)))} / {k}\t({i/k:.0%})', end="\r")
            #print()
            x[i+1,:,:,:] = x[i,:,:,:] + dt*rk4(t_[i], x[i,:,:,:], dt, morse9)
        print()
        #assert False

        '''def behavior2(t, x_t, tindex):
            result = np.stack(((t_[tindex]-t)*(interaction(t, x_t[:,0,:]) + generalmov(t, x_t[:,0,:])) , interaction(t, x_t[:,0,:]) + generalmov(t, x_t[:,0,:])), axis=1)
            return result

        x2 = np.zeros((n+1, k, 2, 2))  # timesteps; agents; pos,vel; x,y
        x2[0,:,0,:] = p_0
        #x[0,:,1,:] = v_0

        # solving
        for i in range(n):
            print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
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
endfunc = time.time()
durationfunc = endfunc-startfunc
print(f"the computation took {durationfunc:.1f} seconds")
#print(result[:,:, 0,:4,:,:])
#print(result[:,:, 1,:4,:,:])
#print(f"xs[..., 1,-3, :, :]={xs[:,:, 1,-3,:,:]}\n\nxs[...,-1,-3, :, :]={xs[:,:,-1,-3,:,:]}")
axis = [[axisfinder(xs[img_y,img_x,:,:,0,:]) for img_y in range(len(ls))] for img_x in range(len(Cs))]
print(axis)




order = [k-i for i in range(k+1)]
#print(n, len(order))
#print(order)
assert len(order)==k+1
#visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=[-0.5, 1.5, -0.5, 1.5], T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=False, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, file_identifier="", plottitles=plottitles, res=(38.40, 21.60))
    
timeVisStart = time.time()
visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
timeVisCP = time.time()         
#visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
#timeVisEnd = time.time()
print(f"time for H: {timeVisCP-timeVisStart:.1f} seconds")
#print(f"time for X: {timeVisEnd-timeVisCP:.1f}")

        # visualizations
        #visualize_2d(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, save_anim=True)
        #visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, name=name)
        #visualize_2d_ffmpeg(pos=x2[:,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)
        #visualize_2d_ffmpeg(pos=xStacked[:,:,0,:], vel=None, xmin=-4, xmax=5, ymin=-4, ymax=5, T_max=T, dt=dt, img_save_path=img_save_path, fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=colors_for_xStacked)
end = time.time()
print(f"total time {end-start:.1f} seconds")
