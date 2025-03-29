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
    T1 = n//T
    model_early = model(p[:T1])
    model_mid = model(p[T1:-T1])
    model_late = model(p[-T1:])
    default_ax  = np.array([-1., 1., -1., 1.])
    unit_ax  = np.array([-0.5, 0.5, -0.5, 0.5])
    outer_ax = 2*np.array([-10.,10.,-10.,10.])
    ax = greatest(model_late, greatest(toCenter(model_mid), toCenter(model_early)))
    return greatest(smallest(outer_ax, ax), unit_ax)


def morse1(t, x_t):
    result = np.zeros((k, 2))
    for i in range(k):
        result[i] = (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k):
            if i != j:
                GradU_r = -(x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0])/l_r * np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_r)
                GradU_a = -(x_t[i,0]-x_t[j,0])/np.linalg.norm(x_t[i,0]-x_t[j,0])/l_a * np.array([1., 1.]) * np.exp(- np.linalg.norm(x_t[i,0]-x_t[j,0]) /l_a)
                result[i] -= C_r * GradU_r - C_a * GradU_a
                #print((C_r * GradU_r - C_a * GradU_a).shape)
                #print(result[i])
    return np.stack((x_t[:,1,:], result), axis=1)

'''def morse2(t, x_t):
    result = np.zeros((k, 2))
    for i in range(k):
        result[i] = (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)'''

def morse2(t, x_t):
    result = np.zeros((k, 2))
    for i in range(k):
        result[i] = (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]

    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse3(t, x_t):
    result = x_t[:,1].copy()
    for i in range(k):
        #print("x_t    ",x_t[i,1])
        #print("fac    ",(α - β * np.linalg.norm(x_t[i,1])**2))
        #print("fac*x_t",(α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1])
        result[i] *= (α - β * np.linalg.norm(x_t[i,1])**2)
        #print("res[i] ",result[i])
        #print("fac    ",(α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1])
        #print("error  ",result[i] - (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1])
    #print(result)
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse4(t, x_t):
    print("x_t\n",x_t[:3,0,:])
    print("v_t\n",x_t[:3,1,:])
    result = np.zeros((k, 2))
    normsV = np.linalg.norm(x_t[:,1,:], axis = -1)
    print("norms\n",normsV[:3], normsV.shape)
    normsVsq = np.square(normsV)
    print("normsSq\n",normsVsq[:3], normsVsq.shape)
    factors = α - β * normsVsq
    print("factors\n",factors[:3], factors.shape)
    for i in range(k):
        result[i] = (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]
        if i < 3:
            print(f"\tcorrect result 1[{i}] = {result[i]}")
        if i < 3:
            print(f"\tcorrect result 2[{i}] = {x_t[i,1,:]*factors[i]}")

    resultA = x_t[:,1].copy()
    #print("resultA\n",resultA[:3])
    for i in range(k):
        resultA[i] *= (α - β * np.linalg.norm(x_t[i,1])**2)
        if i < 3:
            print(f"\tresultA[{i}] = {resultA[i]}")
    resultB = x_t[:,1].copy() * (α - β * np.square(np.linalg.norm(x_t[:,1], axis=-1))).reshape(40,1)
    #print("resultB\n",resultB)
    #print("resultA\n",resultA)
    print(resultA - resultB)
    #print((resultA == resultB))
    assert (resultA == resultB).all()
    assert (result  == resultB).all()
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse4a(t, x_t):
    result = x_t[:,1,:].copy()
    for i in range(k):
        result[i] *= (α - β * np.square(np.linalg.norm(x_t[i,1])))
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse4b(t, x_t):
    result = x_t[:,1].copy() * (α - β * np.square(np.linalg.norm(x_t[:,1], axis=-1))).reshape(40,1)
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse5(t, x_t):
    result = np.zeros((k, 2))
    result = x_t[:,1,:]#.copy()
    for i in range(k):
        #print((α - β * np.square(np.linalg.norm(x_t[i,1,:], axis=-1))))
        result[i] *= (α - β * np.square(np.linalg.norm(x_t[i,1,:], axis=-1)))
        #print(result[i])
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse6(t, x_t):
    result = x_t[:,1,:] * ((α - β * (np.square(np.linalg.norm(x_t[:,1], axis=-1)))).reshape(40,1))
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[i,0]-x_t[j,0]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[:,1,:], result), axis=1)

def morse7(t, x_t):
    normsV = np.linalg.norm(x_t[:,1,:], axis = -1, keepdims=True)
    #print("norms\n",normsV[:3], normsV.shape)
    normsVsq = np.square(normsV)
    #print("normsSq\n",normsVsq[:3], normsVsq.shape)
    factors = α - β * normsVsq
    #print("factors\n",factors[:3], factors.shape)
    result = x_t[:,1,:] * factors[:]
    for i in range(k):
        #result[i] = (α - β * np.linalg.norm(x_t[i,1])**2) * x_t[i,1]

        diff = x_t[i,0]-x_t[:,0] # for i-th entry we have: diff[i]=0
        #print(diff)
        norm = np.linalg.norm(diff, axis=-1) # for i-th entry we have: norm[i]=0
        #print(norm)
        factor = (C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))
        #print(factor)
        summand = np.nan_to_num(factor/norm) # for i-th entry we have: (factor/norm)[i]=NaN   This line turns Nan into 0
        #print(summand)
        res = summand @ diff
        #print(res)
        result[i] += res
    return np.stack((x_t[:,1,:], result), axis=1)

'''def morse7(t, x_t):
    result = (α - β * np.linalg.norm(x_t[...,:,1,:])**2) * x_t[...,:,1,:]
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = x_t[...,i,0,:]-x_t[...,j,0,:]
                norm = np.linalg.norm(diff)
                result[i] += ((C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))/norm) * diff
    return np.stack((x_t[...,:,1,:], result), axis=1)'''

def morse8(t, x_t):
    result = (α - β * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1].copy()
    for i in range(k):
        diff = x_t[i,0]-x_t[:,0]
        #print(diff)
        norm = np.linalg.norm(diff, axis=-1)
        #print(norm)
        factor = (C_r/l_r * np.exp(-norm/l_r) - C_a/l_a * np.exp(-norm/l_a))
        #print(factor)
        summand = np.nan_to_num(factor/norm)
        #print(summand)
        res = summand @ diff
        #print(res)
        result[i] += res
    return np.stack((x_t[:,1,:], result), axis=1)

def morse9(t, x_t):
    result = (α - β * np.square(np.linalg.norm(x_t[:,1], axis=-1, keepdims=True))) * x_t[:,1]
    for i in range(k):
        diffs = x_t[i,0]-x_t[:,0] # for i-th entry we have: diffs[i]=0
        #print(diffs)
        norms = np.linalg.norm(diffs, axis=-1) # for i-th entry we have: norms[i]=0
        #print(norms)
        factors = (C_r/l_r * np.exp(-norms/l_r) - C_a/l_a * np.exp(-norms/l_a)) / norms # for i-th entry we have: factors[i]=NaN   
        #print(factors)
        factors[i] = 0.0 # fixing i-th entry
        result[i] += factors @ diffs
    return np.stack((x_t[:,1,:], result), axis=1)


# Model parameters
k = 40
n = 100#400
T = 20
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

# 2x3
#Cs = np.arange(0.6, 1.1, 0.4)
#ls = np.arange(0.6, 1.5, 0.4)

# 3x4
#Cs = np.arange(0.6, 1.7, 0.4)
#ls = np.arange(0.6, 1.9, 0.4)

# 4x4
#Cs = np.arange(0.6, 1.2, 0.2)
#ls = np.arange(0.6, 1.2, 0.2)

# 3x5
#Cs = np.arange(0.6, 1.9, 0.4)
#ls = np.arange(0.6, 2.3, 0.4)

# 8x10
#Cs = np.arange(0.6, 2.1, 0.2)
#ls = np.arange(0.4, 2.3, 0.2)

print(Cs, ls)


allparams = [[[1.0, 0.5, 1.0, C_r , 1.0, l_r, f'C_r={C_r:.1f},l_r={l_r:.1f}', k, n, T] for l_r in ls] for C_r in Cs]
#print(len(allparams), len(allparams[0]))

xs = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
print(xs.shape)
p_0 = -0.5*np.ones((k, 2)) + np.random.rand(k, 2) *1 
#v_0 = np.random.rand(k, 2) *0.1

plottitles = [[f'C={C_r:.1f}, l={l_r:.1f}' for l_r in ls] for C_r in Cs]

#start = time.time()
x0  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x1  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x2  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x3  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x4  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x4a = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x4b = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x5  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x6  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x7  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x8  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))
x9  = np.zeros((len(Cs), len(ls), n+1, k, 2, 2))

#allfunctions = [[morse1, "morse1", x1], [morse2, "morse2", x2], [morse3, "morse3", x3], [morse4, "morse4", x4], [morse5, "morse5", x5], [morse6, "morse6", x6], [morse7, "morse7", x7], [morse8, "morse8", x8]]
allfunctions = [[morse1, "morse1", x1], [morse2, "morse2", x2], [morse3, "morse3", x3], [morse4a, "morse4a", x4a], [morse4b, "morse4b", x4b], [morse5, "morse5", x5], [morse6, "morse6", x6], [morse7, "morse7", x7], [morse8, "morse8", x8]]
#allfunctions = [[morse1, "morse1", x1], [morse2, "morse2", x2], [morse8, "morse8", x8], [morse9, "morse9", x9]]
#allfunctions = [[morse9, "morse9", xs]]
for funcArray in allfunctions:
    func, funcname, result = funcArray
    startfunc = time.time()
    for img_x in range(len(Cs)):
        for img_y in range(len(ls)):
            [α, β, C_a, C_r, l_a, l_r, name, k, n, T] = allparams[img_x][img_y]
            dt = T/n

            #print(f'img_x={img_x}, img_y={img_y},\tparams={allparams[img_x][img_y]}, dt={dt}')

            t_ = np.linspace(0, T, n+1)
            x = np.zeros((n+1, k, 2, 2))  # timesteps; agents; pos,vel; x,y

            # initial condition: spread agents over [0,8]x[0,8] square
            #jp_p_0 = jsonpickle.encode(p_0.tolist(), indent=4)
            #with open("positions3.json", "w+") as f:
            #    f.write(jp_p_0)

            #with open("positions2.json", "r+") as f:
            #    jp_loaded = f.read()
            #    p_0 = np.array(jsonpickle.decode(jp_loaded))
            x[0,:,0,:] = p_0
            #x[0,:,1,:] = v_0



            # solving
            for i in range(n):
                print(f'solving PDE. timestep\t{str(i).rjust(len(str(n)))} / {n}\t({i/n:.0%})', end="\r")
                #print()
                x[i+1,:,:,:] = x[i,:,:,:] + dt*rk4(t_[i], x[i,:,:,:], dt, func)
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

            if np.isfinite(x[-1,...]).all():
                timeVisStart = time.time()
                #visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=-1., xmax=1., ymin=-1., ymax=1., T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier=name)
                timeVisCP = time.time()
                #visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=-1., xmax=1., ymin=-1., ymax=1., T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier=plottitles[img_x][img_y])
                #timeVisEnd = time.time()
                print(f"time for H: {timeVisCP-timeVisStart:.1f} seconds")
                #print(f"time for X: {timeVisEnd-timeVisCP:.1f}")
            else:
                print(f"invalid values encountered: {x[-1,0,0,:]}")
            
            result[img_x, img_y] = x
    endfunc = time.time()
    durationfunc = endfunc-startfunc
    print(f"the function {funcname} took {durationfunc:.1f} seconds")
    #print(result[:,:, 0,:4,:,:])
    #print(result[:,:, 1,:4,:,:])
    #print(result[:,:, 1,-1,:,:],"\n\n")
    #print(result[:,:,-1,-1,:,:])
    axis = [[axisfinder(xs[img_x,img_y,:,:,0,:]) for img_y in range(len(ls))] for img_x in range(len(Cs))]
    #print(axis)




    order = [n-i for i in range(n+1)]
    #print(n, len(order))
    #print(order)
    assert len(order)==n+1
    #visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=[-0.5, 1.5, -0.5, 1.5], T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=False, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, file_identifier="", plottitles=plottitles, res=(38.40, 21.60))
    
    #timeVisStart = time.time()
    #visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
    #timeVisCP = time.time()         
    #visualization_ffmpeg_multi.visualize_2d_ffmpeg_multi(poss= xs[:,:,:,:,0,:], vels=None, axis=axis, T_max=T, dt=dt, img_save_path=img_save_path, fps=fps, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[-0.5, 0.5, -0.5, 0.5], origin_color=None, file_identifier="", plottitles=plottitles, res=(19.20, 10.80), order=order)
    #timeVisEnd = time.time()
    #print(f"time for H: {timeVisCP-timeVisStart:.1f} seconds")
    #print(f"time for X: {timeVisEnd-timeVisCP:.1f}")

        # visualizations
        #visualize_2d(pos=p, vel=None, xmin=0, xmax=8, ymin=0, ymax=8, save_anim=True)
        #visualize_2d_ffmpeg(pos= x[:,:,0,:], vel=None, xmin=-0.5, xmax=1.5, ymin=-0.5, ymax=1.5, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=False, inverse_arrow_scale_factor=1, draw_rectangle=[0., 1., 0., 1.], origin_color=None, name=name)
        #visualize_2d_ffmpeg(pos=x2[:,:,0,:], vel=None, xmin=0, xmax=8, ymin=0, ymax=8, T_max=T, dt=dt, img_save_path=img_save_path, fps=10, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None)
        #visualize_2d_ffmpeg(pos=xStacked[:,:,0,:], vel=None, xmin=-4, xmax=5, ymin=-4, ymax=5, T_max=T, dt=dt, img_save_path=img_save_path, fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=colors_for_xStacked)
#end = time.time()
#print(end-start)
#print(time.ctime(start))
#print(time.ctime(end))
#print(time.ctime(end-start))