import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess
import time
import os

print("Matplotlib Backend:", matplotlib.get_backend())


def plotOneFrame(fig, ax, pos, vel=None, timestep=0, numtimesteps=0, xmin=-5, xmax=5, ymin=-5, ymax=5, T_max=0, dt=0, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=None, subplotName=None):
    assert timestep <= numtimesteps
    assert (pos.ndim == 2), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    assert (pos.shape[1] == 2), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    if vel is not None:
        assert (pos.shape == vel.shape), 'Velocities need to have the same shape as the positions'
    k_agents = pos.shape[0]

    assert xmin < xmax, 'xmin needs to be smaller than xmax'
    assert ymin < ymax, 'ymin needs to be smaller than ymax'

    assert draw_rectangle[0] <= draw_rectangle[1], 'draw_rectangle has invalid x coordinates. Format is draw_rectangle=[xmin, xmax, ymin, ymax]'
    assert draw_rectangle[2] <= draw_rectangle[3], 'draw_rectangle has invalid y coordinates. Format is draw_rectangle=[xmin, xmax, ymin, ymax]'
    
    colormap = np.arange(k_agents)
    if colors is not None:
        assert colors.shape == pos.shape[0:2], f'colors are assigned but shape of color {colors.shape} doesn\'t match the first 2 shape coordinates of pos {pos.shape[0:2]}'


    # Setting correct axis
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))  # constant reference frame
    ax.set_aspect(1)  # spacing the same for x and y direction

    if subplotName is not None:
        ax.set_title(subplotName)

    # Plotting of data (and optional velocity, origin, rectangle)
    if draw_rectangle != [0., 0., 0., 0.]:
        ax.vlines(x=draw_rectangle[0:2],
                    ymin=draw_rectangle[2], ymax=draw_rectangle[3],
                    color='gray', linestyle='--', linewidth=1)
        ax.hlines(y=draw_rectangle[2:4],
                    xmin=draw_rectangle[0], xmax=draw_rectangle[1],
                    color='gray', linestyle='--', linewidth=1)
    if origin_color is not None:
        ax.scatter(0., 0., c=origin_color, marker='+')
    if vel is not None:
        ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1],
                    pivot='tail', width=0.002, angles='xy', scale_units='xy', scale=inverse_arrow_scale_factor, color='#c0c0c0')
    if colors is None:
        ax.scatter(pos[:, 0], pos[:, 1], c=colormap, cmap='tab10', marker=".")
    else:
        ax.scatter(pos[:, 0], pos[:, 1], color=colors[timestep])






def visualize_2d_ffmpeg_multi(poss, vels=None, axis=[-1.,2.,-1.,2.], T_max=0, dt=0, img_save_path='H:/dump/', fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None, colors=None, file_identifier="", plottitles=None, res=(19.20, 10.80), order=None):
    '''
    Visualizes a timeseries of 2d positions.

            Parameters:
                    pos (numpy.array): An array of all positions for all timesteps.
                    Shape should be (<number of timesteps>, <number of individuals>, 2)
                    vel (numpy.array): An array of all velocities for all timesteps
                    Shape should be (<number of timesteps>, <number of individuals>, 2)
                    xmin, xmax, ymin, ymax (float): Limits for the x- and y-axis
                    save_anim: unused parameter
                    T_max (float): end of time domain
                    dt (float): time step size
                    arrow_scale_factor (float): scale factor for arrows. Works inversely, e.g. 2 halfs the arrows. Set 0 for auto
                    img_save_path (string): the path where the frames and animation are saved. Frames may be deleted. Must end in '/'
                    fps (int): Fps. Must be int. Reccommended values: 5-60
                    delete_frames (bool): whether the frames should be deleted after the animation is exported
                    show_plots (bool): whether the plots should be shown in dedicated windows
                    open_anim (bool): whether to open the animation file (in your default program) when done
                    draw_rectangle (list of 4 ints): plots a square at [xmin, xmax, ymin, ymax]. For example [0,1,0,1] gives the unit square.
                    origin_color (string): if None, the origin is not plotted. Otherwise, the origin is plotted. Examples: 'cyan', 'r'
            Returns:
                    The created animation.
    '''
    if vels is not None:
        assert poss.shape[:2] == vels.shape[:2]
    plotgrid = poss.shape[:2]

    n_frames, k_agents = poss.shape[2:4]

    assert img_save_path[-1] == '/', 'folder path "img_save_path" needs to terminate with a slash ("/")'
    assert type(fps) == int, 'fps needs to be of int type.'
    assert fps >= 1
    duration_m, duration_s = divmod(n_frames/fps, 60)
    print(f'Animation Parameters:\n\tfps = {fps}\n\tdur = {duration_m:2.0f}:{duration_s:02.0f}')

    print(axis)

    if order==None:
        order=[t for t in range(n_frames)]
    assert len(order) == n_frames

    if vels is not None and inverse_arrow_scale_factor == 0:
        maxVel = np.max(np.linalg.norm(vels, axis=2).ravel())
        inverse_arrow_scale_factor = min([xmax-xmin, ymax-ymin]) * 0.2/maxVel

    plt.style.use('dark_background')
    plt.rcParams["font.family"] = "monospace"
    colormap = np.arange(k_agents)
    
    hasName = True
    if file_identifier == "":
        file_identifier = str(int(time.time()) % 100000000)
        hasName = False
    print(f'file_identifier: {file_identifier}')
    framelist = []
    framesfolder = img_save_path+f'frames_{file_identifier}/'
    os.makedirs(framesfolder)
    print(framesfolder)

    time_start = time.time()

    progress_counter = 0
    for t in order:
        # Progress update & Loading percentages
        if progress_counter != 0:
            time_now = time.time()
            time_passed = time_now - time_start
            eta_total = (time_now - time_start) * n_frames/progress_counter
            eta_left = eta_total - time_now + time_start
            print(
                f'generating images\t{str(progress_counter).rjust(len(str(n_frames-1)))} / {n_frames}\t({t/progress_counter:.0%})\tTime: passed {time_passed/60:.1f} min,  remaining {eta_left/60:.1f} min,  total {eta_total/60:.1f} min', end="\r")
        progress_counter += 1

        fig, axs = plt.subplots(*plotgrid)
        fig.set_size_inches(*res)

        #Setting suptitle
        suptitle = f'Timestep{str(t).rjust(len(str(n_frames-1)))}/{n_frames-1}'
        '''mins = np.array([xmin, ymin])
        maxs = np.array([xmax, ymax])
        agents_on_screen = np.sum(
            (np.all(poss[img_x, img_y, t, ...] >= mins, 1) & np.all(poss[img_x, img_y, t, ...] <= maxs, 1)))
        subplottitle = f'a.v. {str(agents_on_screen).rjust(len(str(k_agents)))}/{k_agents}  '
        '''
        if T_max != 0:
            suptitle += f'  T={T_max}'
        if dt != 0:
            suptitle += f'  \u0394t={dt}'
        if vels is not None:
            suptitle += f'  arrow factor={1/inverse_arrow_scale_factor:.0%}'
        fig.suptitle(suptitle)

        for img_x, img_y in np.ndindex(plotgrid):

            xmin=axis[img_x][img_y][0]
            xmax=axis[img_x][img_y][1]
            ymin=axis[img_x][img_y][2]
            ymax=axis[img_x][img_y][3]
            # Setting subplot title
            subplottitle = plottitles[img_x][img_y]
            plotOneFrame(fig, axs[img_x, img_y], poss[img_x, img_y, t, ...], vel=None, timestep=t, numtimesteps=n_frames, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, T_max=T_max, dt=dt, inverse_arrow_scale_factor=inverse_arrow_scale_factor, draw_rectangle=draw_rectangle, origin_color=origin_color, colors=colors, subplotName=subplottitle)


        # Saving images
        plt.savefig(framesfolder+f'{t:08}.png')
        framelist += [framesfolder+f'{t:08}.png']
        if show_plots:
            plt.show()
        else:
            plt.close()

    print("\n")
    # ffmpeg .mov
    subprocess.run(['ffmpeg',  '-framerate', str(fps),
                    '-i', framesfolder+f'%08d.png',
                    f'anim_{file_identifier}.mov'], cwd=img_save_path)

    # ffmpeg .mp4
    #subprocess.run(['ffmpeg', 
    #                '-i', framesfolder+f'%08d.png', 
    #                '-c:v', 'libx264', '-r', str(fps), 
    #                f'anim_{file_identifier}.mp4'], cwd=img_save_path)
    print()

    if delete_frames:
        print(f'Deleting {len(framelist)} files:\n\tfrom\t{framelist[0]}\n\tto\t{framelist[-1]}')
        for framename in framelist:
            os.remove(framename)
        print(f'Deleting empty folder:  {framesfolder}')
        os.rmdir(framesfolder)
    if open_anim:
        print(f'Opening file:\t{img_save_path}anim_{file_identifier}.mov')
        os.startfile(img_save_path+f'anim_{file_identifier}.mov')