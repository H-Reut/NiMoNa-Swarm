import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import subprocess
import time
import os

print("Matplotlib Backend:", matplotlib.get_backend())


def visualize_2d_ffmpeg(pos, vel=None, xmin=-5, xmax=5, ymin=-5, ymax=5, T_max=0, dt=0, img_save_path='H:/dump/', fps=30, delete_frames=True, show_plots=False, open_anim=True, inverse_arrow_scale_factor=1, draw_rectangle=[0., 0., 0., 0.], origin_color=None):
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
    assert img_save_path[
        -1] == '/', 'folder path "img_save_path" needs to terminate with a slash ("/")'
    assert type(fps) == int, 'fps needs to be of int type.'
    assert (pos.ndim == 3), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    assert (
        pos.shape[2] == 2), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    if vel is not None:
        assert (
            pos.shape == vel.shape), 'Velocities need to have the same shape as the positions'

    assert xmin < xmax, 'xmin needs to be smaller than xmax'
    assert ymin < ymax, 'ymin needs to be smaller than ymax'

    assert draw_rectangle[0] <= draw_rectangle[1], 'draw_rectangle has invalid x coordinates. Format is draw_rectangle=[xmin, xmax, ymin, ymax]'
    assert draw_rectangle[2] <= draw_rectangle[3], 'draw_rectangle has invalid y coordinates. Format is draw_rectangle=[xmin, xmax, ymin, ymax]'

    n_steps, k_agents, _ = pos.shape

    if vel is not None and inverse_arrow_scale_factor == 0:
        maxVel = np.max(np.linalg.norm(vel, axis=2).ravel())
        inverse_arrow_scale_factor = min([xmax-xmin, ymax-ymin]) * 0.2/maxVel

    plt.style.use('dark_background')
    plt.rcParams["font.family"] = "monospace"
    colormap = np.arange(k_agents)

    file_identifier = int(time.time()) % 100000000
    print(f'file_identifier: {file_identifier}')
    framelist = []
    framesfolder = img_save_path+f'frames_{file_identifier}/'
    os.makedirs(framesfolder)
    print(framesfolder)

    time_start = time.time()

    for t in range(n_steps):
        # Progress update & Loading percentages
        if t != 0:
            time_now = time.time()
            eta_total = (time_now - time_start) * n_steps/t
            eta_left = eta_total - time_now + time_start
            print(
                f'generating images\t{str(t).rjust(len(str(n_steps-1)))} / {n_steps}\t({t/n_steps:.0%})\tTime left ca. {eta_left/60:.1f} minutes\tTotal time ca. {eta_total/60:.1f} minutes', end="\r")

        # Setting correct axis
        fig, ax = plt.subplots()
        fig.set_size_inches(19.20, 10.80)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))  # constant reference frame
        ax.set_aspect(1)  # spacing the same for x and y direction

        # Setting image title
        mins = np.array([xmin, ymin])
        maxs = np.array([xmax, ymax])
        agents_on_screen = np.sum(
            (np.all(pos[t, :] >= mins, 1) & np.all(pos[t, :] <= maxs, 1)))
        title = f'agents visible {str(agents_on_screen).rjust(len(str(k_agents)))}/{k_agents}    Timestep {str(t).rjust(len(str(n_steps-1)))}/{n_steps-1}  '
        if T_max != 0:
            title += f'  T={T_max}'
        if dt != 0:
            title += f'  \u0394t={dt}'
        if vel is not None:
            title += f'  arrow factor={1/inverse_arrow_scale_factor:.0%}'
        ax.set_title(title)

        # Plotting of data (and optional velocity, origin, rectangle)
        if draw_rectangle != [0., 0., 0., 0.]:
            plt.vlines(x=draw_rectangle[0:2],
                       ymin=draw_rectangle[2], ymax=draw_rectangle[3],
                       color='gray', linestyle='--', linewidth=1)
            plt.hlines(y=draw_rectangle[2:4],
                       xmin=draw_rectangle[0], xmax=draw_rectangle[1],
                       color='gray', linestyle='--', linewidth=1)
        if origin_color is not None:
            plt.scatter(0., 0., c=origin_color, marker='+')
        if vel is not None:
            plt.quiver(pos[t, :, 0], pos[t, :, 1], vel[t, :, 0], vel[t, :, 1],
                       pivot='tail', width=0.002, angles='xy', scale_units='xy', scale=inverse_arrow_scale_factor, color='#c0c0c0')
        plt.scatter(pos[t, :, 0], pos[t, :, 1], c=colormap, cmap='tab10')

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