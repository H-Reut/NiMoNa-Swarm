import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def visualize_2d(pos, vel=None, xmin=-5, xmax=5, ymin=-5, ymax=5, save_anim=False, filename='', interval=0., format='.gif'):
    '''
    Visualizes a timeseries of 2d positions.

            Parameters:
                    pos (numpy.array): An array of all positions for all timesteps.
                    Shape should be (<number of timesteps>, <number of individuals>, 2)
                    vel (numpy.array): An array of all velocities for all timesteps
                    Shape should be (<number of timesteps>, <number of individuals>, 2)
                    xmin, xmax, ymin, ymax (float): Limits for the x- and y-axi
                    save_anim (bool): whether the animation should be saved as a gif
            Returns:
                    The created animation.
    '''
    assert (pos.ndim == 3), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    assert (pos.shape[2] == 2), 'Positions need to have the shape (number of timesteps) x (number of individuals) x 2'
    if vel is not None:
        assert (pos.shape == vel.shape), 'Velocities need to have the same shape as the positions'

    assert xmin < xmax, 'xmin needs to be smaller than xmax'
    assert ymin < ymax, 'ymin needs to be smaller than ymax'

    k_steps, n_agents, dim = pos.shape
    assert dim == 2


    plt.rcParams["font.family"] = "monospace"
    fig, ax = plt.subplots()
    #fig.set_size_inches((19.20, 10.80))
    #fig.tight_layout()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))  # constant reference frame
    ax.set_aspect(1)  # spacing the same for x and y direction
    ax.set_title(f'Timestep {str(0).rjust(len(str(k_steps-1)))}/{k_steps-1}')
    plot = ax.scatter(pos[0, :, 0], pos[0, :, 1], marker='.', cmap='tab10', c=np.arange(n_agents))#marker='o', c='b')#, c='black')  # plot first timestep
    if vel is not None:
        maxVel = np.max(np.linalg.norm(vel,axis=2).ravel())
        scaleFactor = min([xmax-xmin, ymax-ymin]) * 0.2/maxVel
        vectors = ax.quiver(pos[0, :, 0], pos[0, :, 1], vel[0, : ,0], vel[0, : ,1],
                            pivot='tail', width=0.002, angles='xy',
                            scale_units='xy', scale=scaleFactor)

    #fig.tight_layout()
    #fig.set_size_inches((19.20, 10.80))
    def anim(frame):
        plot.set_offsets(pos[frame, :, :])  # update positions in each frame
        if vel is not None:
            vectors.set_offsets(pos[frame, :, :])
            vectors.set_UVC(vel[frame, :, 0], vel[frame, :, 1])
        ax.set_title(f'Timestep {str(frame).rjust(len(str(k_steps-1)))}/{k_steps-1}')

    duration = k_steps * interval / 1000
    print(f'animation duration: {duration:.1f} sec')
    if interval == 0.:
        interval = (duration*1000)//k_steps
    anim_created = FuncAnimation(fig, anim, frames=k_steps, interval=interval, repeat=True)
    #plt.show()
    if save_anim:
        if filename == '':
            filename = str(input('Enter filename for the animation or <no> if you dont want to save: '))
        if not filename == 'no' and not filename == '':
            if format == '.gif':
                anim_created.save(filename + '.gif', writer=PillowWriter(fps=50))
            if format == '.mp4':
                anim_created.save(filename + '.mp4')#, writer=FFMpegWriter(fps=50))
            else:
                assert format[0] == '.'
                anim_created.save(filename + format)

    plt.close()
    return anim_created