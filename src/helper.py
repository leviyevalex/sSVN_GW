import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def scan(f, init, xs, length=None):
    """ 
    Helper method to debug functions using lax.scan

    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)

def rejection_sampling(iid_samples, lower_bound, upper_bound):
    """ 
    Method which prunes samples outside of defined bounds
    
    """
    truth_table = ((iid_samples > lower_bound) & (iid_samples < upper_bound))
    idx = np.where(np.all(truth_table, axis=1))[0]
    print('%i samples obtained from rejection sampling' % idx.shape[0])
    return np.array(iid_samples[idx])

def plot_cross_section(index1, index2, func, n_grid, injection, a, b, labels):
    """ 
    Plots cross sections of func
    
    """
    assert index1 < index2

    N = n_grid ** 2
    d = len(a)

    x = np.linspace(a[index1], b[index1], n_grid)
    y = np.linspace(a[index2], b[index2], n_grid)

    # Get grid x and y positions
    X, Y = np.meshgrid(x, y)

    # Get heat value on (X,Y)
    particle_grid = np.tile(injection, N).reshape(N, d)
    particle_grid[:, index1] = X.flatten()
    particle_grid[:, index2] = Y.flatten()
    Z = func(particle_grid).reshape(n_grid,n_grid)

    # Plot settings
    fig, ax = plt.subplots(figsize = (5, 5))
    cp = ax.contourf(X, Y, Z)
    ax.scatter(injection[index1], injection[index2], s=0.1, marker='x', c='r')
    ax.axis('equal')
    plt.colorbar(cp)
    ax.set_xlabel(labels[index1])
    ax.set_ylabel(labels[index2])
    ax.set_title('Energy cross section')

    # Save cross section
    # filename = str(index1) + str(index2) + '.png'
    # path = os.path.join('cross_sections', filename)
    # fig.savefig(path)

    return fig, ax

def animate_particles(particle_history, frame_stride, fig, ax, save_path):
    """ 
    Animates particle_history of shape (n_iter, N, d)
    """
    n_iter = particle_history.shape[0]
    frames = np.arange(0, n_iter, frame_stride) # Use to animate whole flow
    scat = ax.scatter([], [], marker=".", color='#51AEFF', s=8) # Initial frame (empty scatterplot)

    def update(i):
        particles_x_i = particle_history[i,:,0]
        particles_y_i = particle_history[i,:,1]
        scat.set_offsets(np.c_[particles_x_i, particles_y_i])
        ax.set_title('Gradient flow: Frame %i' % (i))
    anim = FuncAnimation(fig, update, interval=3000 / n_iter, frames=frames, repeat_delay=2000)
    anim.save(save_path, writer='imagemagick')