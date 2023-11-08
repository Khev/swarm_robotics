import argparse
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from utils import do_kmeans, scatter_plot  # Ensure these are available in utils.py
from sklearn.cluster import KMeans


def scatter_plot_custom(x, y, theta, L, ax=None):
    """
    Plot the positions (x, y) and orientations (theta) of the entities on a scatter plot.

    Parameters:
    - x: array-like, x-coordinates of the entities
    - y: array-like, y-coordinates of the entities
    - theta: array-like, orientations of the entities
    - L: float, size of the plot area
    - ax: Matplotlib Axes object (optional), the axis on which to draw the plot
    """

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)
    ax.set_xlim([-L,L])
    ax.set_ylim([-L,L])
    ax.set_aspect('equal')

    norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
    scatter = ax.scatter(x, y, c=np.mod(theta,2*np.pi), alpha=0.6, cmap='gist_rainbow', norm=norm)
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical',fraction=0.046, shrink=0.8)
    cbar.set_label('theta', fontsize=16)  # Add label to colorbar

    if ax is None:
        plt.show()


def plot_representatives(z_representatives, args):
    # Maximum number of plots per figure
    max_plots_per_figure = 36  # 6x6 grid
    num_figures = len(z_representatives) // max_plots_per_figure + \
                  (len(z_representatives) % max_plots_per_figure > 0)

    for fig_num in range(num_figures):
        # Create a new figure for each set of 36 representatives
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(30, 30))  # Adjust figsize if needed
        axes = axes.flatten()
        
        # Calculate the start and end index for the current figure
        start_idx = fig_num * max_plots_per_figure
        end_idx = min((fig_num + 1) * max_plots_per_figure, len(z_representatives))
        
        for i, z in enumerate(z_representatives[start_idx:end_idx], start=start_idx):
            ax = axes[i - start_idx]
            n = len(z) // 3
            x, y, theta = z[:n], z[n:2*n], z[2*n:]
            scatter_plot_custom(x, y, theta, args.L, ax=ax)
            ax.set_title(f'Cluster {i}')
            # Additional plotting configurations...

        # Hide any unused subplots if your figure grid is not completely filled
        for j in range(end_idx - start_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        # Save each figure with a unique name
        fname = f'{args.dir_figure}/{args.date}/cluster_representatives_n_{args.n}_k_{args.k}_fig_{fig_num}.png'
        plt.savefig(fname)
        print(f"Figure {fig_num} saved to {fname}")
        plt.close(fig)  # Close the figure to free up memory


def main(args):

    # Load data
    date = args.date
    fname = f'{args.dir_data}/{date}/swarm_simulation_n_pars_{args.n}.json'
    with open(fname, 'r') as f:
        data = json.load(f)

    # Perform k-means clustering
    t1 = time.time()
    cluster_labels, z_representatives, params_representatives = do_kmeans(data, args.k)
    t2 = time.time()
    print(f'Clustering took: {(t2-t1)/60.0/60.0:.2f} hr')

    # Plot results
    plot_representatives(z_representatives, args)

    # Save figure
    dirn = f'{args.dir_figure}{date}'
    os.makedirs(dirn, exist_ok=True)
    fname = f'{args.dir_figure}{date}/k-means_n_{args.n}.png'
    plt.savefig(os.path.join(dirn, f'cluster_representatives_n_{args.n}_k_{args.k}.png'))

    # Save k means data
    par_names = ['Jx','Jy','K','alpha','betax','betay','a','p','q','n_group']
    kmeans_data = {
        'k': args.k,
        'cluster_labels': cluster_labels.tolist(), 
        'z_representatives': z_representatives,
        'par_names': par_names,
        'params_representatives': params_representatives
    }

    dirn = f'{args.dir_data}/{date}'
    os.makedirs(dirn, exist_ok=True)
    fname = f'{dirn}/k-means_data_n_{args.n}_k_{args.k}.json'
    with open(fname, 'w') as f_out:
        json.dump(kmeans_data, f_out, indent=4)
    print(f"K-means data saved to {fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--date', type=str, help='The number of parameters', default='2023-11-08')
    parser.add_argument('--n', type=int, help='The number of parameters', default=625)
    parser.add_argument('--k', type=int, help='The number of clusters to find', default=9)
    parser.add_argument('--L', type=float, help='plot limits', default=2.5)
    parser.add_argument('--dir_figure', type=str, help='Directory to save figures', default='figures/')
    parser.add_argument('--dir_data', type=str, help='Directory to save figures', default='data/')


    args = parser.parse_args()
    main(args)
