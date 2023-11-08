import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import circvar
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min

# def initialize_theta(n, n_groups):
#     """
#     Initialize theta values drawn from n_groups distinct values spaced equally on [0, 2*pi].
    
#     :param n: Number of swarmalators.
#     :param n_groups: Number of distinct theta groups.
#     :return: Initialized theta values.
#     """
#     if n % n_groups != 0:
#         raise ValueError("n must be divisible by n_groups for equal distribution.")
    
#     # Create the distinct theta values spaced equally on [0, 2*pi]
#     distinct_theta_values = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    
#     # Assign each group an equal share of the distinct theta values
#     theta0 = np.repeat(distinct_theta_values, n // n_groups)
#     np.random.shuffle(theta0)  # Shuffle to ensure random distribution across the swarmalators

#     return theta0

def initialize_theta(n, n_groups):
    """
    Initialize theta values drawn from n_groups distinct values spaced equally on [0, 2*pi].
    
    :param n: Number of swarmalators.
    :param n_groups: Number of distinct theta groups.
    :return: Initialized theta values.
    """
    # Create the distinct theta values spaced equally on [0, 2*pi]
    distinct_theta_values = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    
    # Calculate the base count of swarmalators per group
    base_count = n // n_groups
    
    # Calculate the remainder to distribute
    remainder = n % n_groups
    
    # Initialize an empty array for theta values
    theta0 = np.array([])
    
    # Distribute the base count to all groups
    for theta in distinct_theta_values:
        theta0 = np.append(theta0, np.full(base_count, theta))
    
    # Distribute the remainder one by one to the groups until none left
    extra_thetas = np.repeat(distinct_theta_values[:remainder], 1)
    theta0 = np.append(theta0, extra_thetas)
    
    # Shuffle to ensure random distribution across the swarmalators
    np.random.shuffle(theta0)

    return theta0

def euler(z, F, dt, *args):
    """
    Euler integration step.
    
    :param z: Numpy array of the current state variables.
    :param F: Function that calculates the derivatives (RHS of the ODEs).
    :param dt: Time step for the integration.
    :param args: Additional arguments required by the function F.
    :return: Numpy array of the state variables after an Euler step.
    """
    k1 = F(z, *args)
    z_new = z + dt * k1
    return z_new

def rk4(z, F, dt, *args):
    """
    Fourth-order Runge-Kutta integration step.
    
    :param z: Numpy array of the current state variables.
    :param F: Function that calculates the derivatives (RHS of the ODEs).
    :param dt: Time step for the integration.
    :param args: Additional arguments required by the function F.
    :return: Numpy array of the state variables after a Runge-Kutta step.
    """
    k1 = F(z, *args)
    k2 = F(z + 0.5 * dt * k1, *args)
    k3 = F(z + 0.5 * dt * k2, *args)
    k4 = F(z + dt * k3, *args)
    z_new = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_new

def rhs(z, n, Jx, Jy, K, alpha, betax, betay, a, p, q, n_groups):

    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if len(z) != 3 * n:
        raise ValueError("z must have exactly 3*n elements.")
    
    # Extract x, y, and theta components
    x, y, theta = z[:n], z[n:2*n], z[2*n:]

    # Compute pairwise differences
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta

    # Calculate powers of distance
    dist_sq = xd**2 + yd**2
    dist = np.sqrt(dist_sq + 1e-12)  # Add a small number to prevent division by zero
    dist_p = dist ** p
    dist_q = dist ** q

    # Set the diagonals to zero to avoid self-interaction
    np.fill_diagonal(dist_p, 0)
    np.fill_diagonal(dist_q, 0)

    # Compute the alignment factors
    alignment_factor_x = (1 + Jx * np.cos(theta_d - betax))
    alignment_factor_y = (1 + Jy * np.cos(theta_d - betay))

    # Compute the RHS of the equations
    with np.errstate(divide='ignore', invalid='ignore'):
        x_rhs = -xd * (alignment_factor_x / dist_p - a / dist_q)
        y_rhs = -yd * (alignment_factor_y / dist_p - a / dist_q)
        theta_rhs = -K * np.sin(theta_d - alpha) / dist_p

    # Set the diagonals to zero
    np.fill_diagonal(x_rhs, 0)
    np.fill_diagonal(y_rhs, 0)
    np.fill_diagonal(theta_rhs, 0)

    # Sum over the second axis, ignoring self-interactions
    x_next = np.sum(x_rhs, axis=1) / n
    y_next = np.sum(y_rhs, axis=1) / n
    theta_next = np.sum(theta_rhs, axis=1) / n

    return np.concatenate((x_next, y_next, theta_next))


def rhs_alphaij(z, n, Jx, Jy, K, alpha, betax, betay, a, p, q, n_groups):

    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if len(z) != 3 * n:
        raise ValueError("z must have exactly 3*n elements.")
    
    # Extract x, y, and theta components
    x, y, theta = z[:n], z[n:2*n], z[2*n:]

    # Create masks for the upper and lower triangles
    upper_triangle_mask = np.tri(n, k=-1)  # 1s in the upper triangle, 0s in the diagonal and lower triangle
    lower_triangle_mask = upper_triangle_mask.T  # Transpose to get the lower triangle mask

    # Modify alpha, betax, betay depending on the indices i and j
    alpha_matrix = np.full((n, n), alpha)  # Matrix filled with alpha
    betax_matrix = np.full((n, n), betax)  # Matrix filled with betax
    betay_matrix = np.full((n, n), betay)  # Matrix filled with betay

    # Flip signs in the upper triangle
    alpha_matrix -= 2 * alpha * upper_triangle_mask
    betax_matrix -= 2 * betax * upper_triangle_mask
    betay_matrix -= 2 * betay * upper_triangle_mask

    # Compute pairwise differences
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta

    # Calculate powers of distance
    dist_sq = xd**2 + yd**2
    dist = np.sqrt(dist_sq + 1e-12)  # Add a small number to prevent division by zero
    dist_p = dist ** p
    dist_q = dist ** q

    # Set the diagonals to zero to avoid self-interaction
    np.fill_diagonal(dist_p, 0)
    np.fill_diagonal(dist_q, 0)

    # Compute the alignment factors
    alignment_factor_x = (1 + Jx * np.cos(theta_d - betax_matrix))
    alignment_factor_y = (1 + Jy * np.cos(theta_d - betay_matrix))

    # Compute the RHS of the equations
    with np.errstate(divide='ignore', invalid='ignore'):
        x_rhs = -xd * (alignment_factor_x / dist_p - a / dist_q)
        y_rhs = -yd * (alignment_factor_y / dist_p - a / dist_q)
        theta_rhs = -K * np.sin(theta_d - alpha_matrix) / dist_p

    # Set the diagonals to zero
    np.fill_diagonal(x_rhs, 0)
    np.fill_diagonal(y_rhs, 0)
    np.fill_diagonal(theta_rhs, 0)

    # Sum over the second axis, ignoring self-interactions
    x_next = np.sum(x_rhs, axis=1) / n
    y_next = np.sum(y_rhs, axis=1) / n
    theta_next = np.sum(theta_rhs, axis=1) / n

    return np.concatenate((x_next, y_next, theta_next))

def scatter_plot(x, y, theta, L=2):
    """
    Make a scatter plot of points in the (x, y) plane where points are colored according to their phase.
    
    :param x: array-like, the x-coordinates of the points.
    :param y: array-like, the y-coordinates of the points.
    :param theta: array-like, the phases of the points used for coloring.
    """

    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)
    ax.set_xlim([-L,L])
    ax.set_ylim([-L,L])
    
    norm = mcolors.Normalize(vmin=0, vmax=2*np.pi)
    scatter = ax.scatter(x, y, c=np.mod(theta,2*np.pi), s=200, alpha=0.9, cmap='gist_rainbow', norm=norm)
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('theta', fontsize=16)  # Add label to colorbar
    plt.show()



def rhs_deprecated(z, n, Jx, Jy, K, alpha, betax, betay, a, p, q, n_groups):
    """
    Computes the right-hand side of the swarmalator differential equations.

    :param z: An array of shape (3*n,) where the first n entries are x positions,
              the second n are y positions, and the third n are theta values.
    :param Jx: Coupling strength for x component.
    :param Jy: Coupling strength for y component.
    :param K: Coupling strength for theta component.
    :param alpha: Phase lag parameter.
    :param betax: Phase lag for x component.
    :param betay: Phase lag for y component.
    :param n: Number of swarmalators.
    :return: The concatenated derivatives of x, y, and theta.
    """
    
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if len(z) != 3 * n:
        raise ValueError("z must have exactly 3*n elements.")
    
    # Extract x, y, and theta components
    x, y, theta = z[:n], z[n:2*n], z[2*n:]
    
    # Compute pairwise differences
    xd = x[:, np.newaxis] - x
    yd = y[:, np.newaxis] - y
    theta_d = theta[:, np.newaxis] - theta
    
    # Calculate inverse distance squared, handle small numbers to avoid division by zero
    dist_sq = xd**2 + yd**2
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse_dist_sq = np.where(dist_sq > 1e-12, 1.0 / dist_sq, 0.0)
    
    # Set the diagonals to zero
    np.fill_diagonal(inverse_dist_sq, 0)
    
    # Compute the RHS of the equations
    factor = (1 + Jx * np.cos(theta_d - betax))
    x_rhs = -xd * (factor * np.sqrt(inverse_dist_sq) - inverse_dist_sq)
    y_rhs = -yd * (factor * np.sqrt(inverse_dist_sq) - inverse_dist_sq)
    theta_rhs = -K * np.sin(theta_d - alpha) * np.sqrt(inverse_dist_sq)
    
    # Sum over the second axis, ignoring self-interactions, and normalize
    x_next = np.sum((1 - np.eye(n)) * x_rhs, axis=1) / n
    y_next = np.sum((1 - np.eye(n)) * y_rhs, axis=1) / n
    theta_next = np.sum((1 - np.eye(n)) * theta_rhs, axis=1) / n
    
    return np.concatenate((x_next, y_next, theta_next))



def featurize(z):
    # Assuming z is structured as [x1,...,xn, y1,...,yn, theta1,...,thetan]
    n = len(z) // 3
    x = np.array(z[:n])
    y = np.array(z[n:2*n])
    theta = np.array(z[2*n:])

    # Calculate the centroid of the swarmalators
    centroid = np.array([np.mean(x), np.mean(y)])
    positions = np.stack((x, y), axis=-1)

    # Calculate radial distances from the centroid
    radial_distances = np.linalg.norm(positions - centroid, axis=1)
    
    # Feature 1: Radial Variance
    radial_variance = np.var(radial_distances)

    # Feature 2: Circular Variance of theta
    theta_circular_variance = circvar(theta)
    
    # Feature 3: Density (Estimate based on the number of points in a given area)
    # Here we use a simple estimation by taking the inverse of the average distance between points
    avg_distance = np.mean([np.linalg.norm(p1-p2) for i, p1 in enumerate(positions[:-1]) for p2 in positions[i+1:]])
    density = 1 / avg_distance if avg_distance != 0 else 0
    
    # Feature 4: Radius of Gyration
    radius_of_gyration = np.sqrt(np.mean(np.square(radial_distances)))

    # Feature 5: Order pars
    Z = np.mean( np.exp(1j*theta) )
    phi = np.arctan2(x,y)
    Wp, Wm = np.mean( np.exp(1j*(phi+theta)) ), np.mean( np.exp(1j*(phi-theta)) )
    R, Sp, Sm = np.abs(Z), np.abs(Wp), np.abs(Wm)

    # Return the feature vector for this state
    return np.array([radial_variance, theta_circular_variance, density, radius_of_gyration, R, Sp, Sm])


def find_representatives(Z, kmeans_model):
    # Get the centroids of the clusters
    centroids = kmeans_model.cluster_centers_

    # Find the index of the closest point to each centroid
    closest, _ = pairwise_distances_argmin_min(centroids, Z)
    
    # `closest` now contains the index in Z of the closest point to each centroid
    return closest


def filter_data(zs, params, filter_cutoff=6):
    filtered_zs = []
    filtered_params = []
    for z, p in zip(zs, params):
        n = len(z) // 3
        x, y = z[:n], z[n:2*n]
        if max(x) <= filter_cutoff and max(y) <= filter_cutoff:
            filtered_zs.append(z)
            filtered_params.append(p)
    return filtered_zs, filtered_params


def do_kmeans(data, k, filter_cutoff=6):
    
    # Extract the parameters and corresponding z values
    all_params = data['all_params']
    zs = data['all_zs']

    # Filter data based on the cutoff
    zs, all_params = filter_data(zs, all_params, filter_cutoff=filter_cutoff)
    
    features = [featurize(z) for z in zs]
    feature_matrix = np.array(features)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    
    # Compute the silhouette score
    silhouette = silhouette_score(feature_matrix, cluster_labels)
    print(f"K-Means Silhouette Score: {silhouette:.2f}")
    
    # Find the most representative members
    representative_idxs = find_representatives(feature_matrix, kmeans)
    z_representatives = [zs[idx] for idx in representative_idxs]
    params_representatives = [all_params[idx] for idx in representative_idxs]
    
    return cluster_labels, z_representatives, params_representatives


# Define a color class to print colorful text
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def fancy_print(message, color=Colors.OKGREEN, endchar='\n'):
    print(color + message + Colors.ENDC, end=endchar)

def header_print(string):
    length = max(50, len(string))
    print('\n')
    fancy_print('=' * length, Colors.HEADER)
    fancy_print(string, Colors.OKGREEN)
    fancy_print('=' * length, Colors.HEADER)
