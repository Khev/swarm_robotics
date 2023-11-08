import argparse
import os
import itertools
import logging
import json
import time
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from utils import rk4, scatter_plot, rhs, initialize_theta, rhs_alphaij

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def simulate(dt, T, n, Jx, Jy, K, alpha, betax, betay, a, p, q, n_groups):
    L = 2
    x0, y0 = np.random.uniform(-L,L,n), np.random.uniform(-L,L,n)
    theta0 = initialize_theta(n, n_groups)
    z = np.concatenate((x0,y0,theta0))
    NT = int(T/dt)
    args = (n, Jx, Jy, K, alpha, betax, betay, a, p, q, n_groups)
    for t in range(NT):
        z = rk4(z, rhs_alphaij, dt, *args)
    return z

def simulate_wrapper(args):
    return simulate(*args)


def save_data(all_zs, all_params, args):

    # Create the data directory
    current_date = datetime.now().strftime('%Y-%m-%d')
    data_directory = f'data/{current_date}/'
    os.makedirs(data_directory, exist_ok=True)
    
    # Create a base filename using some of the args attributes
    base_filename = f"swarm_simulation_n_pars_{len(all_params)}"
    
    # Prepare the data dictionary
    data_dict = vars(args).copy()  # Copy all arguments
    data_dict.update({
        'all_params': [list(param) for param in all_params],  # Convert tuple to list for JSON serialization
        'all_zs': all_zs                                      # Include all z values
    })
    
    # Save to JSON file
    json_fname = os.path.join(data_directory, f"{base_filename}.json")
    with open(json_fname, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    logging.info(f'Data saved to {json_fname}')


def main(args):

    # Generate parameter ranges
    Jx_range = np.linspace(args.Jx_min, args.Jx_max, args.Jx_num_par)
    Jy_range = np.linspace(args.Jy_min, args.Jy_max, args.Jy_num_par)
    K_range = np.linspace(args.K_min, args.K_max, args.K_num_par)
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_num_par)
    betax_range = np.linspace(args.betax_min, args.betax_max, args.betax_num_par)
    betay_range = np.linspace(args.betay_min, args.betay_max, args.betay_num_par)
    a_range = np.linspace(args.a_min, args.a_max, args.a_num_par)
    p_range = np.linspace(args.p_min, args.p_max, args.p_num_par)
    q_range = np.linspace(args.q_min, args.q_max, args.q_num_par)
    ngroup_range = np.linspace(args.ngroup_min, args.ngroup_max, args.ngroup_num_par)
    ngroup_range = [int(n) for n in ngroup_range]
    ngroup_range = [2,3,4, args.n]

    # All combinations of parameters
    all_params = list(itertools.product(
        Jx_range, Jy_range, K_range, alpha_range, 
        betax_range, betay_range, a_range, p_range, q_range, ngroup_range
    ))

    # Filter out cases where Jy < Jx, betay < betax, and p > q
    all_params = [
        params for params in all_params
        if params[1] >= params[0] and params[5] >= params[4] and params[7] <= params[8]
    ]

    logging.info(f'Starting simulations with {len(all_params)} different parameter combinations.')
    all_zs = []
    start_time = time.time()
    total_params = len(all_params)

    # Make data
    if args.parallel:
        # Use multiprocessing Pool
        with Pool(processes=args.processes) as pool:
            all_args = [(args.dt, args.T, args.n, *params) for params in all_params]
            for i, z_values in enumerate(pool.imap_unordered(simulate_wrapper, all_args)):
                all_zs.append(z_values.tolist())
                param_end_time = time.time()
                time_taken = param_end_time - start_time
                logging.info(f'Completed simulation {i+1}/{len(all_params)} in {time_taken/60.0:.2f} mins.')
    else:
        # Sequential processing
        for i, params in enumerate(all_params):
            param_start_time = time.time()
            logging.info(f'Starting simulation {i+1}/{total_params}')
            z_values = simulate(args.dt, args.T, args.n, *params)
            all_zs.append(z_values.tolist())
            param_end_time = time.time()

            time_taken = param_end_time - param_start_time
            logging.info(f'Completed simulation {i+1}/{total_params} in {time_taken:.2f} seconds.')
            average_time_per_param = (param_end_time - start_time) / (i + 1)
            estimated_time_left = average_time_per_param * (total_params - i - 1)
            logging.info(f'Estimated time remaining: {estimated_time_left/60.0/60.0:.2f} hours.')

    # Save data
    save_data(all_zs, all_params, args)
    

    return all_zs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the swarmalator simulation.")
    parser.add_argument('--dt', type=float, default=0.25, help='Time step size')
    parser.add_argument('--T', type=float, default=200, help='Total time for simulation')
    parser.add_argument('--n', type=int, default=100, help='Number of swarmalators')

    num_par = 1

    # Jx
    parser.add_argument('--Jx_min', type=float, default=-2.0, help='Minimum value for Jx')
    parser.add_argument('--Jx_max', type=float, default=2.0, help='Maximum value for Jx')
    parser.add_argument('--Jx_num_par', type=int, default=11, help='Number of parameters for Jx')

    # Jy
    parser.add_argument('--Jy_min', type=float, default=-2.0, help='Minimum value for Jy')
    parser.add_argument('--Jy_max', type=float, default=2.0, help='Maximum value for Jy')
    parser.add_argument('--Jy_num_par', type=int, default=11, help='Number of parameters for Jy')

    # K
    parser.add_argument('--K_min', type=float, default=-2.0, help='Minimum value for K')
    parser.add_argument('--K_max', type=float, default=2.0, help='Maximum value for K')
    parser.add_argument('--K_num_par', type=int, default=11, help='Number of parameters for K')

    # alpha
    parser.add_argument('--alpha_min', type=float, default=0, help='Minimum value for alpha')
    parser.add_argument('--alpha_max', type=float, default=np.pi/2, help='Maximum value for alpha')
    parser.add_argument('--alpha_num_par', type=int, default=5, help='Number of parameters for alpha')

    # betax
    parser.add_argument('--betax_min', type=float, default=0, help='Minimum value for betax')
    parser.add_argument('--betax_max', type=float, default=np.pi/2, help='Maximum value for betax')
    parser.add_argument('--betax_num_par', type=int, default=5, help='Number of parameters for betax')

    # betay
    parser.add_argument('--betay_min', type=float, default=0, help='Minimum value for betay')
    parser.add_argument('--betay_max', type=float, default=np.pi/2, help='Maximum value for betay')
    parser.add_argument('--betay_num_par', type=int, default=5, help='Number of parameters for betay')

    # a
    parser.add_argument('--a_min', type=float, default=1, help='Minimum value for a')
    parser.add_argument('--a_max', type=float, default=1, help='Maximum value for a')
    parser.add_argument('--a_num_par', type=int, default=num_par, help='Number of parameters for a')

    # p
    parser.add_argument('--p_min', type=float, default=1, help='Minimum value for p')
    parser.add_argument('--p_max', type=float, default=1, help='Maximum value for p')
    parser.add_argument('--p_num_par', type=int, default=1, help='Number of parameters for p')

    # q
    parser.add_argument('--q_min', type=float, default=2, help='Minimum value for q')
    parser.add_argument('--q_max', type=float, default=2, help='Maximum value for q')
    parser.add_argument('--q_num_par', type=int, default=1, help='Number of parameters for q')

    # num_groups
    parser.add_argument('--ngroup_min', type=int, default=100, help='Minimum value for q')
    parser.add_argument('--ngroup_max', type=int, default=100, help='Maximum value for q')
    parser.add_argument('--ngroup_num_par', type=int, default=1, help='Number of parameters for q')

    #
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--processes', type=int, default=9, \
                        help='Number of processes to use for parallel processing')

    args = parser.parse_args()

    main(args)





