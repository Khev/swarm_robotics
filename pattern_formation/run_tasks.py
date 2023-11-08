import os
import logging
import time
import numpy as np
from utils import header_print

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# Args
Jxmin, Jxmax, Jxnum = -2,2,1
Jymin, Jymax, Jynum = -2,2,1
Kmin, Kmax, Knum = -2,2,1
alpha_min, alpha_max, alpha_num = 0, np.pi/2.0, 1
betax_min, betax_max, betax_num = 0, np.pi/2.0, 1
betay_min, betay_max, betay_num = 0, np.pi/2.0, 1
pmin, pmax, pnum = -1, 1, 2
qmin, qmax, qnum =  0, 2, 2
k = 100 # kmean value

# Make data
header = f'Starting simulation'
header_print(header) 
start_time = time.time()  # Get the current time in seconds since the Epoch
cmd_train = (
    f'python train.py --parallel '
    f'--Jx_min {Jxmin} '
    f'--Jx_max {Jxmax} '
    f'--Jx_num_par {Jxnum} '
    f'--Jy_min {Jymin} '
    f'--Jy_max {Jymax} '
    f'--Jy_num_par {Jynum} '
    f'--K_min {Kmin} '
    f'--K_max {Kmax} '
    f'--K_num_par {Knum} '
    f'--alpha_min {alpha_min} '
    f'--alpha_max {alpha_max} '
    f'--alpha_num_par {alpha_num} '
    f'--betax_min {betax_min} '
    f'--betax_max {betax_max} '
    f'--betax_num_par {betax_num} '
    f'--betay_min {betay_min} '
    f'--betay_max {betay_max} '
    f'--betay_num_par {betay_num} '
    f'--p_min {pmin} '
    f'--p_max {pmax} '
    f'--p_num_par {pnum} '
    f'--q_min {qmin} '
    f'--q_max {qmax} '
    f'--q_num_par {qnum}'
    )
os.system(cmd_train)

# Do clustering
header = f'Starting clustering'
header_print(header) 
cmd_cluster = (f'python clustering.py --k={k}')
os.system(cmd_)

end_time = time.time()  # Get the current time in seconds since the Epoch
elapsed_time = end_time - start_time
logger.info(f"Finished simulation for N")
logger.info(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
