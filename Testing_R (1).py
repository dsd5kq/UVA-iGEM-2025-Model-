import sys
import numpy as np

# Get job index from SLURM
job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Optional: use job_id to seed random generator
np.random.seed(np.random.randint(1,10000))

# Import your model code
from functions import ccdA_conc
from functions import simulate_markov_chain
from functions import ccda_odes
from functionsplus import ccdr_odes
from functionsplus import TF_conc
import math as mt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Initial conditions and parameters
initial_conditionsTF = [0, 0, 0, 0, 0, 0] # R_MRNA, R, R_2, R_4, R_8, TF

n_steps = range(500000) #how long the simulation runs for
dt = 1 #the timesteps of the simulations

# Run simulation for CcdR
times, R_MRNA_traj, R_traj, R_2_traj, R_4_traj, R_8_traj, TF_traj = TF_conc(initial_conditionsTF, n_steps, dt)

# Save results
output_file = f"outputs/simR_{job_id}.npz"
np.savez(output_file, times=times, R_MRNA=R_MRNA_traj, R=R_traj, R_2=R_2_traj, R_4=R_4_traj, R_8=R_8_traj, TF=TF_traj)

data = np.load(f'outputs/simR_{job_id}.npz')