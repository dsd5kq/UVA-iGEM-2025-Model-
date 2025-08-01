import sys
import numpy as np

# Get job index from SLURM
job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Optional: use job_id to seed random generator
np.random.seed(job_id)

# Import your model code
from functions import ccdA_conc
from functions import simulate_markov_chain
from functions import ccda_odes
import math as mt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

TFP = 0.000000016
k_on_TFP = 10000000
k_off_TFP = 0.5
P_n = 500

T_on = 1-(mt.e**-(TFP*k_on_TFP))

T_off= 1-(mt.e**-(k_off_TFP))

P = 1-(1-T_on)**P_n

print(T_on)

print(T_off)

# Transition matrix for the Markov chain:
# Rows correspond to current states ['unbound', 'bound']
# Columns correspond to next states ['unbound', 'bound']
# Each entry [i][j] is the probability of transitioning from state i to state j
transition_matrix=np.array([
[ 1-T_on , T_on ],
[ T_off , 1-T_off ]
])

# Initial conditions and parameters
initial_conditions = [0, 0, .2, 0]
start_state = "unbound"
n_steps = range(500)
dt = 1

# Run simulation
times, A_MRNA_traj, B_MRNA_traj, A_traj, B_traj, states = ccdA_conc(initial_conditions, start_state, n_steps, dt, transition_matrix)

# Save results
output_file = f"outputs/sim_{job_id}.npz"
np.savez(output_file, times=times, A=A_traj, B=B_traj, A_MRNA=A_MRNA_traj, B_MRNA=B_MRNA_traj, states=states)