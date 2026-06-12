import sys
import numpy as np

# Get job index from SLURM
job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Optional: use job_id to seed random generator
np.random.seed(np.random.randint(1,10000))

# Import functions from functions_plus_plus
from functions import ccdA_conc
from functions import simulate_markov_chain
from functions import ccda_odes
from functions_plus_plus import ccdr_odes
from functions_plus_plus import TF_conc_plus_plus
import math as mt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

cys_df = pd.read_csv(r'ECMpy/dfba_results.csv', header = 0, usecols = ['intra_cys_conc_mol'])
cys_all = cys_df['intra_cys_conc_mol'].astype(float).tolist()

# Initial conditions and parameters
initial_conditionsTF = [0, (3*10**-11), (2.5*10**-11), (9*10**-11), (1*10**-11), 0] # R_MRNA, R, R_2, R_4, R_8, TF

n_steps = range(150000) #how long the simulation runs for
dt = 1 #the timesteps of the simulations

# Run simulation for CcdR
times, R_MRNA_traj, R_traj, R_2_traj, R_4_traj, R_8_traj, TF_traj = TF_conc_plus_plus(initial_conditionsTF, n_steps, dt, cys_all)

# Save results
output_file = f"outputs/simR_{job_id}.npz"
np.savez(output_file, times=times, R_MRNA=R_MRNA_traj, R=R_traj, R_2=R_2_traj, R_4=R_4_traj, R_8=R_8_traj, TF=TF_traj)

data = np.load(f'outputs/simR_{job_id}.npz')

times = data['times']
R_MRNA = data['R_MRNA']
R = data['R']
R_2 = data['R_2']
R_4 = data['R_4']
R_8 = data['R_8']
TF = data['TF']

plt.plot(times[:99999], TF[:99999], label='Transcription Factor Concentration')
plt.plot(times[:99999], R[:99999], label='CcdR Concentration')
plt.plot(times[:99999], R_2[:99999], label='Dimer Concentration')
plt.plot(times[:99999], R_4[:99999], label='Tertramer Concentration')
plt.plot(times[:99999], R_8[:99999], label='Octamer Concentration')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (M)')
plt.legend()
plt.savefig(f'output_figures/sim_{job_id}_TF_graph.png')


