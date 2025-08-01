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
from functions_plus_plus import ccdr_odes
from functions_plus_plus import TF_conc
from functions_plus_plus import stoc_mod
import math as mt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

states=['unbound','bound']

# Initial conditions and parameters
initial_conditionsTF = [0, (3*10**-11), (2.5*10**-11), (9*10**-11), (1*10**-11), 0] # R_MRNA, R, R_2, R_4, R_8, TF

initial_conditionsA = [0, 0, 2500, 0, 0, 0] # A_MRNA, B_MRNA, A, B, C, C_2

steps = range(300000) #how long the simulation runs for
dt = 1 #the timesteps of the simulations

start_state='unbound'

times, A_MRNA_traj, B_MRNA_traj, A_traj, B_traj, C_traj, C_2_traj, TF_traj, states, transitions = stoc_mod(initial_conditionsA, initial_conditionsTF, start_state, steps, dt)

# Save results
output_file = f"outputs/sim_{job_id}.npz"
np.savez(output_file, times=times, A=A_traj, B=B_traj, A_MRNA=A_MRNA_traj, B_MRNA=B_MRNA_traj, C=C_traj, C_2=C_2_traj, states=states)

data = np.load(f'outputs/sim_{job_id}.npz')

# Access arrays:
times = data["times"]
A = data["A"]
B = data["B"]
A_MRNA = data["A_MRNA"]
B_MRNA = data["B_MRNA"]
C = data["C"]
C_2 = data["C_2"]
states = data["states"]

fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(8, 6))

# Plot mRNA concentrations
#plt.plot(times, A_MRNA, label='CcdA mRNA')
#plt.plot(times, B_MRNA, label='CcdB mRNA')

#Plot protein concentrations
axs[0].plot(times, A, label='CcdA protein')
axs[0].plot(times, B, label='CcdB protein')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('# of Molecules')
axs[0].set_title('# of Proteins Over Time')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(times, C, label='TA Complex')
axs[1].plot(times, C_2, label='TAT Complex')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('# of Free Molecules')
axs[1].set_title('# of Complex Over Time')
axs[1].grid(True)
axs[1].legend()

fig.tight_layout()  # Reserve space for suptitle
plt.savefig(f'output_figures/sim_{job_id}_graph.jpeg')

print(transitions)