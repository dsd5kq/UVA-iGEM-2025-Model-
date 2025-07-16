import numpy as np
import sys
import math as mt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


states=['unbound','bound']

# Function to simulate the Markov chain
def simulate_markov_chain(start_state, steps, transition_matrix):
    current_state = start_state
    history = [current_state]
    transitions=0

    for t in steps:
        # Get the probabilities for the current state
        probabilities = transition_matrix[states.index(current_state)]

        # Generate a random number and choose the next state
        random_num = np.random.rand()
        cumulative_probability = 0
        next_state = None

        #Chooses the next state in the chain
        for i, prob in enumerate(probabilities):
            cumulative_probability += prob
            if random_num <= cumulative_probability:
                next_state = states[i]
                break
        #Counts the number of state changes
        if next_state!=current_state:
            transitions += 1
        history.append(next_state)
        current_state = next_state

    return history, transitions

#The ode controlling the formation and degradation of CcdA
def ccda_odes(t, y, vk_BMRNA):
    #Variable for CcdA expression
    A_MRNA, B_MRNA, A, B=y #Creates a vector for the coupling of the mRNA CcdA system

    #Ensures that the concentrations are non-negative
    A = max(A, 0)
    B = max(B, 0)
    A_MRNA = max(A_MRNA, 0)
    B_MRNA = max(B_MRNA, 0)

    #Rate constants for CcdA expression
    k_AMRNA = 0.01923 #Guesstimation - CcdA mRNA formation rate real value 0.1923
    k_degM = 0.00203 #From Gelens - mRNA degradation rate
    k_Adeg = 0.00577623 #From Gelens - Degradation rate of CcdA 0.00115524
    k_bind = 2000000 #From Gelens - Binding rate of CcdA and CcdB 2000000
    k_Atrans =  0.139 #From Gelens - Rate of translation of CcdA mRNA

    #Variable for CcdB expression
    k_BMRNA = vk_BMRNA #Guesstimation - CcdB mRNA formation rate
    k_Btrans = 0.033 #From Gelens - CcdB translation rate
    k_Bdeg = 0.00577623 #From Uniprot - Ccdb decay rate real is 0.00577623

    #Fluxes controlling CcdA expression
    v_Atrans = k_Atrans*A_MRNA #Translation flux of CcdA
    v_Adeg = k_Adeg*A #Degradation flux of CcdA
    v_AB = k_bind*A*B #Flux due to the binding of CcdA to CcdB

    #Fluxes controlling CcdB expression
    v_Btrans = k_Btrans*B_MRNA
    v_Bdeg = k_Bdeg*B
    
    #ODEs controlling the system
    dA_MRNA = k_AMRNA-k_degM*A_MRNA
    dB_MRNA = k_BMRNA-k_degM*B_MRNA
    dB = k_Btrans*B_MRNA-k_Bdeg*B-k_bind*A*B 
    dA = k_Atrans*A_MRNA-k_Adeg*A-k_bind*A*B 
    return [dA_MRNA, dB_MRNA, dA, dB]

def ccdA_conc(initial_conditions, start_state, steps, dt, transition_matrix):
    y = initial_conditions
    t_total = 0
    t_all = []
    A_MRNA_all = []
    B_MRNA_all = []
    A_all = []
    B_all = []
    
    history,_ = simulate_markov_chain(start_state,steps,transition_matrix)

    for state in history:
        k_BMRNA = 0 if state=='unbound' else 0.3846 #0.3846
        t_start = t_total
        t_end = t_start + dt
        sol = solve_ivp(ccda_odes, [t_start, t_end], y, method='LSODA', args=(k_BMRNA,), t_eval=([t_end]))
        y = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]

        t_total += dt
        t_all.append(t_total)
        A_MRNA_all.append(y[0])
        B_MRNA_all.append(y[1])
        A_all.append(y[2])
        B_all.append(y[3])
        if (y[3])>(y[2]):
            break

    return np.array(t_all), np.array(A_MRNA_all), np.array(B_MRNA_all), np.array(A_all), np.array(B_all), history