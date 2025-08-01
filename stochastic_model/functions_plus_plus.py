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
        
        #Appends the history of the chain
        history.append(next_state)
        current_state = next_state

    return history, transitions

# Function to simulate the Markov chain
def simulate_multi_markov_chain(start_state, steps, transition_list):
    current_state = start_state
    history = [current_state]
    transitions=0

    for matrix in transition_list:
        # Get the probabilities for the current state
        probabilities = matrix[states.index(current_state)]

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
        
        #Appends the history of the chain
        history.append(next_state)
        current_state = next_state

    return history, transitions

#The ode controlling the formation and degradation of CcdA
def ccda_odes(t, y, vk_BMRNA):
    #Variable for CcdA expression
    A_MRNA, B_MRNA, A, B, C, C_2=y #Creates a vector for the coupling of the mRNA CcdA system

    #Ensures that the concentrations are non-negative
    A = max(A, 0)
    B = max(B, 0)
    A_MRNA = max(A_MRNA, 0)
    B_MRNA = max(B_MRNA, 0)
    C = max(C, 0)
    C_2 = max(C_2, 0)
    
    #Complex 1 values
    F = 0.2 # Gelens - percentage of decay rate for antitoxin within the complex
    k_Cdeg = 0.00000714972 # Gelens - rate of complex degradation
    k_bind = 0.055 # Gelens - rate of complex formation 0.055

    #Rate constants for CcdA expression
    k_AMRNA = 0.053515 #Guesstimation - CcdA mRNA formation rate real value 0.02751
    k_degM = 0.00203 #From Gelens - mRNA degradation rate
    k_Adeg = 0.00115524 #From Gelens - Degradation rate of CcdA 0.00115524
    k_Atrans =  0.139 #From Gelens - Rate of translation of CcdA mRNA
#0.01834
    #Variable for CcdB expression
    k_BMRNA = vk_BMRNA #Guesstimation - CcdB mRNA formation rate - has arg set to the variable transcription rate based on bound/unbound state
    k_Btrans = 0.033 #From Gelens - CcdB translation rate
    k_Bdeg = 0.00577623 #From Uniprot - Ccdb decay rate real is 0.00577623

    #ODEs controlling the system
    dA_MRNA = k_AMRNA-k_degM*A_MRNA #CcdA mRNA formation
    dB_MRNA = k_BMRNA-k_degM*B_MRNA #CcdB mRNA formation
    dB = k_Btrans*B_MRNA-k_Bdeg*B-k_bind*A*B+k_Cdeg*C+2*F*k_Adeg*C_2+F*k_Adeg*C #CcdB formation
    dA = k_Atrans*A_MRNA-k_Adeg*A-k_bind*A*B+k_Cdeg*C #CcdA formation
    dC = k_bind*A*B+k_Cdeg*C_2-k_Cdeg*C-F*k_Adeg*C-k_bind*C*B-k_Bdeg*C #TA complex formation
    dC_2 = k_bind*B*C-k_Cdeg*C_2-F*k_Adeg*C_2-k_Bdeg*C_2 #TAT complex formation
    return [dA_MRNA, dB_MRNA, dA, dB, dC, dC_2]

def ccdr_odes(t, x):
    R_MRNA, R, R_2, R_4, R_8, TF=x
    
    #Sets the bounds on values - can't be less than 0
    R_MRNA = max(R_MRNA, 0)
    R = max(R, 0)
    R_2 = max(R_2, 0)
    R_4 = max(R_4, 0)
    R_8 = max(R_8, 0)
    TF = max(TF, 0)
    
    #CcdR mRNA values
    
    k_RMRNA = (1*10**-14) #Guess based on tuned CcdA param. - CcdR mRNA formation rate
    k_degM = (0.0203) #From Gelens - mRNA degradation rate
    
    #CcdR values
    
    k_Rtrans = (0.09) #Guess - translation rate of CcdA
    k_Rdeg = (0.00115524) #Guess based on Gelens - degradation rate of Ccdr
    
    #Dimer values
    
    k_d2 = (8.4*10**-18) # complete guess - based on Gao saying they only found tetramer naturally therefore both must have low K_d
    k_2f = 1 #Tuning paramater to tune dimerization
    k_2u = 1 #Tuning paramater to tune dimer use
    
    #Tetramer values
    
    k_d4 = (8.9*10**-9) # complete guess - based on Gao saying they only found tetramer naturally therefore both must have low K_d 1*10**-17
    k_4f = 1 #Tuning paramater to tune tetramerization
    k_4u = 1#Tuning paramater to tune tetramer use
    
    #Octamer values
    
    k_d8 = (7.9*10**-16) # guess - Based on Gao 
    k_8f = 1 #Tuning paramater to tune octamerization
    k_8u = 1 #Tuning paramater to tune octamer use
    
    #Transcription factor values
    
    k_dTF = (1*10**-15) # guess - Based on Gao 
    k_TFf = 1 #Tuning paramater to tune cysteine binding to the octamer
    
    #Cysteine concentration
    
    O = (7*10**-4) # guess - Based on cytotoxic concentration of cysteine taken one order of magnitude lower
    
    #Differential equations
    
    dR_MRNA = k_RMRNA-k_degM*R_MRNA # CcdR mRNA formation
    dR = k_Rtrans*R_MRNA-k_Rdeg*R-((2*k_2f*R**3)/(k_d2+R**2)) # CcdR formation
    dR_2 = ((k_2f*R**3)/(k_d2+R**2))-((k_2u*2*R_2**3)/(k_d4+R_2**2)) # Dimer formation
    dR_4 = ((k_4f*R_2**3)/(k_d4+R_2**2))-((k_4u*2*R_4**3)/(k_d8+R_4**2)) # Tetramer formation
    dR_8 = ((k_8f*R_4**3)/(k_d8+R_4**2))-((k_8u*(R_8*O)**2)/(k_dTF+R_8*O**2)) # Octamer formation
    dTF = ((k_TFf*(R_8*O)**2)/(k_dTF+R_8*O**2)) # Transcription factor formation
          
    return [dR_MRNA, dR, dR_2, dR_4, dR_8, dTF]
          
def TF_conc(initial_conditionsTF, steps, dt):
    
    x = initial_conditionsTF #sets the initial conditions
    
    t_total = 0 #sets initial time to 0
    
    #creates empty sets for data
    t_all = [] 
    R_MRNA_all = []
    R_all = []
    R_2_all = []
    R_4_all = []
    R_8_all = []
    TF_all = []
    
    for step in steps:
        t_start = t_total #defines starting time
        t_end = t_start + dt #defines time step
        
        #sets sol to the outputs of the solved ccdr_odes function at the given time point
        sol = solve_ivp(ccdr_odes, [t_start, t_end], x, method='BDF', t_eval=([t_end]))
        
        #creates the set x using the ouputs of the solver
        x = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]] 
        
        t_total += dt #changes new starting step
        
        #Appends sets with the data from the current time step
        t_all.append(t_total) 
        R_MRNA_all.append(x[0])
        R_all.append(x[1])
        R_2_all.append(x[2])
        R_4_all.append(x[3])
        R_8_all.append(x[4])
        TF_all.append(x[5])
        
    return np.array(t_all), np.array(R_MRNA_all), np.array(R_all), np.array(R_2_all), np.array(R_4_all), np.array(R_8_all), np.array(TF_all)
          
#Solves the CcdA ODES and creates a dataset of the solutions at each time point

def ccdA_conc(initial_conditionsA, start_state, steps, dt, transition_matrix):
    
    
    y = initial_conditionsA # sets the initial conditions
    
    t_total = 0 # sets intial time to 0
    
    #creates empty sets for data
    t_all = []
    A_MRNA_all = []
    B_MRNA_all = []
    A_all = []
    B_all = []
    C_all = []
    C_2_all = []
     
    #Creates a set of the entire simulated markov chain    
    history,_ = simulate_markov_chain(start_state,steps,transition_matrix)

    #Iterates on the set created above and based on the state chooses the expression rate of CcdB
    for state in history:
        k_BMRNA = 0.0339 if state=='unbound' else 0.5055 #0.3846 - when bound changes the expression rate of CcdB to on
        t_start = t_total #sets start time
        t_end = t_start + dt #sets end time
        
        #solves the CcdA ODES and creates a vector of the output data at that time step
        sol = solve_ivp(ccda_odes, [t_start, t_end], y, method='BDF', args=(k_BMRNA,), t_eval=([t_end]))
        y = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]]

        #moves the start time forward one timestep
        t_total += dt
        
        #appends the empty sets with the data of that timestep
        t_all.append(t_total)
        A_MRNA_all.append(y[0])
        B_MRNA_all.append(y[1])
        A_all.append(y[2])
        B_all.append(y[3])
        C_all.append(y[4])
        C_2_all.append(y[5])
        
        #checks whether or not CcdB concentration has exceeded CcdA concentration, indicating growth arrest
        if (y[3])>(y[2]):
            break

    return np.array(t_all), np.array(A_MRNA_all), np.array(B_MRNA_all), np.array(A_all), np.array(B_all), np.array(C_all), np.array(C_2_all), history

def multi_ccdA_conc(initial_conditionsA, start_state, states, dt):
    
    
    y = initial_conditionsA # sets the initial conditions
    
    t_total = 0 # sets intial time to 0
    
    #creates empty sets for data
    t_all = []
    A_MRNA_all = []
    B_MRNA_all = []
    A_all = []
    B_all = []
    C_all = []
    C_2_all = []
     
    #Iterates on the set created above and based on the state chooses the expression rate of CcdB
    for state in states:
        k_BMRNA = 0.0339 if state=='unbound' else 0.5055 #0.3846 - when bound changes the expression rate of CcdB to on
        t_start = t_total #sets start time
        t_end = t_start + dt #sets end time
        
        #solves the CcdA ODES and creates a vector of the output data at that time step
        sol = solve_ivp(ccda_odes, [t_start, t_end], y, method='BDF', args=(k_BMRNA,), t_eval=([t_end]))
        y = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]]

        #moves the start time forward one timestep
        t_total += dt
        
        #appends the empty sets with the data of that timestep
        t_all.append(t_total)
        A_MRNA_all.append(y[0])
        B_MRNA_all.append(y[1])
        A_all.append(y[2])
        B_all.append(y[3])
        C_all.append(y[4])
        C_2_all.append(y[5])
        
        #checks whether or not CcdB concentration has exceeded CcdA concentration, indicating growth arrest
        if (y[3])>(y[2]):
            break

    return np.array(t_all), np.array(A_MRNA_all), np.array(B_MRNA_all), np.array(A_all), np.array(B_all), np.array(C_all), np.array(C_2_all)

#Big daddy function that puts everything together
def stoc_mod(initial_conditionsA, initial_conditionsTF, start_state, steps, dt):
    translist = []
    TF_arr = TF_conc(initial_conditionsTF, steps, dt)[6]
    print(TF_arr[-10:-1])
    for TF in TF_arr:
        k_on_TFP = 100000000
        k_off_TFP = 0.01
        T_on = 1-(mt.e**-(TF*k_on_TFP))
        T_off= 1-(mt.e**-(k_off_TFP))
        transition_matrix=np.array([
        [ 1-T_on , T_on ],
        [ T_off , 1-T_off ]
        ])
        
        translist.append(transition_matrix)
     
    transition_list = translist
    states, transitions = simulate_multi_markov_chain(start_state, steps, transition_list)
    times, A_MRNA_traj, B_MRNA_traj, A_traj, B_traj, C_traj, C_2_traj = multi_ccdA_conc(initial_conditionsA, start_state, states, dt)
    
    return times, A_MRNA_traj, B_MRNA_traj, A_traj, B_traj, C_traj, C_2_traj, TF_arr, states, transitions
    