#!/usr/bin/env python
# coding: utf-8

# Helper Functions 

# setting media conditions 
# - use media conditions from csv to update uptake rates
# - only for initial media conditions 

# In[ ]:
import pandas as pd 
import numpy as np
from cobra import Model, Reaction
import sys

def set_media(model, media_file, column_name):

    #reset all uptake conditions
    for rxn in model.reactions:
        if rxn.id.endswith('_reverse') and rxn.id.startswith('EX_'):
            rxn.upper_bound = 0.0  
    
    df = pd.read_csv(media_file)
    df.columns = df.columns.str.strip()
    for _, row in df.iterrows():
        rxn_id = row['Reverse Reaction']
        uptake_rate = row[column_name]
        
        # skip if any value is missing or not a number
        if pd.isna(rxn_id) or pd.isna(uptake_rate):
            continue
    
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            rxn.upper_bound = uptake_rate 
        except KeyError:
            print(f"Reaction {rxn_id} not found in model, skipping.")
    
    model.reactions.get_by_id('EX_cys__L_e_reverse').upper_bound = 0
    model.reactions.get_by_id('EX_ser__L_e_reverse').upper_bound = 0
    model.medium
    return model 


# lexicographic optimization + run fba 

# In[ ]:


def lexico_optimization(model, epsilon):
    flux_dict = {}
    model.objective = 'BIOMASS_Ec_iML1515_core_75p37M'
    sol1 = model.optimize()
    print("sol1", sol1) 
    z1_opt = sol1.objective_value
    lb = z1_opt * (1 - epsilon)  
    model.reactions.BIOMASS_Ec_iML1515_core_75p37M.lower_bound = lb
    model.objective = 'EX_cys__L_e'
    sol2 = model.optimize() 
    flux_dict = sol2.fluxes.to_dict()
    mu = flux_dict['BIOMASS_Ec_iML1515_core_75p37M']
    return flux_dict, mu


# update biomass concentration 
# -start with initial concentration and update with growth rate 

# In[ ]:


def update_biomass(initial_biomass, mu, dt):
    initial_biomass = initial_biomass * np.exp(mu * dt)
    return initial_biomass


# load initial concentrations of media 

# In[ ]:


def load_initial_concentrations(initial_conc_file):
    df = pd.read_csv(initial_conc_file)
    df.columns = df.columns.str.strip()

    conc_dict = {}
    for _, row in df.iterrows():
        rxn_id = row['Reverse Reaction']
        conc = row['initial conc']
        if pd.notna(rxn_id) and pd.notna(conc):
            conc_dict[rxn_id] = conc
    return conc_dict


# update uptake bounds
# - based on initial concentrations
# - include ode equation based on euler's method
# - reset uptake bounds

# In[ ]:


def update_extracellular_conc(conc_dict, flux_dict, initial_biomass, dt):
    for rxn_id, conc in conc_dict.items():
        flux = flux_dict.get(rxn_id, 0.0)
        delta_S = -flux * initial_biomass * dt
        conc_dict[rxn_id] = max(0, conc + delta_S)
    
    return conc_dict


# update uptake rates based on new concentration 

# In[ ]:


def update_uptake_rates(model, conc_dict, initial_biomass, dt):
    for rxn_id, conc in conc_dict.items(): 
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            vmax = conc / (dt * initial_biomass)
            rxn.upper_bound = vmax
        except KeyError:
            print(f"Warning: {rxn_id} not found in model, skipping.")
    return model


# cysteine sink/accumulation
# - create sink reaction
# - add constraint in model using alpha
# - track over time using dt? 

# In[ ]:


def cysteine_accumulation(model, alpha):
    cys_sink = Reaction("CYSSINK")
    cys_sink.name = "L-Cysteine Sink"
    cys_sink.lower_bound = 0 
    cys_sink.upper_bound = 1000
    
    cys_L = model.metabolites.get_by_id("cys__L_c")
    
    cys_sink.add_metabolites({
        cys_L: -1,
    })
    model.add_reactions([cys_sink])
    return model 


# big daddy func that puts everything together 
# - add concentration values to a ongoing spreadsheet/list to develop graphs from
# - -handle tracking history of biomass concentrations 

# In[ ]:


def dfba_simulation(model, steps, dt, media_file, column_name, initial_conc_file): 
    t_total = 0
    t_all = []
    #add in initial biomass
    initial_biomass = 0.1
    biomass_history =[]
    thiosulfate_uptake = []
    thiosulfate_conc = []
    l_cys_export = []
    #add in tracking cysteine afterwards 
    model = set_media(model, media_file, column_name)
    conc_dict = load_initial_concentrations(initial_conc_file)  #make sure to define this with inputs 
    for step in range(steps):
        flux_dict, mu = lexico_optimization(model, epsilon=0.9)
        
        tsul_flux = flux_dict.get('EX_tsul_e_reverse', 0.0)
        thiosulfate_uptake.append(tsul_flux)
        
        tsul_conc = conc_dict.get('EX_tsul_e_reverse', 0.0)
        thiosulfate_conc.append(tsul_conc)

        lcys_flux = flux_dict.get('EX_cys__L_e', 0.0) 
        l_cys_export.append(lcys_flux) 
        
        conc_dict = update_extracellular_conc(conc_dict, flux_dict, initial_biomass, dt)
        model = update_uptake_rates(model, conc_dict, initial_biomass, dt)
        initial_biomass = update_biomass(initial_biomass, mu, dt)
        biomass_history.append(initial_biomass) 
        t_all.append(t_total)
        t_total += dt
    results_df = pd.DataFrame({
        'time_hr': t_all,
        'biomass': biomass_history,
        'thiosulfate_uptake': thiosulfate_uptake,
        'thiosulfate_concentration': thiosulfate_conc, 
        'l_cys_export': l_cys_export
    })
    return results_df

