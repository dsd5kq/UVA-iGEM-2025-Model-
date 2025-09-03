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
    z1_opt = sol1.objective_value
    lb = z1_opt * (1 - epsilon) 
    #print("biomass", lb)
    model.reactions.BIOMASS_Ec_iML1515_core_75p37M.lower_bound = lb
    model.objective = 'EX_cys__L_e'
    sol2 = model.optimize() 
    flux_dict = sol2.fluxes.to_dict()
    mu = flux_dict['BIOMASS_Ec_iML1515_core_75p37M']
    return flux_dict, mu


# update biomass concentration 
# -start with initial concentration and update with growth rate 

# In[ ]:


def update_biomass(initial_biomass, mu, dt_hr, t_total):

    if t_total < 3600:
            # Lag phase – very slow or no growth
            effective_mu = 0.05 * mu
    elif t_total < 18000:
        # Exponential phase – normal max growth
        effective_mu = mu
    elif t_total < 36000:
        # Stationary phase – slow or zero growth
        effective_mu = 0.01 * mu
    else:
        # Death phase – biomass starts to decline
        effective_mu = -0.05 * mu

    initial_biomass = initial_biomass * np.exp(effective_mu * dt_hr)
    return initial_biomass

# load initial concentrations of media 

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


def update_extracellular_conc(conc_dict, flux_dict, initial_biomass, dt_hr):
    for rxn_id, conc in conc_dict.items():
        flux = flux_dict.get(rxn_id, 0.0)
        delta_S = -flux * initial_biomass * dt_hr
        conc_dict[rxn_id] = max(0, conc + delta_S)
    
    return conc_dict


# update uptake rates based on new concentration 

# In[ ]:


def update_uptake_rates(model, conc_dict, initial_biomass, dt_hr):
    for rxn_id, conc in conc_dict.items(): 
        try:
            rxn = model.reactions.get_by_id(rxn_id)
            vmax = conc / (dt_hr * initial_biomass)
            rxn.upper_bound = vmax
        except KeyError:
            print(f"Warning: {rxn_id} not found in model, skipping.")
    return model


# cysteine sink/accumulation
# - create sink reaction
# - add constraint in model using alpha
# - track over time using dt? 

# In[ ]:


def cysteine_constraint(model, alpha):
    v_sink = model.reactions.get_by_id('CYSSINK').flux_expression
    v_export = model.reactions.get_by_id('EX_cys__L_e').flux_expression
    model.add_cons_vars(model.problem.Constraint(v_sink - alpha * v_export, lb=0, ub=0,
    name="CYSSINK_CONSTRAINT"))
    return model 

def intra_cys_accumulation(flux_dict, initial_biomass, dt_hr): 
    flux = flux_dict.get('CYSSINK', 0.0)
    delta_cys = flux * initial_biomass * dt_hr
    return delta_cys 

def extra_cys_accumulation(flux_dict, initial_biomass, dt_hr): 
    flux = flux_dict.get('EX_cys__L_e', 0.0)
    delta_cys = flux * initial_biomass * dt_hr
    return delta_cys 

def intra_cys_list(intra_cys_mmol): 
    intra_cys_conc = []
    intra_cys_conc = [x / 1000 for x in intra_cys_mmol]
    print(intra_cys_conc) 
    return intra_cys_conc 

# big daddy func that puts everything together 
# - add concentration values to a ongoing spreadsheet/list to develop graphs from
# - -handle tracking history of biomass concentrations 

# In[ ]:


def dfba_simulation(model, steps, dt, media_file, column_name, initial_conc_file): 
    dt_hr = dt / 3600
    t_total = 0
    t_all = []
    #add in initial biomass (assuming in g/L) 
    gDW_frac = 0.35
    initial_biomass = 0.035
    intra_cys_conc = 0.0
    extra_cys_conc = 0.0
    biomass_history_gDW =[]
    thiosulfate_uptake = []
    thiosulfate_conc_mmol = []
    l_cys_export = []
    growth_rate = []
    intra_cys_rate = []
    intra_cys_mmol = []
    extra_cys_mmol = []
    #add in tracking cysteine afterwards 
    model.solver._pending_modifications.add_constr = [
    c for c in model.solver._pending_modifications.add_constr 
    if c.name != "CYSSINK_CONSTRAINT"]
    
    model.solver.update()
    if "CYSSINK_CONSTRAINT" in model.constraints:
        model.remove_cons_vars([model.constraints["CYSSINK_CONSTRAINT"]])
    
    model = set_media(model, media_file, column_name)
    conc_dict = load_initial_concentrations(initial_conc_file)  #make sure to define this with inputs 
    lexico_optimization(model, epsilon=0.7)
    model = cysteine_constraint(model, alpha=0.6)
    for step in range(steps):
        flux_dict, mu = lexico_optimization(model, epsilon=0.7)
        #v_sink = flux_dict.get('CYSSINK', 0.0) 
        #v_export = flux_dict.get('EX_cys__L_e', 0.0) 
        #print(mu) 
        growth_rate.append(mu) 
        intra_cys_flux = flux_dict.get('CYSSINK', 0.0)
        intra_cys_rate.append(intra_cys_flux) 

        delta_intra_cys = intra_cys_accumulation(flux_dict, initial_biomass, dt_hr)
        intra_cys_conc += delta_intra_cys 
        intra_cys_mmol.append(intra_cys_conc) 

        delta_extra_cys = extra_cys_accumulation(flux_dict, initial_biomass, dt_hr) 
        extra_cys_conc += delta_extra_cys 
        extra_cys_mmol.append(extra_cys_conc) 
        
        tsul_flux = flux_dict.get('EX_tsul_e_reverse', 0.0)
        thiosulfate_uptake.append(tsul_flux)
        
        tsul_conc = conc_dict.get('EX_tsul_e_reverse', 0.0)
        thiosulfate_conc_mmol.append(tsul_conc)

        lcys_flux = flux_dict.get('EX_cys__L_e', 0.0) 
        l_cys_export.append(lcys_flux) 
        
        conc_dict = update_extracellular_conc(conc_dict, flux_dict, initial_biomass, dt_hr)
        model = update_uptake_rates(model, conc_dict, initial_biomass, dt_hr)
        initial_biomass = update_biomass(initial_biomass, mu, dt_hr, t_total)
        biomass_history_gDW.append(initial_biomass) 
        t_all.append(t_total)
        t_total += dt

    biomass_history_g = [x / gDW_frac for x in biomass_history_gDW]
    thiosulfate_conc_mol = [x/1000 for x in thiosulfate_conc_mmol]
    intra_cys_mol = [x / 1000 for x in intra_cys_mmol]
    extra_cys_mol = [x/1000 for x in extra_cys_mmol]
    intra_cys_mg_per_L = [x * 121160 for x in intra_cys_mol]
    extra_cys_mg_per_L = [x * 121160 for x in extra_cys_mol]
    growth_rate_s = [mu / 3600 for mu in growth_rate]
    results_df = pd.DataFrame({ 
        'time_s': t_all,
        'growth_rate': growth_rate,
        'growth_rate_s' : growth_rate_s, 
        'biomass_gDW': biomass_history_gDW,
        'biomass_g': biomass_history_g, 
        'l_cys_export': l_cys_export, #this is not in seconds 
        'extra_cys_conc_mol': extra_cys_mol,
        'intra_cys_rate': intra_cys_rate, #not in seconds 
        'intra_cys_conc_mol': intra_cys_mol,
        'intra_cys_mg_L': intra_cys_mg_per_L, 
        'extra_cys_mg_L': extra_cys_mg_per_L,
        'thiosulfate_uptake': thiosulfate_uptake, #not in seconds 
        'thiosulfate_concentration_mmol': thiosulfate_conc_mmol, 
        'thiosulfate_concentration_mol': thiosulfate_conc_mol, 
    })
    intra_cys_list(intra_cys_mmol)
    return results_df

