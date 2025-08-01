# UVA-iGEM-2025-Model

Part 1: ECMpy Enzyme Constrained Model


Part 2: ODE/Stochastic Insert 2 Model 

All necessary functions for reproducing our results are contained within the functions_plus_plus.py file.

- ccda_odes, ccda_conc, and simulate_markov_chain are used for testing of the initial toxin-antitoxin system with non-variable stochastic fluctuations, meaning the probability is not designed to be changed once initially implemented.

- ccdr_odes and TF_conc are used to model the production of our transcription factor, the concentration of which influences the probability of binding to the promoter and stimulating ccdb expression.

- The functions simulate_multi_markov_chain, multi_ccda_conc, TF_conc, and stoc_mod are designed to put all of these functions together and create a dynamic probability system that describes ccdb expression against that of ccda.

Our analysis files for the individual systems are provided in data_analysis_ccda and data_analysis_TF for testing each system separately.

The file stoc_mod.py is used for testing the entire system and outputs the desired graphs of ccda and ccdb.



