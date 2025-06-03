from __future__ import print_function
import cobrame
import cobra
from cobrame.util import dogma, building
import cobrame.util.building
import cobra.test
from collections import defaultdict

#import warnings
#warnings.filterwarnings('ignore')

me = cobrame.MEModel('test')
ijo=cobra.test.create_test_model('ecoli')

for met in ijo.metabolites:
    me.add_metabolites(met)
for rxn in ijo.reactions:
    me.add_reaction(rxn)

# "Translational capacity" of organism
me.global_info['kt'] = 4.5  # (in h-1)scott 2010, RNA-to-protein curve fit
me.global_info['r0'] = 0.087  # scott 2010, RNA-to-protein curve fit
me.global_info['k_deg'] = 1.0/5. * 60.0  # 1/5 1/min 60 min/h # h-1

# Molecular mass of RNA component of ribosome
me.global_info['m_rr'] = 1453. # in kDa

# Average molecular mass of an amino acid
me.global_info['m_aa'] = 109. / 1000.  # in kDa

# Proportion of RNA that is rRNA
me.global_info['f_rRNA'] = .86
me.global_info['m_nt'] = 324. / 1000.  # in kDa
me.global_info['f_mRNA'] = .02

# tRNA associated global information
me.global_info['m_tRNA'] = 25000. / 1000.  # in kDA
me.global_info['f_tRNA'] = .12

# Define the types of biomass that will be synthesized in the model
me.add_biomass_constraints_to_model(["protein_biomass", "mRNA_biomass", "tRNA_biomass", "rRNA_biomass",
                                     "ncRNA_biomass", "DNA_biomass", "lipid_biomass", "constituent_biomass",
                                     "prosthetic_group_biomass", "peptidoglycan_biomass"])
sequence = ("ATG" + "TTT" * 12 + "TAT" * 12 +
            "ACG" * 12 + "GAT" * 12 + "AGT" * 12 + "TGA")
gene = cobrame.TranscribedGene('RNA_a', 'mRNA', sequence)
me.add_metabolites([gene])
print(TranscriptionData('gene'))