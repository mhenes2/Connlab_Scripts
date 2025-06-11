from schrodinger.application.desmond.packages import traj, topo, analysis
import csv
from datetime import datetime
import os
from collections import Counter
import itertools

startTime = datetime.now()

# directory where cms files and trajectories are kept
base_dir = "/media/labconn/Seagate/Mina/Structural_and_Evolutionary_Basis_of_Multiprotein_Recognition_by_Outer_Membrane_Protein_OprM/Production/analysis/no_water/"

# list of trajectory files
infile = "MexE-OprM_rep1_nowater_trj/"

cms_file = "MexE-OprM_rep1_nowater-out.cms"

# Load the trajectory
trj_out = traj.read_traj(os.path.join(base_dir, infile))

# read the msys_model and cms_model from the cms_file provided
msys_model, cms_model = topo.read_cms(os.path.join(base_dir, cms_file))

Hbonds = analysis.ProtProtHbondInter(msys_model,cms_model,asl='protein')

results = analysis.analyze(trj_out, Hbonds)

def get_atom_ids_solo(cms_model, pairs):
    # to get the info of the pairs
    atoms = [a for a in cms_model.atom]
    # atom_ids = []

    a1 = atoms[pairs[0]]
    a2 = atoms[pairs[1]]
    chain1 = a1.chain.strip()
    chain2 = a2.chain.strip()
    resnum1 = a1.resnum
    resnum2 = a2.resnum
    resname1 = a1.pdbres.strip()
    resname2 = a2.pdbres.strip()
    atom_name1 = a1.pdbname.strip()
    atom_name2 = a2.pdbname.strip()
    x = '{}_{}_{}_{} - {}_{}_{}_{}'.format(chain1, resnum1, resname1, atom_name1,
                                                       chain2, resnum2, resname2, atom_name2)

    return x


hbond_bb = []
hbond_ss = []
hbond_sb = []
hbond_bs = []

for frame in range(len(results)):
    for key in results[frame]:
        globals()[key].append(results[frame][key])


hbond_bb = list(itertools.chain(*hbond_bb))
hbond_ss = list(itertools.chain(*hbond_ss))
hbond_sb = list(itertools.chain(*hbond_sb))
hbond_bs = list(itertools.chain(*hbond_bs))

hbond_bb_gid = [(x + 1, y + 1) for (x, y) in hbond_bb]
hbond_ss_gid = [(x + 1, y + 1) for (x, y) in hbond_ss]
hbond_sb_gid = [(x + 1, y + 1) for (x, y) in hbond_sb]
hbond_bs_gid = [(x + 1, y + 1) for (x, y) in hbond_bs]

hbond_bb_counter = Counter(hbond_bb_gid)
hbond_ss_counter = Counter(hbond_ss_gid)
hbond_sb_counter = Counter(hbond_sb_gid)
hbond_bs_counter = Counter(hbond_bs_gid)

hbond_bb_filtered = []
hbond_ss_filtered = []
hbond_sb_filtered = []
hbond_bs_filtered = []

for item, count in hbond_bb_counter.items():
    if (count / float(len(trj_out))) * 100. > 50.:
        # split it up
        info = get_atom_ids_solo(cms_model,item)
        hbond_bb_filtered.append([info, float(count)])

with open('test_bb_hbond.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(hbond_bb_filtered)


for item, count in hbond_ss_counter.items():
    if (count / float(len(trj_out))) * 100. > 50.:
        # split it up
        info = get_atom_ids_solo(cms_model,item)
        hbond_ss_filtered.append([info, float(count)])

with open('test_ss_hbond.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(hbond_ss_filtered)


for item, count in hbond_sb_counter.items():
    if (count / float(len(trj_out))) * 100. > 50.:
        # split it up
        info = get_atom_ids_solo(cms_model,item)
        hbond_sb_filtered.append([info, float(count)])

with open('test_sb_hbond.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(hbond_sb_filtered)


for item, count in hbond_bs_counter.items():
    if (count / float(len(trj_out))) * 100. > 50.:
        # split it up
        info = get_atom_ids_solo(cms_model,item)
        hbond_bs_filtered.append([info, float(count)])

with open('test_bs_hbond.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(hbond_bs_filtered)

# for item, count in hbond_sb_counter.items():
#     if (count / 20.) * 100. > 50.:
#         # split it up
#         info = get_atom_ids_solo(cms_model, item)
#         first, second = info.split(" - ")
#         chain_1, resi_1, resnam_1, atomname_1 = first.split("_")
#         chain_2, resi_2, resnam_2, atomname_2 = second.split("_")
#
#         cmd.select(name='ca1', selection='c. {} and name {} and resi {}'.format(chain_1, atomname_1, resi_1))
#         cmd.select(name='ca2', selection='c. {} and name {} and resi {}'.format(chain_2, atomname_2, resi_2))
#
#         cmd.distance(name="hbonds", selection1='ca1', selection2='ca2')

#         cmd.color("blue", "hbonds")
#
# cmd.hide("labels", "all")
# cmd.set("dash_gap", "0")
