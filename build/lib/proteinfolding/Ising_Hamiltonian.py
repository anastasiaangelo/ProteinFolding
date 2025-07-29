### Take input pdb, score, repack and extract one and two body energies
# Script for generating the test sturcture with its rotamers 
#  lines 23 and 87 to vary
import pyrosetta
pyrosetta.init()

from pyrosetta.teaching import *
from pyrosetta import *

import csv
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.rotamer_set import *
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta import PyMOLMover

from .paths import PYROSETTA_PDB_DATA_DIR, PYROSETTA_ENERGY_DATA_DIR_MUTABLE

def create_energy_files(num_res, num_rot):

    # Initiate structure, scorefunction, change PDB files
    # num_res = 2
    # num_rot = 3
    pose = pyrosetta.pose_from_pdb(f"{PYROSETTA_PDB_DATA_DIR}/{num_res}residue.pdb")


    residue_count = pose.total_residue()
    sfxn = get_score_function(True)
    # print(pose.sequence())
    # print(residue_count)


    relax_protocol = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax_protocol.set_scorefxn(sfxn)
    relax_protocol.apply(pose)

    # Define task, interaction graph and rotamer sets (model_protein_csv.py)
    task_pack = TaskFactory.create_packer_task(pose) 

    rotsets = RotamerSets()
    pose.update_residue_neighbors()
    sfxn.setup_for_packing(pose, task_pack.designing_residues(), task_pack.designing_residues())
    packer_neighbor_graph = pyrosetta.rosetta.core.pack.create_packer_graph(pose, sfxn, task_pack)
    rotsets.set_task(task_pack)
    rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph)
    rotsets.prepare_sets_for_packing(pose, sfxn) 
    ig = InteractionGraphFactory.create_interaction_graph(task_pack, rotsets, pose, sfxn, packer_neighbor_graph)
    # print("built", rotsets.nrotamers(), "rotamers at", rotsets.nmoltenres(), "positions.")
    rotsets.compute_energies(pose, sfxn, packer_neighbor_graph, ig, 1)

    # Output structure to be visualised in pymol
    pose.dump_pdb(f"{PYROSETTA_PDB_DATA_DIR}/{num_res}residue_output_repacked.pdb")

    # Define dimension for matrix
    max_rotamers = 0
    for residue_number in range(1, residue_count+1):
        n_rots = rotsets.nrotamers_for_moltenres(residue_number)
        # print(f"Residue {residue_number} has {n_rots} rotamers.")
        if n_rots > max_rotamers:
            max_rotamers = n_rots

    # print("Maximum number of rotamers:", max_rotamers)


    E = np.zeros((max_rotamers, max_rotamers))
    Hamiltonian = np.zeros((max_rotamers, max_rotamers))

    E1 = np.zeros((max_rotamers, max_rotamers))
    Hamiltonian1 = np.zeros((max_rotamers, max_rotamers))

    data_list = []
    data_list1 = []
    df = pd.DataFrame(columns=['res i', 'res j', 'rot A_i', 'rot B_j', 'E_ij'])
    df1 = pd.DataFrame(columns=['res i', 'rot A_i', 'E_ii'])


    # # Visualisation of structure after repacking with rotamers
    # pmm = PyMOLMover()
    # clone_pose = Pose()
    # clone_pose.assign(pose)
    # pmm.apply(clone_pose)
    # pmm.send_hbonds(clone_pose)
    # pmm.keep_history(True)
    # pmm.apply(clone_pose)

    # to limit to n rotamers per residue, change based on how many rotamers desired
    # num_rot = 2

    # Loop to find Hamiltonian values Jij - interaction of rotamers on NN residues
    for residue_number in range(1, residue_count):
        rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
        if rotamer_set_i == None: # skip if no rotamers for the residue
            continue

        residue_number2 = residue_number + 1
        residue2 = pose.residue(residue_number2)
        rotamer_set_j = rotsets.rotamer_set_for_residue(residue_number2)
        if rotamer_set_j == None:
            continue

        molten_res_i = rotsets.resid_2_moltenres(residue_number)
        molten_res_j = rotsets.resid_2_moltenres(residue_number2)

        edge_exists = ig.find_edge(molten_res_i, molten_res_j)
            
        if not edge_exists:
                continue
        
        for rot_i in range(1, rotamer_set_i.num_rotamers() + 1):
            for rot_j in range(1, rotamer_set_j.num_rotamers() + 1):
                E[rot_i-1, rot_j-1] = ig.get_two_body_energy_for_edge(molten_res_i, molten_res_j, rot_i, rot_j)
                Hamiltonian[rot_i-1, rot_j-1] = E[rot_i-1, rot_j-1]

        for rot_i in range(10, num_rot + 10):       #, rotamer_set_i.num_rotamers() + 1):
            for rot_j in range(10, num_rot + 10):       #, rotamer_set_j.num_rotamers() + 1):
                # print(f"Interaction energy between rotamers of residue {residue_number} rotamer {rot_i} and residue {residue_number2} rotamer {rot_j} :", Hamiltonian[rot_i-1, rot_j-1])
                data = {'res i': residue_number, 'res j': residue_number2, 'rot A_i': rot_i, 'rot B_j': rot_j, 'E_ij': Hamiltonian[rot_i-1, rot_j-1]}
                data_list.append(data)
        

    # Save the two-body energies to a csv file
    df = pd.DataFrame(data_list)
    df.to_csv(f'{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_two_body_terms.csv', index=False)

    # to choose the two rotamers with the largest energy in absolute value
    # df.assign(abs_E=df['E_ij'].abs()).nlargest(2, 'abs_E').drop(columns=['abs_E']).to_csv('two_body_terms.csv', index=False)


    # Loop to find Hamiltonian values Jii
    for residue_number in range(1, residue_count + 1):
        residue1 = pose.residue(residue_number)
        rotamer_set_i = rotsets.rotamer_set_for_residue(residue_number)
        if rotamer_set_i == None: 
            continue

        molten_res_i = rotsets.resid_2_moltenres(residue_number)

        for rot_i in range(10, num_rot +10):        #, rotamer_set_i.num_rotamers() + 1):
            E1[rot_i-1, rot_i-1] = ig.get_one_body_energy_for_node_state(molten_res_i, rot_i)
            Hamiltonian1[rot_i-1, rot_i-1] = E1[rot_i-1, rot_i-1]
            # print(f"Interaction score values of {residue1.name3()} rotamer {rot_i} with itself {Hamiltonian[rot_i-1,rot_i-1]}")
            data1 = {'res i': residue_number, 'rot A_i': rot_i, 'E_ii': Hamiltonian1[rot_i-1, rot_i-1]}
            data_list1.append(data1)
        


    # Save the one-body energies to a csv file
    df1 = pd.DataFrame(data_list1)
    df1.to_csv(f'{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_one_body_terms.csv', index=False)
    # to choose the two rotamers with the largest energy in absolute value
    # df1.assign(abs_Ei=df1['E_ii'].abs()).nlargest(2, 'abs_Ei').drop(columns=['abs_Ei']).to_csv('one_body_terms.csv', index=False)

    # run all above if called from command line

    # get num_res and num_rot from command line

if __name__ == "__main__":
    # num_res = int(sys.argv[1])
    # num_rot = int(sys.argv[2])
    # main(num_res, num_rot)
    pass


