# %%
# imports

"""Hard coded for Toniro backend, options for the noise model are hardcoded in the script."""
##TODO : Make the backend a parameter
##TODO : Make the noise model params parameters

import numpy as np
import pandas as pd
import time
from copy import deepcopy
import os
import pickle

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')

from proteinfolding.supporting_functions import *
from proteinfolding.paths import QISKT_NOISE_MODEL_DIR
from proteinfolding.logging_utils import setup_logging, log_info, get_git_commit
from proteinfolding.supporting_functions import symmetry_preserving_initial_state

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit

from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import BackendSampler
from qiskit.transpiler import PassManager

import json

def get_q_ham_and_mixer(num_rot, num_res, transverse_field=1):
    H = get_hamiltonian(num_rot, num_res)
    q_hamiltonian = get_q_hamiltonian(num_rot * num_res, H)
    XY_mixer = get_XY_mixer(num_rot * num_res, num_rot, transverse_field=transverse_field)
    return q_hamiltonian, XY_mixer

def get_initial_state(num_res, num_rot, pos=0):
    initial_bitstring = generate_initial_bitstring(num_res * num_rot, num_rot, pos=pos)
    state_vector = np.zeros(2**(num_res * num_rot))
    indexx = int(initial_bitstring, 2)
    state_vector[indexx] = 1
    qc = QuantumCircuit(num_res * num_rot)
    qc.initialize(state_vector, range(num_res * num_rot))
    return qc

def get_initial_parameters(p, cost_bound = 0.1, mixer_bound = 1.0):
    init_point_cost = np.random.uniform(-cost_bound, cost_bound, p)
    init_point_mixer = np.random.uniform(-mixer_bound, mixer_bound, p)
    initial_point = np.zeros(2*p)
    initial_point[0::2] = init_point_cost
    initial_point[1::2] = init_point_mixer
    return initial_point



def postprocessing_intermediate_data(intermediate_data, num_res, num_rot):
    num_qubits = num_res * num_rot
    intermediate_data_dicts = []
    for item in intermediate_data:
        for dict_item in item:
            intermediate_data_dicts.append(dict_item)

    probability = []
    total_arr = []
    cumulative_probability_dict = {}
    cumulative_total_dict = {}

    probability = []
    total_arr = []
    cumulative_probability_dict = {}
    cumulative_total_dict = {}

    for i, dict in enumerate(intermediate_data_dicts):
        #print(f"\n\nIteration {i}")
        #print(f"Dictionary: {dict}")
        hits = 0.0
        total = 0.0
        for key in dict:
            bitstring = int_to_bitstring(key, num_qubits)
            #print(f"\nBitstring: {bitstring}")
            hamming = check_hamming(bitstring, num_rot)
        #  print(f"Hamming condition: {hamming}")
            if check_hamming(bitstring, num_rot):
                hits += dict[key]
                total += dict[key]
                #print(f"Bitstring: {bitstring} has a value of {dict[key]}")
                if bitstring in cumulative_probability_dict:
                    cumulative_probability_dict[bitstring] += dict[key]
                else:
                    cumulative_probability_dict[bitstring] = dict[key]
            else:
                total += dict[key]
            if bitstring in cumulative_total_dict:
                cumulative_total_dict[bitstring] += dict[key]
            else:
                cumulative_total_dict[bitstring] = dict[key]
                #print(f"Bitstring: {bitstring} does not satisfy the Hamming condition.")
                #pass
        
        probability.append(hits)
        total_arr.append(total)

    


    sum_total = sum(cumulative_total_dict.values())
    sum_probability = sum(cumulative_probability_dict.values())


    norm = sum_total
    fraction = sum_probability / sum_total

    return intermediate_data_dicts, probability, total_arr, cumulative_probability_dict, cumulative_total_dict, norm, fraction

def get_all_seen_bitstrings(result1, num_rot, num_res, q_hamiltonian):
    num_qubits = num_rot * num_res
    eigenstate_distribution = result1.eigenstate
    best_measurement = result1.best_measurement
    final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

    all_bitstrings = {}
    all_unrestricted_bitstrings = {}

    for state, prob in final_bitstrings.items():
        bitstring = int_to_bitstring(state, num_qubits)
        energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
        if bitstring not in all_unrestricted_bitstrings:
            all_unrestricted_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
        all_unrestricted_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
        all_unrestricted_bitstrings[bitstring]['energy'] = (all_unrestricted_bitstrings[bitstring]['energy'] * all_unrestricted_bitstrings[bitstring]['count'] + energy) / (all_unrestricted_bitstrings[bitstring]['count'] + 1)
        all_unrestricted_bitstrings[bitstring]['count'] += 1

        if check_hamming(bitstring, num_rot):
            if bitstring not in all_bitstrings:
                all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
            all_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
            all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
            all_bitstrings[bitstring]['count'] += 1

            ## here, count is not related to the number of counts of the optimiser,
            ## it keeps track of number of times the bitstring has been seen in 
            ## different iterations of the optimiser. This is used to calculate the
            ## average energy of the bitstring across iterations. Ideally this should
            ## be weighted by the probability of the bitstring in each iteration.
            ## For the moment the energy is calculated by the statevector simulator,
            ## so it should be fine. ##TODO : Adapt this for noisy simulations.

    return all_bitstrings, all_unrestricted_bitstrings

def get_probability_distributions(intermediate_data_dicts, num_rot, num_res, q_hamiltonian, norm, all_bitstrings, all_unrestricted_bitstrings):
    num_qubits = num_rot * num_res
    for data in intermediate_data_dicts:
        
        for int_bitstring in data:
            probability = data[int_bitstring]
            intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
            energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
            if intermediate_bitstring not in all_unrestricted_bitstrings:
                all_unrestricted_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
            all_unrestricted_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
            all_unrestricted_bitstrings[intermediate_bitstring]['energy'] = (all_unrestricted_bitstrings[intermediate_bitstring]['energy'] * all_unrestricted_bitstrings[intermediate_bitstring]['count'] + energy) / (all_unrestricted_bitstrings[intermediate_bitstring]['count'] + 1)
            all_unrestricted_bitstrings[intermediate_bitstring]['count'] += 1

            if check_hamming(intermediate_bitstring, num_rot):
                if intermediate_bitstring not in all_bitstrings:
                    all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
                all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                
                all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
                all_bitstrings[intermediate_bitstring]['count'] += 1

    # %%
    sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
    sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

    # %%
    # Store information
    probabilities = []
    sorted_bitstrings_arr = []


    probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
            

    sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

    probabilities = np.array(probabilities) / norm

    return probabilities, sorted_bitstrings_arr, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings
    
def encode_results_json(num_res, num_rot, alpha, shots, p, pos, sorted_bitstrings_arr, probabilities, fraction, norm, elapsed_time1, intermediate_data, parameters_arr, cumulative_probability_dict, cumulative_total_dict, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings , transverse_field=1):
    result_dict = {
        'params' : (num_res, num_rot, alpha, shots, p, pos, transverse_field),
        'bitstrings': sorted_bitstrings_arr,
        'probabilities': probabilities,
        'fraction': fraction,
        'norm': norm,
        'energy': sorted_bitstrings[0][1]['energy'],
        'elapsed_time': elapsed_time1,
        'intermediate_data': intermediate_data,
        'parameters': parameters_arr,
        'cumulative_probability_dict': cumulative_probability_dict,
        'cumulative_total_dict': cumulative_total_dict,
        'all_bitstrings': all_bitstrings,
        'all_unrestricted_bitstrings': all_unrestricted_bitstrings,
        'sorted_bitstrings': sorted_bitstrings,
        'sorted_unrestricted_bitstrings': sorted_unrestricted_bitstrings
    }

    return result_dict

def noisy_simulation_XY(num_rot, num_res, alpha, shots, p, pos=0):
    # Log the parameters
    log_info(f"Running noisy_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}")
    
    # Log the current git commit
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    num_qubits = num_rot * num_res
    ### TRANSVERSE FIELD NOT IMPLEMENTED ###
    q_hamiltonian, XY_mixer = get_q_ham_and_mixer(num_rot, num_res, transverse_field=1)


    simulator = Aer.get_backend('qasm_simulator')
    
    if os.path.exists(f'{QISKT_NOISE_MODEL_DIR}/noise_model_torino.pkl'):
        with open(f'{QISKT_NOISE_MODEL_DIR}/noise_model_torino.pkl', 'rb') as f:
            noise_model = pickle.load(f)
    else:
        provider = IBMProvider()

        device_backend = provider.get_backend('ibm_torino')
        noise_model = NoiseModel.from_backend(device_backend)

        with open(f'{QISKT_NOISE_MODEL_DIR}/noise_model_torino.pkl', 'wb') as f:
            pickle.dump(noise_model, f)

    # %%
    options= {
        "noise_model": noise_model,
        "basis_gates": simulator.configuration().basis_gates,
        "coupling_map": simulator.configuration().coupling_map,
        "seed_simulator": 42,
        "shots": shots,
        "optimization_level": 3,
        "resilience_level": 3
    }

    # %%
    def callback(quasi_dists, parameters, energy):
        intermediate_data.append(
            quasi_dists
        )
        parameters_arr.append(
            parameters
        )

    intermediate_data = []
    parameters_arr = []
    noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())
    
    initial_point = get_initial_parameters(p, cost_bound = 0.1, mixer_bound = 1.0)
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

    start_time1 = time.time()
    qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point,callback=callback, aggregation=alpha)
    result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1


    intermediate_data_dicts, probability, total_arr, cumulative_probability_dict, cumulative_total_dict, norm, fraction = postprocessing_intermediate_data(intermediate_data, num_res, num_rot)

    all_bitstrings, all_unrestricted_bitstrings = get_all_seen_bitstrings(result1, num_rot, num_res, q_hamiltonian)

    # get probability distributions
    probabilities, sorted_bitstrings_arr, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings = get_probability_distributions(intermediate_data_dicts, num_rot, num_res, q_hamiltonian, norm, all_bitstrings, all_unrestricted_bitstrings)

    result = encode_results_json(num_res, num_rot, alpha, shots, p, pos, sorted_bitstrings_arr, probabilities, fraction, norm, elapsed_time1, intermediate_data, parameters_arr, cumulative_probability_dict, cumulative_total_dict, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings)

    with open(f"{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p.json", 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)





def statevector_simulation_XY(num_rot, num_res, alpha, shots, p, pos=0, transverse_field=1):
    # Log the parameters
    log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
    
    # Log the current git commit
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    num_qubits = num_rot * num_res
    q_hamiltonian, XY_mixer = get_q_ham_and_mixer(num_rot, num_res, transverse_field=transverse_field)
    initial_point = get_initial_parameters(p, cost_bound = 0.1, mixer_bound = 1.0)
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

    simulator = Aer.get_backend('qasm_simulator')
    options= {
        "basis_gates": simulator.configuration().basis_gates,
        "coupling_map": simulator.configuration().coupling_map,
        "seed_simulator": 42,
        "shots": shots,
        "optimization_level": 3,
        "resilience_level": 3
    }
    def callback(quasi_dists, parameters, energy):
        
        intermediate_data.append(
            quasi_dists
        )
        parameters_arr.append(
            parameters
        )

    # %%
    intermediate_data = []
    parameters_arr = []
    sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

    start_time1 = time.time()
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point,callback=callback, aggregation=alpha)
    result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    intermediate_data_dicts, probability, total_arr, cumulative_probability_dict, cumulative_total_dict, norm, fraction = postprocessing_intermediate_data(intermediate_data, num_res, num_rot)

    all_bitstrings, all_unrestricted_bitstrings = get_all_seen_bitstrings(result1, num_rot, num_res, q_hamiltonian)

    # get probability distributions
    probabilities, sorted_bitstrings_arr, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings = get_probability_distributions(intermediate_data_dicts, num_rot, num_res, q_hamiltonian, norm, all_bitstrings, all_unrestricted_bitstrings)

    result = encode_results_json(num_res, num_rot, alpha, shots, p, pos, sorted_bitstrings_arr, probabilities, fraction, norm, elapsed_time1, intermediate_data, parameters_arr, cumulative_probability_dict, cumulative_total_dict, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings, transverse_field=transverse_field)


    with open(f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield.json", 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)


def statevector_simulation_XY_parallel(num_rot, num_res, alpha, shots, p, pos=0, transverse_field=1):
    # Log the parameters
    log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
    
    # Log the current git commit
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    num_qubits = num_rot * num_res
    q_hamiltonian, XY_mixer = get_q_ham_and_mixer(num_rot, num_res, transverse_field=transverse_field)


    initial_point = get_initial_parameters(p, cost_bound = 0.1, mixer_bound = 1.0)


    
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)


    simulator = Aer.get_backend('qasm_simulator')

    options= {
        "basis_gates": simulator.configuration().basis_gates,
        "coupling_map": simulator.configuration().coupling_map,
        "seed_simulator": 42,
        "shots": shots,
        "optimization_level": 3,
        "resilience_level": 3,
        "max_parallel_threads" : 0,
        "max_parallel_experiments" : 0,
        "max_parallel_shots" : 1,
        "statevector_parallel_threshold" : 16
    }

    def callback(quasi_dists, parameters, energy):
        
        intermediate_data.append(
            quasi_dists
        )
        parameters_arr.append(
            parameters
        )

    # %%
    intermediate_data = []
    parameters_arr = []
    sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())

    start_time1 = time.time()
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point,callback=callback, aggregation=alpha)
    result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    intermediate_data_dicts, probability, total_arr, cumulative_probability_dict, cumulative_total_dict, norm, fraction = postprocessing_intermediate_data(intermediate_data, num_res, num_rot)

    all_bitstrings, all_unrestricted_bitstrings = get_all_seen_bitstrings(result1, num_rot, num_res, q_hamiltonian)

    # get probability distributions
    probabilities, sorted_bitstrings_arr, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings = get_probability_distributions(intermediate_data_dicts, num_rot, num_res, q_hamiltonian, norm, all_bitstrings, all_unrestricted_bitstrings)

    result = encode_results_json(num_res, num_rot, alpha, shots, p, pos, sorted_bitstrings_arr, probabilities, fraction, norm, elapsed_time1, intermediate_data, parameters_arr, cumulative_probability_dict, cumulative_total_dict, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings, transverse_field=transverse_field)

    with open(f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield.json", 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)



def simulated_annealing(num_rot, num_res, niter, T, stepsize, minimizer_kwargs):
    # Log the parameters
    log_info(f"Running simulated annealing with parameters: num_rot={num_rot}, num_res={num_res}, niter={niter}, T={T}, stepsize={stepsize}, minimizer_kwargs={minimizer_kwargs}")
    
    # Log the current git commit
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    num_qubits = num_rot * num_res
     # %%
    H = get_hamiltonian(num_rot, num_res)

    from scipy.sparse import lil_matrix

    def construct_sparse_operator(qubit_indices, num_qubits):
        size = 2 ** num_qubits
        operator = lil_matrix((size, size))
        for state in range(size):
            binary_state = f"{state:0{num_qubits}b}"  
            sign = 1
            for idx in qubit_indices:
                if binary_state[num_qubits - 1 - idx] == '1':
                    sign *= -1
            operator[state, state] = sign
        return operator

    C_sparse = lil_matrix((2**num_qubits, 2**num_qubits))

    for i in range(num_qubits):
        operator = construct_sparse_operator([i], num_qubits)
        C_sparse += H[i][i] * operator

    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            operator = construct_sparse_operator([i, j], num_qubits)
            C_sparse += H[i][j] * operator
    
  

    from scipy.optimize import basinhopping
    # initial guess must be bianry
    x0 = np.random.choice([0, 1], size=num_qubits)

    # Basinhopping settings
    

    def binary_state_to_vector(state):
        """Convert a binary state (0 and 1s) into a quantum state vector."""
        vector_size = 2 ** len(state)
        state_index = int("".join(str(int(x)) for x in state), 2)
        state_vector = np.zeros(vector_size)
        state_vector[state_index] = 1
        return state_vector

    def energy_function(state, H):
        """Calculate the energy of a given binary state."""
        state_vector = binary_state_to_vector(state)
        return state_vector @ H @ state_vector

    def objective_function(x):
        binary_state = [1 if xi > 0.5 else 0 for xi in x] 
        return energy_function(binary_state, C_sparse)
    
    intermediate_data = []
    
    def callback(x, f, accept):
        intermediate_data.append(
            x
        )

    result = basinhopping(objective_function, x0, minimizer_kwargs=minimizer_kwargs, niter=niter, T=T, stepsize=stepsize, callback=callback)

    print("Optimisation completed.")
    print("Global minimum with SA: ", result.fun)
    print("Parameters at minimum: ", result.x)
    print("Intermediate data: ", intermediate_data)

    # save the results





# def statevector_simulation_XY_parallel_new(num_rot, num_res, alpha, shots, p, pos=0, transverse_field=1):
#     # Log the parameters
#     log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
    
#     # Log the current git commit
#     commit = get_git_commit()
#     log_info(f"Current git commit: {commit}")

#     num_qubits = num_rot * num_res


#     # %%
#     H = get_hamiltonian(num_rot, num_res)
#     q_hamiltonian = get_q_hamiltonian(num_qubits, H)
#     XY_mixer = get_XY_mixer(num_qubits, num_rot, transverse_field=transverse_field)

#     # %%
#     from qiskit_algorithms.minimum_eigensolvers import QAOA
#     from qiskit.quantum_info.operators import Pauli, SparsePauliOp
#     from qiskit_algorithms.optimizers import COBYLA, SPSA
#     from qiskit.primitives import Sampler
#     from qiskit import QuantumCircuit






#     mixer_boud = 1.0
#     cost_bound = 0.1
#     # generate a random vector initial_point of length 2*p, even indices should be drawn from a uniform distribution with bound cost_bound, odd indices should be drawn from a uniform distribution with bound mixer_bound
#     init_point_cost = np.random.uniform(-cost_bound, cost_bound, p)
#     init_point_mixer = np.random.uniform(-mixer_boud, mixer_boud, p)
#     initial_point = np.zeros(2*p)
#     initial_point[0::2] = init_point_cost
#     initial_point[1::2] = init_point_mixer
    
#     from proteinfolding.supporting_functions import symmetry_preserving_initial_state
    
#     qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

 


#     # %%
#     from qiskit_aer import Aer
#     from qiskit_ibm_provider import IBMProvider
#     from qiskit.primitives import Sampler
#     from qiskit.transpiler import PassManager

#     simulator = Aer.get_backend('qasm_simulator')

#     options= {
#         "seed_simulator": 42,
#         "shots": shots,
#         "max_parallel_threads" : 0,
#         "max_parallel_experiments" : 0,
#         "max_parallel_shots" : 1,
#         "statevector_parallel_threshold" : 16
#     }

#     def callback(quasi_dists, parameters, energy):
        
#         intermediate_data.append(
#             quasi_dists
#         )
#         parameters_arr.append(
#             parameters
#         )

#     # %%
#     intermediate_data = []
#     parameters_arr = []
#     sampler = Sampler(options=options)

#     start_time1 = time.time()
#     qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point,callback=callback, aggregation=alpha)
#     result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1

#     # %%
#     # %% Post Selection



#     # %%
#     intermediate_data_dicts = []
    

#     for item in intermediate_data:
#         for dict_item in item:
#             intermediate_data_dicts.append(dict_item)

    

#     # %%

#     probability = []
#     total_arr = []
#     cumulative_probability_dict = {}
#     cumulative_total_dict = {}

#     for i, dict in enumerate(intermediate_data_dicts):
#         #print(f"\n\nIteration {i}")
#         #print(f"Dictionary: {dict}")
#         hits = 0.0
#         total = 0.0
#         for key in dict:
#             bitstring = int_to_bitstring(key, num_qubits)
#             #print(f"\nBitstring: {bitstring}")
#             hamming = check_hamming(bitstring, num_rot)
#         #  print(f"Hamming condition: {hamming}")
#             if check_hamming(bitstring, num_rot):
#                 hits += dict[key]
#                 total += dict[key]
#                 #print(f"Bitstring: {bitstring} has a value of {dict[key]}")
#                 if bitstring in cumulative_probability_dict:
#                     cumulative_probability_dict[bitstring] += dict[key]
#                 else:
#                     cumulative_probability_dict[bitstring] = dict[key]
#             else:
#                 total += dict[key]
#             if bitstring in cumulative_total_dict:
#                 cumulative_total_dict[bitstring] += dict[key]
#             else:
#                 cumulative_total_dict[bitstring] = dict[key]
#                 #print(f"Bitstring: {bitstring} does not satisfy the Hamming condition.")
#                 #pass
        
#         probability.append(hits)
#         total_arr.append(total)

#     # %%
#     # sum the values of the cumulative_probability_dict and cumulative_total_dict

#     sum_total = sum(cumulative_total_dict.values())
#     sum_probability = sum(cumulative_probability_dict.values())

#     # print(f"Total probability: {sum_probability}, Total: {sum_total}")

#     norm = sum_total
#     fraction = sum_probability / sum_total









#     # %%
#     eigenstate_distribution = result1.eigenstate
#     best_measurement = result1.best_measurement
#     final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

#     all_bitstrings = {}
#     all_unrestricted_bitstrings = {}

#     for state, prob in final_bitstrings.items():
#         bitstring = int_to_bitstring(state, num_qubits)
#         energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
#         if bitstring not in all_unrestricted_bitstrings:
#             all_unrestricted_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
#         all_unrestricted_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
#         all_unrestricted_bitstrings[bitstring]['energy'] = (all_unrestricted_bitstrings[bitstring]['energy'] * all_unrestricted_bitstrings[bitstring]['count'] + energy) / (all_unrestricted_bitstrings[bitstring]['count'] + 1)
#         all_unrestricted_bitstrings[bitstring]['count'] += 1

#         if check_hamming(bitstring, num_rot):
#             if bitstring not in all_bitstrings:
#                 all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
#             all_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
#             all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
#             all_bitstrings[bitstring]['count'] += 1

#             ## here, count is not related to the number of counts of the optimiser,
#             ## it keeps track of number of times the bitstring has been seen in 
#             ## different iterations of the optimiser. This is used to calculate the
#             ## average energy of the bitstring across iterations. Ideally this should
#             ## be weighted by the probability of the bitstring in each iteration.
#             ## For the moment the energy is calculated by the statevector simulator,
#             ## so it should be fine. ##TODO : Adapt this for noisy simulations.


#     for data in intermediate_data_dicts:
        
#         for int_bitstring in data:
#             probability = data[int_bitstring]
#             intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
#             energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)
#             if intermediate_bitstring not in all_unrestricted_bitstrings:
#                 all_unrestricted_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
#             all_unrestricted_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
#             all_unrestricted_bitstrings[intermediate_bitstring]['energy'] = (all_unrestricted_bitstrings[intermediate_bitstring]['energy'] * all_unrestricted_bitstrings[intermediate_bitstring]['count'] + energy) / (all_unrestricted_bitstrings[intermediate_bitstring]['count'] + 1)
#             all_unrestricted_bitstrings[intermediate_bitstring]['count'] += 1

#             if check_hamming(intermediate_bitstring, num_rot):
#                 if intermediate_bitstring not in all_bitstrings:
#                     all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
#                 all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                
#                 all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
#                 all_bitstrings[intermediate_bitstring]['count'] += 1

#     # %%
#     sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
#     sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

#     # %%
#     # Store information
#     probabilities = []
#     sorted_bitstrings_arr = []


#     probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
            

#     sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

#     probabilities = np.array(probabilities) / norm

#     # %%
#     result = {
#         'params' : (num_res, num_rot, alpha, shots, p, pos, transverse_field),
#         'bitstrings': sorted_bitstrings_arr,
#         'probabilities': probabilities,
#         'fraction': fraction,
#         'norm': norm,
#         'energy': sorted_bitstrings[0][1]['energy'],
#         'elapsed_time': elapsed_time1,
#         'intermediate_data': intermediate_data,
#         'parameters': parameters_arr,
#         'cumulative_probability_dict': cumulative_probability_dict,
#         'cumulative_total_dict': cumulative_total_dict,
#         'all_bitstrings': all_bitstrings,
#         'all_unrestricted_bitstrings': all_unrestricted_bitstrings,
#         'sorted_bitstrings': sorted_bitstrings,
#         'sorted_unrestricted_bitstrings': sorted_unrestricted_bitstrings
#     }

#     import json

#     # write json encoder


        
#     # write json file

#     with open(f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield.json", 'w') as f:
#         json.dump(result, f, cls=NumpyEncoder)
    