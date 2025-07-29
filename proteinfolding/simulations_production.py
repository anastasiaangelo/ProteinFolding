# %%
"""Hard coded for Tprio backend, options for the noise model are hardcoded in the script."""
##TODO : Make the backend a parameter
##TODO : Make the noise model params parameters

import numpy as np
import pandas as pd
import time
import os
import pickle
import json
from scipy.sparse import lil_matrix
from scipy.optimize import basinhopping

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
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit import QuantumCircuit

from qiskit_aer import Aer
# from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import BackendSampler, Sampler
from qiskit.transpiler import PassManager


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

    sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
    sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

    # Store information
    probabilities = []
    sorted_bitstrings_arr = []

    probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
    probabilities = np.array(probabilities) / norm

    sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

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
    
    log_info(f"Running noisy_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}")
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

    options= {
        "noise_model": noise_model,
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

    intermediate_data = []
    parameters_arr = []
    noisy_sampler = BackendSampler(backend=simulator, options=options, bound_pass_manager=PassManager())
    
    initial_point = get_initial_parameters(p, cost_bound = 0.1, mixer_bound = 1.0)
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

    start_time1 = time.time()
    qaoa1 = QAOA(sampler=noisy_sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point, callback=callback, aggregation=alpha)
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


def statevector_simulation_XY_parallel(num_rot, num_res, alpha, shots, p, pos=0, transverse_field=1):
    
    log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
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

    probabilities, sorted_bitstrings_arr, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings = get_probability_distributions(intermediate_data_dicts, num_rot, num_res, q_hamiltonian, norm, all_bitstrings, all_unrestricted_bitstrings)

    result = encode_results_json(num_res, num_rot, alpha, shots, p, pos, sorted_bitstrings_arr, probabilities, fraction, norm, elapsed_time1, intermediate_data, parameters_arr, cumulative_probability_dict, cumulative_total_dict, all_bitstrings, all_unrestricted_bitstrings, sorted_bitstrings, sorted_unrestricted_bitstrings, transverse_field=transverse_field)

    with open(f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield.json", 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)


def simulated_annealing(num_rot, num_res, niter, T, stepsize, minimizer_kwargs):
    start_time = time.time()
    log_info(f"Running simulated annealing with parameters: num_rot={num_rot}, num_res={num_res}, niter={niter}, T={T}, stepsize={stepsize}, minimizer_kwargs={minimizer_kwargs}")
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    num_qubits = num_rot * num_res

    H = get_hamiltonian(num_rot, num_res)

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
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Optimisation completed.")
    print("Global minimum with SA: ", result.fun)
    print("Parameters at minimum: ", result.x)
    print("Intermediate data: ", intermediate_data)

    result_dict = {
        'params' : (num_res, num_rot, niter, T, stepsize, minimizer_kwargs),
        'bitstrings': result.x,
        'energy': result.fun,
        'elapsed_time': elapsed_time,
        'intermediate_data': intermediate_data
    }

    with open(f"noiseless-{num_res}res-{num_rot}rot-simulated_annealing", 'w') as f:
        json.dump(result_dict, f, cls=NumpyEncoder)


def statevector_simulation_XY_parallel_new(num_rot, num_res, alpha, shots, p, simulation_id=None, pos=0, transverse_field=1):
    
    log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")

    print("Python script started!")

    num_qubits = num_rot * num_res
    H = get_hamiltonian(num_rot, num_res)
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)
    XY_mixer = get_XY_mixer(num_qubits, num_rot, transverse_field=transverse_field)

    if simulation_id is not None:

        from proteinfolding.paths import XY_QAOA_DATA_DIR
        from proteinfolding.paths import OPTIMAL_PARAMETERS_FILE

        PROCESSED_DATA_DIR = os.path.join(XY_QAOA_DATA_DIR, simulation_id)
        
        df_final_params = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, OPTIMAL_PARAMETERS_FILE))
    
        df_final_params['optimal_parameters'] = df_final_params['optimal_parameters'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        conditions = (
            (df_final_params['p'] == p) &
            (df_final_params['alpha'] == alpha) &
            (df_final_params['num_res'] == num_rot) &
            (df_final_params['num_rot'] == num_res) 
        )

        matching_row = df_final_params[conditions]

        if matching_row.empty:
            print(f"Skipping: No matching row found for num_res={num_res}, num_rot={num_rot}, p={p}, alpha={alpha}, shots={shots}, pos={pos}, transverse_field={transverse_field}")
            return None  

        initial_point = matching_row.iloc[0]['optimal_parameters']
        initial_point = np.array(initial_point).flatten()  
        print(f"Recovered initial point: {initial_point}")

    else:
        mixer_boud = 1.0
        cost_bound = 0.1
        # generate a random vector initial_point of length 2*p, even indices should be drawn from a uniform distribution with bound cost_bound, odd indices should be drawn from a uniform distribution with bound mixer_bound
        init_point_cost = np.random.uniform(-cost_bound, cost_bound, p)
        init_point_mixer = np.random.uniform(-mixer_boud, mixer_boud, p)
        initial_point = np.zeros(2*p)
        initial_point[0::2] = init_point_cost
        initial_point[1::2] = init_point_mixer

        gamma_init = np.linspace(-cost_bound, cost_bound, p)  # Cost angles
        beta_init = np.linspace(-mixer_boud, mixer_boud, p)  # Mixer angles
        initial_point = np.zeros(2*p)
        initial_point[0::2] = gamma_init
        initial_point[1::2] = beta_init
    
    from proteinfolding.supporting_functions import symmetry_preserving_initial_state
    
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

    from proteinfolding.paths import EXACT_DATA_ENERGY_BITSTRING_FILE
    from proteinfolding.data_processing import find_min_energy_and_bitstring_from_exact_energy_dataframe

    exact_data = pd.read_csv(EXACT_DATA_ENERGY_BITSTRING_FILE, compression='gzip')

    df_filtered = exact_data[(exact_data['num_res'] == num_res) & (exact_data['num_rot'] == num_rot)]
    if df_filtered.empty:
        raise Exception(f"No matching rows found for num_res = {num_res} and num_rot = {num_rot}")

    # Instead of raising an error, select the row with the minimum energy
    df_filtered = df_filtered.sort_values(by='energies').head(1)

    min_energy, min_energy_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(exact_data, num_res, num_rot)

    options= {
        "seed_simulator": 42,
        "shots": shots,
        "max_parallel_threads" : 1,
        "max_parallel_experiments" : 1,
        "max_parallel_shots" : 5,
        "statevector_parallel_threshold" : 35
    }   

    def callback(quasi_dists, parameters, energy):
        intermediate_data.append(
            quasi_dists
        )
        parameters_arr.append(
            parameters
        )
    
    def ground_state_found_in_quasi_dists(intermediate_data, min_energy_bitstring):
        """Return True if the ground state is found in the latest quasi distribution."""
        if not intermediate_data:
            return False

        # Check the *last* item in intermediate_data (i.e. from the latest optimizer iteration)
        last_quasi_dists = intermediate_data[-1]  

        for quasi_dist in last_quasi_dists:
            for key, prob in quasi_dist.items():
                bitstring = int_to_bitstring(key, num_qubits)
                if bitstring == min_energy_bitstring:
                    return True

        return False

    intermediate_data = []
    parameters_arr = []

    sampler = Sampler(options=options)

    print("Running main function...")
    start_time1 = time.time()
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point, callback=callback, aggregation=alpha)
    result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)

    # max_iterations = 200  # Prevent infinite looping
    # result1 = None
    # ground_state_info = None
    # max_iter_qaoa = 10

    # print("Starting QAOA optimization loop...")

    # for iteration in range(max_iterations):
    #     qaoa1.optimizer.set_options(maxiter=max_iter_qaoa)
 
    #     result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    #     print(len(intermediate_data))
    #  #   print(f"DEBUG: iteration {iteration+1} - intermediate_data[-1] = {intermediate_data[-1]}")
    
    #     if ground_state_found_in_quasi_dists(intermediate_data, min_energy_bitstring):
    #         print(f"Ground state {min_energy_bitstring} found at iteration {iteration+1}. Stopping optimization.")
    #         gs_found = True
    #         ground_state_info = {
    #             'iteration_found': iteration + 1,
    #             'bitstring': min_energy_bitstring
    #         }   
    #         break
    #     else:
    #         current_parameters = parameters_arr[-1][0]
    #         print(f"current parameters: {current_parameters}")
    #         # print(f"type: {type(current_parameters)}, value: {current_parameters}, shape: {np.shape(current_parameters)}")

    #         qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=current_parameters, callback=callback, aggregation=alpha)
    # else:
    #     print("Reached max_iterations without finding ground state.")

    print("Optimization complete. Final results stored in `result1`.")

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    # import json

    # if not ground_state_info: 
    #     print("Warning: Ground state not found! JSON will not be saved.")
    # else:
    #     print("Ground state found successfully, writing JSON...")
    #     timestamp = int(time.time())
    #     filename = f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield_{timestamp}.json"
        
    #     json_output = {
    #     'params': (num_res, num_rot, alpha, shots, p, pos, transverse_field),
    #     'min_iter': ground_state_info['iteration_found'],
    #     'ground_state_bitstring': ground_state_info['bitstring'],
    #     'parameters' : parameters_arr
    #     }

    #     with open(filename, 'w') as f:
    #         json.dump(json_output, f, cls=NumpyEncoder)

    #     print(f"JSON file saved: {filename}")

    # %% Post Selection
    intermediate_data_dicts = []

    for item in intermediate_data:
        for dict_item in item:
            intermediate_data_dicts.append(dict_item)

    # probability = []
    # total_arr = []
    # cumulative_probability_dict = {}
    # cumulative_total_dict = {}

    found_min_energy = False
    first_iteration = None 

    for i, dict in enumerate(intermediate_data_dicts):
        if found_min_energy:    
            break
        
        print(f"\n\nIteration {i+1}")
        print(f"Dictionary: {dict}")

        hits = 0.0
        total = 0.0

        for key in dict:
            bitstring = int_to_bitstring(key, num_qubits)
            # energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
            # print(f"Bitstring: {bitstring}, Energy: {energy}")
            print(f"Bitstring: {bitstring}")
            
            if bitstring == min_energy_bitstring:
                if first_iteration is None:  # Store only the first occurrence
                    first_iteration = i+1
                    print(f"Ground state {bitstring} first appeared at iteration {first_iteration} with energy {min_energy}")
                found_min_energy = True
                break
            

        #     if check_hamming(bitstring, num_rot):
        #         hits += dict[key]
        #         total += dict[key]
        #         #print(f"Bitstring: {bitstring} has a value of {dict[key]}")
        #         if bitstring in cumulative_probability_dict:
        #             cumulative_probability_dict[bitstring] += dict[key]
        #         else:
        #             cumulative_probability_dict[bitstring] = dict[key]
        #     else:
        #         total += dict[key]
        #     if bitstring in cumulative_total_dict:
        #         cumulative_total_dict[bitstring] += dict[key]
        #     else:
        #         cumulative_total_dict[bitstring] = dict[key]
        #         #print(f"Bitstring: {bitstring} does not satisfy the Hamming condition.")
        #         #pass
        
        # probability.append(hits)
        # total_arr.append(total)

    # %% sum the values of the cumulative_probability_dict and cumulative_total_dict

    # sum_total = sum(cumulative_total_dict.values())
    # sum_probability = sum(cumulative_probability_dict.values())
    # print(f"Total probability: {sum_probability}, Total: {sum_total}")

    # norm = sum_total
    # fraction = sum_probability / sum_total

    # %%
    eigenstate_distribution = result1.eigenstate
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
                

    sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
    sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

    # %% Store information
    # probabilities = []
    # sorted_bitstrings_arr = []

    # probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
    # probabilities = np.array(probabilities) / norm

    sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

    result = {
        'params' : (num_res, num_rot, alpha, shots, p, pos, transverse_field),
        'bitstrings': sorted_bitstrings_arr,
        # 'probabilities': probabilities,
        # 'fraction': fraction,
        # 'norm': norm,
        'energy': sorted_bitstrings[0][1]['energy'],
        # 'elapsed_time': elapsed_time1,
        'intermediate_data': intermediate_data,
        'parameters': parameters_arr,
        # 'cumulative_probability_dict': cumulative_probability_dict,
        # 'cumulative_total_dict': cumulative_total_dict,
        'all_bitstrings': all_bitstrings,
        'all_unrestricted_bitstrings': all_unrestricted_bitstrings,
        'sorted_bitstrings': sorted_bitstrings,
        'sorted_unrestricted_bitstrings': sorted_unrestricted_bitstrings
    }

    print(f"Debug: Is result defined? { 'Yes' if 'result' in locals() else 'No' }")
    print(f"Debug: Number of bitstrings found: {len(sorted_bitstrings)}")
    print(f"Debug: First few bitstrings: {sorted_bitstrings[:5] if len(sorted_bitstrings) > 5 else sorted_bitstrings}")
    print(f"Debug: Found min energy? {found_min_energy}")

    # write json file
    import json

    if not sorted_bitstrings: 
        print("Warning: No bitstrings collected! JSON will not be saved.")
    else:
        print("Bitstrings collected successfully, writing JSON...")
        timestamp = int(time.time())
        filename = f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, cls=NumpyEncoder)

        print(f"JSON file saved: {filename}")



# %%
def statevector_simulation_XY_parallel_trained(num_rot, num_res, alpha, shots, p, simulation_id, ignore_shots=True, pos=0, transverse_field=1):
    log_info(f"Running statevector_simulation with parameters: num_rot={num_rot}, num_res={num_res}, alpha={alpha}, shots={shots}, p={p}, transverse_field={transverse_field}")
    commit = get_git_commit()
    log_info(f"Current git commit: {commit}")
    print("Python script started!")

    # %%
    from proteinfolding.supporting_functions import symmetry_preserving_initial_state

    num_qubits = num_rot * num_res
    H = get_hamiltonian(num_rot, num_res)
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)
    XY_mixer = get_XY_mixer(num_qubits, num_rot, transverse_field=transverse_field)
    
    qc = symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot, theta=np.pi/4)

    # %%
    from proteinfolding.paths import XY_QAOA_DATA_DIR
    from proteinfolding.paths import OPTIMAL_PARAMETERS_FILE

    # simulation id of trianing simulation, must be same hyperparameters
    PROCESSED_DATA_DIR = os.path.join(XY_QAOA_DATA_DIR, simulation_id)
    
    df_final_params = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, OPTIMAL_PARAMETERS_FILE))
  
    df_final_params['optimal_parameters'] = df_final_params['optimal_parameters'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    conditions = (
        (df_final_params['num_res'] == num_res) &
        (df_final_params['num_rot'] == num_rot) &
        (df_final_params['p'] == p) &
        (df_final_params['alpha'] == alpha) 
    )

    if not ignore_shots:
        conditions &= (df_final_params['shots'] == shots)

    if 'pos' in df_final_params.columns and pos is not None:
        conditions &= (df_final_params['pos'] == pos)

    if 'transverse_field' in df_final_params.columns and transverse_field is not None:
        conditions &= (df_final_params['transverse_field'] == transverse_field)

    matching_row = df_final_params[conditions]

    if matching_row.empty:
        print(f"Skipping: No matching row found for num_res={num_res}, num_rot={num_rot}, p={p}, alpha={alpha}, shots={shots}, pos={pos}, transverse_field={transverse_field}")
        return None  

    if ignore_shots:
        matching_row = matching_row.iloc[[0]]  # Take the first row
    else:
        matching_row = matching_row 

    initial_point = matching_row.iloc[0]['optimal_parameters']
    initial_point = np.array(initial_point).flatten()  
    print(f"Recovered initial point: {initial_point}")

    print("Running main function...")

    from proteinfolding.paths import EXACT_DATA_ENERGY_BITSTRING_FILE
    from proteinfolding.data_processing import find_min_energy_and_bitstring_from_exact_energy_dataframe

    exact_data = pd.read_csv(EXACT_DATA_ENERGY_BITSTRING_FILE, compression='gzip')

    df_filtered = exact_data[(exact_data['num_res'] == num_res) & (exact_data['num_rot'] == num_rot)]
    if df_filtered.empty:
        raise Exception(f"No matching rows found for num_res = {num_res} and num_rot = {num_rot}")

    df_filtered = df_filtered.sort_values(by='energies').head(1)

    min_energy, min_energy_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(exact_data, num_res, num_rot)


    options= {
        "seed_simulator": 42,
        "shots": shots,
        "max_parallel_threads" : 4,
        "max_parallel_experiments" : 2,
        "max_parallel_shots" : 4,
        "statevector_parallel_threshold" : 50
    }

    def callback(quasi_dists, parameters, energy):
        intermediate_data.append(
            quasi_dists
        )
        parameters_arr.append(
            parameters
        )

    def ground_state_found_in_quasi_dists(intermediate_data, min_energy_bitstring):
        """Return True if the ground state is found in the latest quasi distribution."""
        if not intermediate_data:
            return False

        # Check the *last* item in intermediate_data (i.e. from the latest optimizer iteration)
        last_quasi_dists = intermediate_data[-1]  

        for quasi_dist in last_quasi_dists:
            for key, prob in quasi_dist.items():
                bitstring = int_to_bitstring(key, num_qubits)
                if bitstring == min_energy_bitstring:
                    return True

        return False


    intermediate_data = []
    parameters_arr = []
    sampler = Sampler(options=options)

    start_time1 = time.time()
    qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=initial_point, callback=callback, aggregation=alpha)
 
    max_iterations = 50  
    result1 = None
    ground_state_info = None
    max_iter_qaoa = 10

    print("Starting QAOA optimization loop...")

    
    for iteration in range(max_iterations):
        qaoa1.optimizer.set_options(maxiter=max_iter_qaoa)
 
        result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
        print(len(intermediate_data))
     #   print(f"DEBUG: iteration {iteration+1} - intermediate_data[-1] = {intermediate_data[-1]}")
    
        if ground_state_found_in_quasi_dists(intermediate_data, min_energy_bitstring):
            print(f"Ground state {min_energy_bitstring} found at iteration {iteration+1}. Stopping optimization.")
            gs_found = True
            ground_state_info = {
                'iteration_found': iteration + 1,
                'bitstring': min_energy_bitstring
            }   
            break
        else:
            current_parameters = parameters_arr[-1][0]
            print(f"current parameters: {current_parameters}")
            
            qaoa1 = QAOA(sampler=sampler, optimizer=COBYLA(), reps=p, initial_state=qc, mixer=XY_mixer, initial_point=current_parameters, callback=callback, aggregation=alpha)
    else:
        print("Reached max_iterations without finding ground state.")

    print("Optimization complete. Final results stored in `result1`.")

    # result1 = qaoa1.compute_minimum_eigenvalue(q_hamiltonian)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    if not ground_state_info: 
        print("Warning: Ground state not found! JSON will not be saved.")
    else:
        print("Ground state found successfully, writing JSON...")
        timestamp = int(time.time())
        filename = f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield_{timestamp}.json"
        
        json_output = {
        'params': (num_res, num_rot, alpha, shots, p, pos, transverse_field),
        'min_iter': ground_state_info['iteration_found'],
        'ground_state_bitstring': ground_state_info['bitstring'],
        'parameters' : parameters_arr
        }

        with open(filename, 'w') as f:
            json.dump(json_output, f, cls=NumpyEncoder)

        print(f"JSON file saved: {filename}")


    # # %% Post Selection
    # intermediate_data_dicts = []

    # for item in intermediate_data:
    #     for dict_item in item:
    #         intermediate_data_dicts.append(dict_item)

    # probability = []
    # total_arr = []
    # cumulative_probability_dict = {}
    # cumulative_total_dict = {}

    # found_min_energy = False
    # first_iteration = None 

    # for i, dict in enumerate(intermediate_data_dicts):
    #     if found_min_energy:    
    #         break
        
    #     print(f"\n\nIteration {i+1}")
    #     print(f"Dictionary: {dict}")

    #     hits = 0.0
    #     total = 0.0

    #     for key in dict:
    #         bitstring = int_to_bitstring(key, num_qubits)
    #         # energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
    #         # print(f"Bitstring: {bitstring}, Energy: {energy}")
    #         print(f"Bitstring: {bitstring}")

    #         if bitstring == min_energy_bitstring:
    #             if first_iteration is None:  # Store only the first occurrence
    #                 first_iteration = i+1
    #                 print(f"Ground state {bitstring} first appeared at iteration {first_iteration} with energy {min_energy}")
    #             found_min_energy = True
    #             break
            

    #         if check_hamming(bitstring, num_rot):
    #             hits += dict[key]
    #             total += dict[key]
    #             if bitstring in cumulative_probability_dict:
    #                 cumulative_probability_dict[bitstring] += dict[key]
    #             else:
    #                 cumulative_probability_dict[bitstring] = dict[key]
    #         else:
    #             total += dict[key]
    #         if bitstring in cumulative_total_dict:
    #             cumulative_total_dict[bitstring] += dict[key]
    #         else:
    #             cumulative_total_dict[bitstring] = dict[key]
    #             #pass
        
    #     probability.append(hits)
    #     total_arr.append(total)

    # # %% sum the values of the cumulative_probability_dict and cumulative_total_dict

    # sum_total = sum(cumulative_total_dict.values())
    # sum_probability = sum(cumulative_probability_dict.values())
    # norm = sum_total
    # # fraction = sum_probability / sum_total

    # # %%
    # eigenstate_distribution = result1.eigenstate
    # final_bitstrings = {state: probability for state, probability in eigenstate_distribution.items()}

    # all_bitstrings = {}
    # all_unrestricted_bitstrings = {}

    # for state, prob in final_bitstrings.items():
    #     bitstring = int_to_bitstring(state, num_qubits)
    #     energy = calculate_bitstring_energy(bitstring, q_hamiltonian)
    #     if bitstring not in all_unrestricted_bitstrings:
    #         all_unrestricted_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
    #     all_unrestricted_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
    #     all_unrestricted_bitstrings[bitstring]['energy'] = (all_unrestricted_bitstrings[bitstring]['energy'] * all_unrestricted_bitstrings[bitstring]['count'] + energy) / (all_unrestricted_bitstrings[bitstring]['count'] + 1)
    #     all_unrestricted_bitstrings[bitstring]['count'] += 1

    #     if check_hamming(bitstring, num_rot):
    #         if bitstring not in all_bitstrings:
    #             all_bitstrings[bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
    #         all_bitstrings[bitstring]['probability'] += prob  # Aggregate probabilities
    #         all_bitstrings[bitstring]['energy'] = (all_bitstrings[bitstring]['energy'] * all_bitstrings[bitstring]['count'] + energy) / (all_bitstrings[bitstring]['count'] + 1)
    #         all_bitstrings[bitstring]['count'] += 1

    
    # for data in intermediate_data_dicts:
    #     for int_bitstring in data:
    #         probability = data[int_bitstring]
    #         intermediate_bitstring = int_to_bitstring(int_bitstring, num_qubits)
    #         energy = calculate_bitstring_energy(intermediate_bitstring, q_hamiltonian)

    #         if intermediate_bitstring not in all_unrestricted_bitstrings:
    #             all_unrestricted_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}

    #         all_unrestricted_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
    #         all_unrestricted_bitstrings[intermediate_bitstring]['energy'] = (all_unrestricted_bitstrings[intermediate_bitstring]['energy'] * all_unrestricted_bitstrings[intermediate_bitstring]['count'] + energy) / (all_unrestricted_bitstrings[intermediate_bitstring]['count'] + 1)
    #         all_unrestricted_bitstrings[intermediate_bitstring]['count'] += 1

    #         if check_hamming(intermediate_bitstring, num_rot):
    #             if intermediate_bitstring not in all_bitstrings:
    #                 all_bitstrings[intermediate_bitstring] = {'probability': 0, 'energy': 0, 'count': 0}
    #             all_bitstrings[intermediate_bitstring]['probability'] += probability  # Aggregate probabilities
                
    #             all_bitstrings[intermediate_bitstring]['energy'] = (all_bitstrings[intermediate_bitstring]['energy'] * all_bitstrings[intermediate_bitstring]['count'] + energy) / (all_bitstrings[intermediate_bitstring]['count'] + 1)
    #             all_bitstrings[intermediate_bitstring]['count'] += 1


    # sorted_bitstrings = sorted(all_bitstrings.items(), key=lambda x: x[1]['energy'])
    # sorted_unrestricted_bitstrings = sorted(all_unrestricted_bitstrings.items(), key=lambda x: x[1]['energy'])

    # # %% Store information
    # probabilities = []
    # sorted_bitstrings_arr = []

    # probabilities = [data['probability'] for bitstring, data in sorted_bitstrings]
    # probabilities = np.array(probabilities) / norm

    # sorted_bitstrings_arr = [bitstring for bitstring, data in sorted_bitstrings]

    # result = {
    #     'params' : (num_res, num_rot, alpha, shots, p, pos, transverse_field),
    #     'bitstrings': sorted_bitstrings_arr,
    #     'probabilities': probabilities,
    #     # 'fraction': fraction,
    #     'norm': norm,
    #     'energy': sorted_bitstrings[0][1]['energy'],
    #     'elapsed_time': elapsed_time1,
    #     'intermediate_data': intermediate_data,
    #     'parameters': parameters_arr,
    #     'cumulative_probability_dict': cumulative_probability_dict,
    #     'cumulative_total_dict': cumulative_total_dict,
    #     'all_bitstrings': all_bitstrings,
    #     'all_unrestricted_bitstrings': all_unrestricted_bitstrings,
    #     'sorted_bitstrings': sorted_bitstrings,
    #     'sorted_unrestricted_bitstrings': sorted_unrestricted_bitstrings
    # }

    # print(f"Debug: Is result defined? { 'Yes' if 'result' in locals() else 'No' }")
    # print(f"Debug: Number of bitstrings found: {len(sorted_bitstrings)}")
    # print(f"Debug: First few bitstrings: {sorted_bitstrings[:5] if len(sorted_bitstrings) > 5 else sorted_bitstrings}")
    # print(f"Debug: Found min energy? {found_min_energy}")

    # # write json file
    # import json

    # if not sorted_bitstrings: 
    #     print("Warning: No bitstrings collected! JSON will not be saved.")
    # else:
    #     print("Bitstrings collected successfully, writing JSON...")

    #     timestamp = int(time.time())
    #     filename = f"noiseless-{num_res}res-{num_rot}rot-{alpha}alpha-{shots}shots-{p}p-{transverse_field}transversefield_{timestamp}.json"
    #     with open(filename, 'w') as f:
    #         json.dump(result, f, cls=NumpyEncoder)

    #     print(f"JSON file saved: {filename}")

