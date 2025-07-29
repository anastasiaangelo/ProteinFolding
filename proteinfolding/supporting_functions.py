import numpy as np
from copy import deepcopy
import pandas as pd
import ast
import json
import pickle
from itertools import product, combinations
import os, sys

print(os.getcwd())
from proteinfolding.paths import PYROSETTA_ENERGY_DATA_DIR, SIMULATION_SUMMARY_FILE

from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile
from itertools import product, combinations
import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
import time
from scipy.optimize import dual_annealing



def get_hyperparameters(jobid, *args):
    """
    Generate all combinations of parameters and return the combination corresponding to the job ID.

    This function accepts a variable number of arguments. Each argument can be a list of values for a particular parameter or a single number. In the latter case, it will be treated as a list with one element.

    The function generates all combinations of these parameters using the Cartesian product. The combinations are zero-indexed, and the combination corresponding to the job ID is returned.

    """

    args = [[arg] if not isinstance(arg, list) else arg for arg in args]
    all_combinations = list(product(*args))

    if jobid <= len(all_combinations):
        return all_combinations[jobid - 1]
    else:
        raise ValueError("Job ID is out of range.")
    

def get_job_ids():
    file_path = SIMULATION_SUMMARY_FILE
    with open(file_path, 'r') as file:
        lines = file.readlines()

    ids = []

    for line in lines:
        id = line.split()[0]
        ids.append(id)

    return ids


# Directory where the PyRosetta energy data is stored 

#### NEEDS TO BE CHANGED FOR CLUSTER - LOCAL ####

# PYROSETTA_ENERGY_DATA_DIR = "/u/aag/proteinfolding/data/raw/pyrosetta_energy_files_immutable"

# PYROSETTA_ENERGY_DATA_DIR = "/home/b/aag/proteinfolding/data/raw/pyrosetta_energy_files_immutable"
from proteinfolding import Ising_Hamiltonian as IHP
PYROSETTA_ENERGY_DATA_DIR_MUTABLE = "/Users/aag/Documents/proteinfolding/data/raw/pyrosetta_energy_files"

## just nearest neighbours ##

def get_hamiltonian(num_rot, num_res):

    one_body_file = f"{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_one_body_terms.csv"
    two_body_file = f"{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_two_body_terms.csv"

    # Check if both files exist
    if not os.path.exists(one_body_file) or not os.path.exists(two_body_file):
        print(f"Missing energy files for num_rot={num_rot}, num_res={num_res}. To generate uncomment IHP. Generating them...")
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # IHP.create_energy_files(num_res, num_rot)
        sys.stdout = original_stdout  

    print('files already present, not generating')
    df1 = pd.read_csv(one_body_file, usecols=['E_ii'])
    df = pd.read_csv(two_body_file, usecols=['E_ij'])

    q = df1['E_ii'].values
    num = len(q)
    value = df['E_ij'].values
    Q = np.zeros((num, num), dtype=np.float64)
    n = 0
    idx = np.arange(0, num - num_rot, num_rot) 

    for j in idx:
        i_range = np.arange(j, j + num_rot) 
        j_range = np.arange(j + num_rot, j + 2 * num_rot)  

        for i in i_range:
            for j_offset, j_val in enumerate(j_range):
                Q[i, j_val] = value[n]
                Q[j_val, i] = value[n]
                n += 1

    H = 0.25 * Q 
    np.fill_diagonal(H, -(0.5 * q + np.sum(0.25 * Q, axis=1))) 

    return H


def get_qubo_hamiltonian(num_rot, num_res):

    one_body_file = f"{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_one_body_terms.csv"
    two_body_file = f"{PYROSETTA_ENERGY_DATA_DIR_MUTABLE}/{num_rot}rot_{num_res}res_two_body_terms.csv"

    # Check if both files exist
    if not os.path.exists(one_body_file) or not os.path.exists(two_body_file):
        print(f"Missing energy files for num_rot={num_rot}, num_res={num_res}. To generate uncomment IHP. Generating them...")
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # IHP.create_energy_files(num_res, num_rot)
        sys.stdout = original_stdout  

    print('files already present, not generating')
    df1 = pd.read_csv(one_body_file, usecols=['E_ii'])
    df = pd.read_csv(two_body_file, usecols=['E_ij'])

    q = df1['E_ii'].values
    num = len(q)
    value = df['E_ij'].values
    Q = np.zeros((num, num), dtype=np.float64)
    n = 0
    idx = np.arange(0, num - num_rot, num_rot) 

    for j in idx:
        i_range = np.arange(j, j + num_rot) 
        j_range = np.arange(j + num_rot, j + 2 * num_rot)  

        for i in i_range:
            for j_offset, j_val in enumerate(j_range):
                Q[i, j_val] = value[n]
                Q[j_val, i] = value[n]
                n += 1

    return Q


from proteinfolding.paths import PYROSETTA_ENERGY_DATA_ALL

## all possible interactions ##
def get_hamiltonian_nonNN(num_rot, num_res):

    os.makedirs(PYROSETTA_ENERGY_DATA_ALL, exist_ok=True)
    one_body_file = os.path.join(PYROSETTA_ENERGY_DATA_ALL, f"{num_rot}rot_{num_res}res_one_body_terms.csv")
    two_body_file = os.path.join(PYROSETTA_ENERGY_DATA_ALL, f"{num_rot}rot_{num_res}res_two_body_terms.csv")

    # Check if both files exist
    if not os.path.exists(one_body_file) or not os.path.exists(two_body_file):
        print(f"Missing energy files for num_rot={num_rot}, num_res={num_res}. To generate uncomment IHP. Generating them...")
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        IHP.create_energy_files(num_res, num_rot)
        sys.stdout = original_stdout  
        print('energy file generated')

    print('files already present, not generating')
    df1 = pd.read_csv(one_body_file, usecols=['E_ii'])
    df = pd.read_csv(two_body_file, usecols=['E_ij'])

    q = df1['E_ii'].values
    num = len(q)
    value = df['E_ij'].values
    Q = np.zeros((num, num), dtype=np.float64)
    n = 0

    for r1 in range(num_res):  
        for r2 in range(r1 + 1, num_res):
            r1_base = r1 * num_rot
            r2_base = r2 * num_rot

            for i in range(num_rot):
                for j in range(num_rot):
                    idx1 = r1_base + i
                    idx2 = r2_base + j
                    Q[idx1, idx2] = value[n]
                    Q[idx2, idx1] = value[n]
                    n += 1

    H = 0.25 * Q 

    np.fill_diagonal(H, -(0.5 * q + np.sum(0.25 * Q, axis=1))) 

    k = 0
    for i in range(num_res*num_rot):
        k += 0.5 * q[i]

    for i in range(num_res*num_rot):
        for j in range(num_res*num_rot):
            if i != j:
                k += 0.5 * 0.25 * Q[i][j]
    return H


def X_op(i, num_qubits):
    """Return an X Pauli operator on the specified qubit in a num-qubit system."""
    op_list = ['I'] * num_qubits
    op_list[i] = 'X'
    return SparsePauliOp(Pauli(''.join(op_list)))


def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
    pauli_str = ['I']*n

    if i == j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))


def get_XY_mixer(num_qubits, num_rot, transverse_field=1):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])

    for i in range(0, num_qubits - num_rot + 1, num_rot):          
        for j in range(num_rot):
            for k in range(j + 1, num_rot):
                xx_term = ['I'] * num_qubits
                yy_term = ['I'] * num_qubits

                xx_term[i + j] = 'X'
                xx_term[i + k] = 'X'
                yy_term[i + j] = 'Y'
                yy_term[i + k] = 'Y'

                xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])

                hamiltonian += xx_op + yy_op

    hamiltonian *= transverse_field
    return -hamiltonian if num_rot == 2 else hamiltonian

    
def generate_pauli_zij(n, i, j):
    if i<0 or i >= n or j<0 or j>=n:
        raise ValueError(f"Indices out of bounds for n={n} qubits. ")
        
    pauli_str = ['I']*n

    if i == j:
        pauli_str[i] = 'Z'
    else:
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'

    return Pauli(''.join(pauli_str))


def get_q_hamiltonian(num_qubits, H):
    pauli_list = []
    coeffs = []

    for i in range(num_qubits):
        for j in range(i, num_qubits):
            if H[i][j] != 0:
                z_term = ['I'] * num_qubits
                z_term[i] = 'Z'
                if i != j:
                    z_term[j] = 'Z'
                pauli_list.append(''.join(z_term))
                coeffs.append(H[i][j])

    pauli_list.append('I' * num_qubits)
    coeffs.append(0.0)

    return SparsePauliOp(pauli_list, coeffs)


def generate_initial_bitstring(num_qubits, num_rot):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    pattern = ['0'] * num_rot  
    pattern[0] = '1'  # Ensure at least one '1' per group
    bitstring = ''.join(pattern * ((num_qubits // num_rot) + 1))[:num_qubits]

    return bitstring
    

def format_sparsepauliop(op):
    terms = []
    labels = [pauli.to_label() for pauli in op.paulis]
    coeffs = op.coeffs
    for label, coeff in zip(labels, coeffs):
        terms.append(f"{coeff:.10f} * {label}")
    return '\n'.join(terms)


def int_to_bitstring(state, total_bits):
    """Converts an integer state to a binary bitstring with padding of leading zeros."""
    return format(state, '0{}b'.format(total_bits))


def bitstring_to_int(bitstring):
    """Converts a binary bitstring to an integer."""
    return int(bitstring, 2)


def check_hamming(bitstring, substring_size):
    if not all(bit in ['0', '1'] for bit in bitstring):
        raise ValueError("Bitstring contains characters other than '0' and '1'.")
    """Check if each substring contains exactly one '1'."""
    substrings = [bitstring[i:i+substring_size] for i in range(0, len(bitstring), substring_size)]
    return all(sub.count('1') == 1 for sub in substrings)


def generate_bitstrings(num_res, num_rot):
    """
    Generates all valid bitstrings where each residue has exactly one '1'.
    """
    base_strings = ['0' * i + '1' + '0' * (num_rot - i - 1) for i in range(num_rot)]  # One-hot for a single residue

    # Generate all combinations where each residue picks one valid state
    bitstrings = [''.join(bits) for bits in product(base_strings, repeat=num_res)]

    return bitstrings


def calculate_bitstring_energy(bitstring, hamiltonian, backend=None):
  
    bitstring = bitstring[::-1]
    # Prepare the quantum circuit for the bitstring
    num_qubits = len(bitstring)
    qc = QuantumCircuit(num_qubits)
    for i, char in enumerate(bitstring):
        if char == '1':
            qc.x(i)  # Apply X gate if the bit in the bitstring is 1
    
    
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')

    qc = transpile(qc, backend)
    estimator = Estimator()
    resultt = estimator.run(observables=[hamiltonian], circuits=[qc], backend=backend).result()

    return resultt.values[0].real


def calculate_bitstring_energy_efficient(bitstring, hamiltonian):
    """
    Computes the energy of a bitstring by evaluating the expectation value 
    without expanding the Hamiltonian to a full matrix.
    """
    bit_array = np.array([int(b) for b in bitstring], dtype=int)

    # Convert bitstring to a computational basis state in Pauli representation
    pauli_expectation = 0

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        term_expectation = 1  # Start neutral
        for i, op in enumerate(str(pauli)):
            if op == 'Z':
                term_expectation *= (1 if bit_array[i] == 0 else -1)  # Z on |0> = +1, Z on |1> = -1
        pauli_expectation += coeff * term_expectation

    return pauli_expectation


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {value}, {e}")
        return None



def calculate_classical_energy_unconstrained(x, H):
    """
    Compute Ising energy from an arbitrary binary vector x (not necessarily one-hot).
    """
    x = np.asarray(x, dtype=float)

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.inf

    # Threshold to binary (optional)
    binary_state = np.array([1 if xi > 0.5 else 0 for xi in x], dtype=int)

    # Convert binary to Z basis: 0 → +1, 1 → -1
    z = 1 - 2 * binary_state

    energy = 0.0
    for i in range(len(z)):
        for j in range(len(z)):
            if i != j:
                energy += 0.5 * H[i][j] * z[i] * z[j]
    energy += np.sum(np.diag(H) * z)

    return energy


def calculate_classical_energy_unconstrained2(x, H):
    """
    Fully matches original Ising loop even if H is not symmetric.
    """
    x = np.asarray(x, dtype=float)

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.inf

    binary_state = (x > 0.5).astype(int)
    z = 1 - 2 * binary_state

    # Manually compute off-diagonal (i ≠ j)
    off_diag = np.sum((z[:, None] * z[None, :]) * H) - np.sum(np.diag(H) * z * z)
    off_diag *= 0.5

    # Diagonal terms
    diag = np.sum(np.diag(H) * z)

    return off_diag + diag


def get_ground_state_SA(num_rot_array, num_res_array):
    results = []

    for num_res in num_res_array:
        for num_rot in num_rot_array:
            num_bits = num_res * num_rot

            from proteinfolding.paths import EXACT_DATA_ENERGY_BITSTRING_FILE_ALL, EXACT_DATA_ENERGY_BITSTRING_FILE
            from proteinfolding.data_processing import find_min_energy_and_bitstring_from_exact_energy_dataframe
            
            exact_data = pd.read_csv("/Users/aag/Documents/proteinfolding/data/processed/exact/exact_ground_state_and_energy.csv")
            # exact_data = pd.read_csv(EXACT_DATA_ENERGY_BITSTRING_FILE_ALL, compression='gzip')
            df_filtered = exact_data[(exact_data['num_res'] == num_res) & (exact_data['num_rot'] == num_rot)]
            if df_filtered.empty:
                print(f"No exact energy found for num_rot={num_rot}, num_res={num_res}. Skipping.")
                continue

            # df_filtered = df_filtered.sort_values(by='energies').head(1)
            # min_energy, min_energy_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(exact_data, num_res, num_rot)
            
            df_filtered = df_filtered.sort_values(by='gs_energy').head(1)
            min_energy = df_filtered['gs_energy']
            print(min_energy)

            class FoundGroundState(Exception):
                pass

            class CallTracker:
                def __init__(self):
                    self.calls = 0
                    self.total_time = 0.0

            tracker = CallTracker()

            # H = get_hamiltonian_nonNN(num_res=num_res, num_rot=num_rot)
            H = get_hamiltonian(num_res=num_res, num_rot=num_rot)


            def objective_function(x):
                binary_state = [1 if xi > 0.5 else 0 for xi in x]
                tracker.calls += 1
                start_time = time.process_time()
                energy = calculate_classical_energy_unconstrained(binary_state, H, num_rot)
                tracker.total_time += time.process_time() - start_time
                if np.isclose(energy, min_energy, atol=1e-6):
                    raise FoundGroundState("Ground state energy found!")
                return energy

                        
            bounds = [(0, 1)] * num_bits

            found = False
            max_restarts = 50
            restarts = 0

            while not found and restarts < max_restarts:
                try:
                    result = dual_annealing(objective_function, bounds)
                    # fallback check (if exception wasn't raised)
                    found = np.isclose(result.fun, min_energy, atol=1e-6)
                except FoundGroundState:
                    print("Ground state energy reached. Interrupting SA early.")
                    result = type("DummyResult", (object,), {})()
                    result.fun = float(min_energy)
                    result.x = None
                    found = True
                restarts += 1

            if not found:
                print(f"\n⚠️ WARNING: Ground state NOT found after {restarts} restarts.")
            else:
                print(f"\n✅ Ground state found in {restarts} restart(s).")

            total_complexity = tracker.calls * (num_bits**2)
            
            print("Optimisation completed.")
            print("Global minimum with SA: ", result.fun)
            print("Parameters at minimum: ", result.x)
            print("Total CPU calls (energy function calls):", tracker.calls)
            print("Total evaluation time:", tracker.total_time, "seconds")
            print("Total complexity:", total_complexity)

            results.append({
                "num_res": num_res,
                "num_rot": num_rot,
                "num_qubits": num_bits,
                "ground_state": result.fun,
                "cpu_calls": total_complexity
            })

    return pd.DataFrame(results) 

import timeit


def find_best_dual_annealing_params(objective_func_template, bounds, ground_truth_energy, num_trials=10):
    accept_vals = [-1, -3, -5, -7]
    visit_vals = [1.5, 2.0, 2.5, 3.0]
    maxiter_vals = [1000, 2000, 3000]

    best_params = None
    best_success_rate = 0

    for accept, visit, maxiter in product(accept_vals, visit_vals, maxiter_vals):
        success_count = 0
        for _ in range(num_trials):
            class FoundGroundState(Exception):
                pass

            tracker = type("Tracker", (), {"calls": 0, "total_time": 0.0})()

            def objective_function(x):
                binary_state = [1 if xi > 0.5 else 0 for xi in x]
                tracker.calls += 1
                energy = objective_func_template(binary_state)  # Call the energy evaluator
                if np.isclose(energy, ground_truth_energy, atol=1e-6):
                    raise FoundGroundState()
                return energy

            try:
                result = dual_annealing(
                    objective_function,
                    bounds,
                    maxiter=maxiter,
                    seed=np.random.randint(0, 1e6),
                    accept=accept,
                    visit=visit
                )
            except FoundGroundState:
                success_count += 1

        success_rate = success_count / num_trials
        print(f"[accept={accept}, visit={visit}, maxiter={maxiter}] → success: {success_rate:.2f}")

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_params = (accept, visit, maxiter)

    print(f"\n✅ Best combo: accept={best_params[0]}, visit={best_params[1]}, maxiter={best_params[2]}")
    return best_params



def get_ground_state_SA_stat(num_rot_array, num_res_array, n_total_runs=100):
    results = []
    all_cpu_calls = {}

    for num_res in num_res_array:
        for num_rot in num_rot_array:
            num_bits = num_res * num_rot

            # Load exact ground state data
            exact_data = pd.read_csv("/Users/aag/Documents/proteinfolding/data/processed/exact/exact_ground_state_and_energy.csv")
            df_filtered = exact_data[(exact_data['num_res'] == num_res) & (exact_data['num_rot'] == num_rot)]
            if df_filtered.empty:
                print(f"❌ No exact energy for num_rot={num_rot}, num_res={num_res}. Skipping.")
                continue

            # Extract true minimum energy
            min_energy = float(df_filtered.sort_values(by='gs_energy').iloc[0]['gs_energy'])
            min_energy_bitstring = str(df_filtered.sort_values(by='gs_energy').iloc[0]['gs_bitstring'])

            print(f"✔️ Ground state energy for (res={num_res}, rot={num_rot}): {min_energy}")
            print(f"✔️ Ground state bitstring for (res={num_res}, rot={num_rot}): {min_energy_bitstring}")

            # Get Hamiltonian (user-defined function)
            H = get_hamiltonian(num_res=num_res, num_rot=num_rot)
            q_hamiltonian = get_q_hamiltonian(num_qubits=num_res*num_rot, H=H)

            # # Run n_total_runs independent SA calls
            # bounds = [(0, 1)] * num_bits 
            # best_accept, best_visit, best_maxiter = find_best_dual_annealing_params(
            #     lambda x: calculate_bitstring_energy_efficient_classical(x, H, num_rot),
            #     bounds,
            #     min_energy,
            #     num_trials=20
            # )
            cpu_calls_list = []
            best_bitstrings = []

            # print(f"\n✅ Best parameters: accept={best_accept}, visit={best_visit}, maxiter={best_maxiter}")

            for run in range(n_total_runs):
                class FoundGroundState(Exception):
                    pass

                class CallTracker:
                    def __init__(self):
                        self.calls = 0
                        self.total_time = 0.0

                tracker = CallTracker()

                def one_hot_penalty(x, num_rot):
                    penalty = 0.0
                    for i in range(0, len(x), num_rot):
                        group_sum = np.sum(x[i:i + num_rot])
                        penalty += (group_sum - 1) ** 2  # Encourages sum=1
                    return penalty
               
                best_bitstring = None
                best_energy = np.inf

                def objective_function(x):
                    nonlocal best_bitstring, best_energy

                    binary_state = [1 if xi > 0.5 else 0 for xi in x]
                    tracker.calls += 1
                    start_time = time.process_time()
                    bitstring = ''.join(map(str, binary_state))
                    # energy = calculate_classical_energy_unconstrained(binary_state, H)
                    energy = calculate_classical_energy_unconstrained2(binary_state, H)

                    penalty_weight = 1.5
                    penalty = one_hot_penalty(binary_state, num_rot)
                    total_energy = energy + penalty_weight * penalty

                    tracker.total_time += time.process_time() - start_time

                    # if np.isclose(total_energy, min_energy, atol=1e-6):
                    #     raise FoundGroundState("Ground state energy found!")

                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_bitstring = bitstring

                    if bitstring == min_energy_bitstring:
                        best_bitstring = bitstring
                        # print(f'GS FOUND via bitstring on call {tracker.calls}')
                        raise FoundGroundState("Ground state energy found!")
                    
                    return total_energy

                bounds = [(0, 1)] * num_bits
        
                try:
                    result = dual_annealing(objective_function, bounds, maxiter=1000, seed=np.random.randint(0, 1e6), accept=0.9, visit=1.01
                                             )
                    # find_best_dual_annealing_params(objective_function, bounds, min_energy, num_trials=20)
                    # if np.isclose(result.fun, min_energy, atol=1e-6):
                    #     cpu_calls_list.append(tracker.calls * (num_bits ** 2))
                    if best_bitstring == min_energy_bitstring:
                        cpu_calls_list.append(tracker.calls * (num_bits ** 2))
                    else:
                        cpu_calls_list.append(np.nan)
                        print(f"Run {run+1}/{n_total_runs} failed to reach ground state.")
                except FoundGroundState:
                    result = type("DummyResult", (object,), {})()
                    result.fun = float(min_energy)
                    cpu_calls_list.append(tracker.calls * (num_bits ** 2))
                    best_bitstrings.append(best_bitstring)  # <--- New line
                    # print(f"Ground state found early in run {run+1}.")

            # Stats
            successful_runs = np.count_nonzero(~np.isnan(cpu_calls_list))
            convergence_ratio = successful_runs / n_total_runs
            mean_cpu_calls = np.nanmean(cpu_calls_list)
            std_cpu_calls = np.nanstd(cpu_calls_list)

            all_cpu_calls[(num_res, num_rot)] = cpu_calls_list

            print(f"\n Finished {n_total_runs} trials for (num_res={num_res}, num_rot={num_rot})")
            print(f"→ Convergence ratio: {convergence_ratio:.2f}")
            print(f"→ Mean complexity: {mean_cpu_calls}")
            print(f"→ Std complexity: {std_cpu_calls}")
            print(f"→ Best bitstring: {best_bitstring}")


            results.append({
                "num_res": num_res,
                "num_rot": num_rot,
                "num_qubits": num_bits,
                "ground_state": min_energy,
                "mean_cpu_calls": mean_cpu_calls,
                "std_cpu_calls": std_cpu_calls,
                "successful_runs": successful_runs,
                "total_runs": n_total_runs,
                "convergence_ratio": convergence_ratio
            })

    with open("cpu_calls_per_config.pkl", "wb") as f:
        pickle.dump(all_cpu_calls, f)

    return pd.DataFrame(results)



import networkx as nx

def pauli_commutes(pauli1, pauli2):
    """Check if two Pauli strings commute."""
    non_identity_1 = [i for i, p in enumerate(pauli1) if p != 'I']
    non_identity_2 = [i for i, p in enumerate(pauli2) if p != 'I']

    # Get the intersection of non-identity positions
    common_indices = set(non_identity_1) & set(non_identity_2)

    # Count non-identity operators at the common positions
    different_paulis = sum(1 for i in common_indices if pauli1[i] != pauli2[i])

    # If different_paulis is even, they commute
    return different_paulis % 2 == 0


def compute_commutative_groups(pauli_strings):
    """Calculate the number of commutative groups in a Hamiltonian."""
    # Create a graph where nodes are Pauli strings
    G = nx.Graph()

    # Add nodes (Pauli strings)
    G.add_nodes_from(pauli_strings)

    # Add edges between non-commuting Pauli strings
    for p1, p2 in combinations(pauli_strings, 2):
        if not pauli_commutes(p1, p2):  # If they do NOT commute, add an edge
            G.add_edge(p1, p2)

    # Compute the chromatic number (minimum number of colors required)
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")

    # Number of commutative groups is the number of unique colors
    num_groups = len(set(coloring.values()))
    
    return num_groups


def get_ground_state_efficient(num_res, num_rot):
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)
    ground_state = bitstrings[min_energy_index]

    return min_energy, ground_state


def get_min_and_max_states_efficient(num_res, num_rot):
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)

    max_energy = max(energies)
    max_energy_index = energies.index(max_energy)
    min_state = bitstrings[min_energy_index]
    max_state = bitstrings[max_energy_index]

    return min_energy, min_state, max_energy, max_state


def get_ground_state_estimator(num_res, num_rot):
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    energies = [calculate_bitstring_energy(bitstring, q_hamiltonian) for bitstring in bitstrings]

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)
    ground_state = bitstrings[min_energy_index]

    return min_energy, ground_state

    
def get_all_states_efficient(num_res, num_rot):
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    return energies, bitstrings
    

def get_all_states_estimator(num_res, num_rot):
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    energies = [calculate_bitstring_energy(bitstring, q_hamiltonian) for bitstring in bitstrings]

    return energies, bitstrings


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        elif isinstance(obj, tuple):
            return str(obj)
        return super().default(obj)
            

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    

def sparsepauliop_dict_serializer(obj):
    if isinstance(obj, dict):
        return {key: {'paulis': value.to_list(), 'coeffs': value.coeffs.tolist()} for key, value in obj.items()}
    raise TypeError(f"Type {type(obj)} not serializable")


def sparsepauliop_dict_deserializer(json_str):
    data = json.loads(json_str)
    return {key: SparsePauliOp.from_list(value['paulis'], value['coeffs']) for key, value in data.items()}


def create_product_state(base_circuit, n):
    num_qubits = base_circuit.num_qubits

    product_circuit = QuantumCircuit(num_qubits * n)

    for i in range(n):
        base_circuit_copy = deepcopy(base_circuit)
        product_circuit.compose(base_circuit_copy, qubits=range(i * num_qubits, (i + 1) * num_qubits), inplace=True)
        
    return product_circuit


def R_gate(theta=np.pi/4):
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(1)
    qc.ry(theta*np.pi/2, 0)
    qc.rz(np.pi, 0)
    op = Operator(qc)
    return op


def A_gate(theta=np.pi/4):
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    rgate = R_gate(theta=theta)
    rgate_adj  = R_gate().adjoint()
    # apply rgate to qubit 0
    qc.unitary(rgate_adj, [1], label='R')
    qc.cx(0, 1)
    qc.unitary(rgate, [1], label='R')
    qc.cx(1, 0)
    return qc


def symmetry_preserving_initial_state(num_res, num_rot, theta=np.pi/4):
    if num_rot < 2:
        raise ValueError("num_rot must be at least 2.")

    qc = QuantumCircuit(num_rot)
    qc.x(num_rot // 2)
    agate = A_gate(theta=theta)

    for i in range(num_rot - 1):
        qc.compose(agate, [i, i + 1], inplace=True)

    init_state = create_product_state(qc, num_res)
    return init_state

    
def aggregate(dist : dict, alpha : float, energies : list) -> float:
    # if keys are not integers, convert them to integers
    if not all(isinstance(key, int) for key in dist.keys()):
        dist = {int(key): value for key, value in dist.items()}
    # measurements = [(value, np.real(calculate_bitstring_energy_efficient(int_to_bitstring(key, num_qubits),q_hamiltonian))) for key, value in dist.items()]
    measurements = []
    count = 0
    for key, value in dist.items():
        pair = (value, energies[count])
        count += 1
        measurements.append(pair)
    sorted_measurements = sorted(measurements, key=lambda x: x[1])

    accumulated_percent = 0.0  # once alpha is reached, stop
    cvar = 0.0
    for probability, value in sorted_measurements:
        cvar += value * min(probability, alpha - accumulated_percent)
        accumulated_percent += probability
        if accumulated_percent >= alpha:
            break

    return cvar / alpha