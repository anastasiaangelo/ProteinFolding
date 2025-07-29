import numpy as np
from scipy.sparse.linalg import eigsh
from copy import deepcopy
import pandas as pd
import ast
import json
from itertools import product
import os, sys

# print the current working directory
print(os.getcwd())
from proteinfolding.paths import PYROSETTA_ENERGY_DATA_DIR, SIMULATION_SUMMARY_FILE

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.quantum_info.operators import Pauli, SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator
from qiskit import QuantumCircuit, transpile



def get_hyperparameters(jobid, *args):
    """
    Generate all combinations of parameters and return the combination corresponding to the job ID.

    This function accepts a variable number of arguments. Each argument can be a list of values for a particular parameter or a single number. In the latter case, it will be treated as a list with one element.

    The function generates all combinations of these parameters using the Cartesian product. The combinations are zero-indexed, and the combination corresponding to the job ID is returned.

    Parameters:
    jobid (int): The job ID. This should be a positive integer. It corresponds to the index of the combination of parameters to return.

    *args: Variable length argument list. Each argument can be a list of values for a particular parameter or a single number.

    Returns:
    tuple: A tuple containing the combination of parameters corresponding to the job ID. The order of the parameters in the tuple is the same as the order of the arguments.

    Raises:
    ValueError: If the job ID is out of range (i.e., it is less than 1 or greater than the total number of combinations).

    Example:
    >>> get_hyperparameters(2, [1, 2], 3, [4, 5])
    (2, 3, 4)
    """
    from itertools import product

    # Ensure all arguments are lists
    args = [[arg] if not isinstance(arg, list) else arg for arg in args]

    # Generate all combinations of parameters
    all_combinations = list(product(*args))

    # Return the combination corresponding to the jobid
    if jobid <= len(all_combinations):
        return all_combinations[jobid - 1]
    else:
        raise ValueError("Job ID is out of range.")
    
def get_job_ids():
    file_path = SIMULATION_SUMMARY_FILE
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the lines of the file
        lines = file.readlines()

    # Initialize an empty list to store the IDs
    ids = []

    # Iterate over each line
    for line in lines:
        # Split the line by spaces and get the first element (the ID)
        id = line.split()[0]
        # Append the ID to the list
        ids.append(id)

    # Return the list of IDs
    return ids



# add directory to path


def get_hamiltonian(num_rot, num_res):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    #IHP.create_energy_files(num_res, num_rot)
    sys.stdout = original_stdout
    

    df1 = pd.read_csv(f"{PYROSETTA_ENERGY_DATA_DIR}/{num_rot}rot_{num_res}res_one_body_terms.csv")
    q = df1['E_ii'].values
    num = len(q)
    N = int(num/num_rot)
    num_qubits = num

    df = pd.read_csv(f"{PYROSETTA_ENERGY_DATA_DIR}/{num_rot}rot_{num_res}res_two_body_terms.csv")
    value = df['E_ij'].values
    Q = np.zeros((num,num))
    n = 0

    if num_rot == 2:
        for i in range(0, num-2):
            if i%2 == 0:
                Q[i][i+2] = deepcopy(value[n])
                Q[i+2][i] = deepcopy(value[n])
                Q[i][i+3] = deepcopy(value[n+1])
                Q[i+3][i] = deepcopy(value[n+1])
                n += 2
            elif i%2 != 0:
                Q[i][i+1] = deepcopy(value[n])
                Q[i+1][i] = deepcopy(value[n])
                Q[i][i+2] = deepcopy(value[n+1])
                Q[i+2][i] = deepcopy(value[n+1])
                n += 2      

    
    elif num_rot == 3:

        for j in range(0, num-3, num_rot):
            for i in range(j, j+num_rot):
                Q[i][j+3] = deepcopy(value[n])
                Q[j+3][i] = deepcopy(value[n])
                Q[i][j+4] = deepcopy(value[n+1])
                Q[j+4][i] = deepcopy(value[n+1])
                Q[i][j+5] = deepcopy(value[n+2])
                Q[j+5][i] = deepcopy(value[n+2])
                n += num_rot

    else:    
        raise ValueError("Number of rotomers not supported.")
    
    H = np.zeros((num,num))

    for i in range(num):
        for j in range(num):
            if i != j:
                H[i][j] = np.multiply(0.25, Q[i][j])

    for i in range(num):
        H[i][i] = -(0.5 * q[i] + sum(0.25 * Q[i][j] for j in range(num) if j != i))

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

def get_XY_mixer(num_qubits, num_rot,transverse_field = 1):
    if num_rot == 2:
        hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])  
        for i in range(0, num_qubits, 2):
            if i + 1 < num_qubits:
                xx_term = ['I'] * num_qubits
                yy_term = ['I'] * num_qubits
                xx_term[i] = 'X'
                xx_term[i+1] = 'X'
                yy_term[i] = 'Y'
                yy_term[i+1] = 'Y'
                xx_op = SparsePauliOp(Pauli(''.join(xx_term)), coeffs=[1/2])
                yy_op = SparsePauliOp(Pauli(''.join(yy_term)), coeffs=[1/2])
                hamiltonian += xx_op + yy_op
        hamiltonian *= transverse_field
        return -hamiltonian 
    elif num_rot == 3:
        hamiltonian = SparsePauliOp(Pauli('I' * num_qubits), coeffs=[0])
        for i in range(0, num_qubits - 2, 3): 
            x1x2 = ['I'] * num_qubits
            y1y2 = ['I'] * num_qubits
            x2x3 = ['I'] * num_qubits
            y2y3 = ['I'] * num_qubits
            x1x3 = ['I'] * num_qubits
            y1y3 = ['I'] * num_qubits

            x1x2[i] = 'X'
            x1x2[i+1] = 'X'
            y1y2[i] = 'Y'
            y1y2[i+1] = 'Y'
            x2x3[i+1] = 'X'
            x2x3[i+2] = 'X'
            y2y3[i+1] = 'Y'
            y2y3[i+2] = 'Y'
            x1x3[i] = 'X'
            x1x3[i+2] = 'X'
            y1y3[i] = 'Y'
            y1y3[i+2] = 'Y' 

            x1x2 = SparsePauliOp(Pauli(''.join(x1x2)), coeffs=[1/2])
            y1y2 = SparsePauliOp(Pauli(''.join(y1y2)), coeffs=[1/2])
            x2x3 = SparsePauliOp(Pauli(''.join(x2x3)), coeffs=[1/2])
            y2y3 = SparsePauliOp(Pauli(''.join(y2y3)), coeffs=[1/2])
            x1x3 = SparsePauliOp(Pauli(''.join(x1x3)), coeffs=[1/2])
            y1y3 = SparsePauliOp(Pauli(''.join(y1y3)), coeffs=[1/2])

            hamiltonian += x1x2 + y1y2 + x2x3 + y2y3 + x1x3 + y1y3
        hamiltonian *= transverse_field
        return hamiltonian
    else:
        raise ValueError("Number of rotomers not supported.")
    
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
    q_hamiltonian = SparsePauliOp(Pauli('I'*num_qubits), coeffs=[0])

    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if H[i][j] != 0:
                pauli = generate_pauli_zij(num_qubits, i, j)
                op = SparsePauliOp(pauli, coeffs=[H[i][j]])
                q_hamiltonian += op

    for i in range(num_qubits):
        pauli = generate_pauli_zij(num_qubits, i, i)
        Z_i = SparsePauliOp(pauli, coeffs=[H[i][i]])
        q_hamiltonian += Z_i

    return q_hamiltonian
    
def generate_initial_bitstring(num_qubits, num_rot, pos=0):
    if num_rot == 2:
        bitstring = ['0' for _ in range(num_qubits)]
        for i in range(0, num_qubits, num_rot):
            bitstring[i + pos] = '1'
        return ''.join(bitstring)
    elif num_rot == 3:
        bitstring = ['0' for _ in range(num_qubits)]
        for i in range(0, num_qubits, num_rot):
            bitstring[i + pos] = '1'
        return ''.join(bitstring)
    else:
        raise ValueError("Number of rotomers not supported.")

    
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
    # throw an exception if bitstring contains characters other than '0' and '1'
    if not all(bit in ['0', '1'] for bit in bitstring):
        raise ValueError("Bitstring contains characters other than '0' and '1'.")
    """Check if each substring contains exactly one '1'."""
    substrings = [bitstring[i:i+substring_size] for i in range(0, len(bitstring), substring_size)]
    return all(sub.count('1') == 1 for sub in substrings)

def calculate_bitstring_energy(bitstring, hamiltonian, backend=None):
    """
    Calculate the energy of a given bitstring for a specified Hamiltonian.

    Args:
        bitstring (str): The bitstring for which to calculate the energy.
        hamiltonian (SparsePauliOp): The Hamiltonian operator of the system, defined as a SparsePauliOp.
        backend (qiskit.providers.Backend): The quantum backend to execute circuits.

    Returns:
        float: The calculated energy of the bitstring.
    """
    # little endian
    
    bitstring = bitstring[::-1]
    # Prepare the quantum circuit for the bitstring
    num_qubits = len(bitstring)
    qc = QuantumCircuit(num_qubits)
    for i, char in enumerate(bitstring):
        if char == '1':
            qc.x(i)  # Apply X gate if the bit in the bitstring is 1
    
    
    # Use Aer's statevector simulator if no backend provided
    if backend is None:
        backend = Aer.get_backend('aer_simulator_statevector')

    qc = transpile(qc, backend)
    estimator = Estimator()
    resultt = estimator.run(observables=[hamiltonian], circuits=[qc], backend=backend).result()

    return resultt.values[0].real

def calculate_bitstring_energy_efficient(bitstring, hamiltonian):
    # little endian
    bitstring = bitstring[::-1]

    def calculate_matrix_element(i, H):

        basis_states = np.array([[1, 0], [0, 1]])

        pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        state = basis_states[i]
 
        matrix_element = np.dot(state, np.dot(pauli_matrices[str(H)], state))

        return matrix_element
    
    expval = 0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        coeff_pauli = 1
        for index, bit in enumerate(bitstring):
            coeff_pauli *= calculate_matrix_element(int(bit), pauli[index])
            if coeff_pauli == 0:
                break

        expval += coeff * coeff_pauli
            



    return expval

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {value}, {e}")
        return None
    

    
def get_ground_state_sparse_diag(H, num_qubits):



    Z_matrix = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)

    def construct_operator(qubit_indices, num_qubits):
        operator = np.eye(1)
        for qubit in range(num_qubits):
            if qubit in qubit_indices:
                operator = np.kron(operator, Z_matrix)
            else:
                operator = np.kron(operator, identity)
        return operator

    C = np.zeros((2**num_qubits, 2**num_qubits))

    for i in range(num_qubits):
        operator = construct_operator([i], num_qubits)
        C += H[i][i] * operator

    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            operator = construct_operator([i, j], num_qubits)
            C += H[i][j] * operator





    eigenvalues, eigenvectors = eigsh(C, k=num_qubits, which='SA')

    return eigenvalues[0]

def get_ground_state_efficient(num_res, num_rot):
    ## generate all Hamming weight preserving bitstrings for the given number of rotamers and residues
    def generate_bitstrings(num_res, num_rot):
        # Generate a single bitstring of length num_rot with one '1'
        single_bitstrings = ['0'*i + '1' + '0'*(num_rot-i-1) for i in range(num_rot)]
        
        # Generate all combinations of these single bitstrings
        bitstrings = [''.join(p) for p in product(single_bitstrings, repeat=num_res)]
        
        return bitstrings
    
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    # Calculate the energy of each bitstring

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    # Find the bitstring with the lowest energy

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)
    ground_state = bitstrings[min_energy_index]

    return min_energy, ground_state

def get_min_and_max_states_efficient(num_res, num_rot):
    ## generate all Hamming weight preserving bitstrings for the given number of rotamers and residues
    def generate_bitstrings(num_res, num_rot):
        # Generate a single bitstring of length num_rot with one '1'
        single_bitstrings = ['0'*i + '1' + '0'*(num_rot-i-1) for i in range(num_rot)]
        
        # Generate all combinations of these single bitstrings
        bitstrings = [''.join(p) for p in product(single_bitstrings, repeat=num_res)]
        
        return bitstrings
    
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    # Calculate the energy of each bitstring

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    # Find the bitstring with the lowest energy

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)

    max_energy = max(energies)
    max_energy_index = energies.index(max_energy)
    min_state = bitstrings[min_energy_index]
    max_state = bitstrings[max_energy_index]

    return min_energy, min_state, max_energy, max_state

def get_ground_state_estimator(num_res, num_rot):
    ## generate all Hamming weight preserving bitstrings for the given number of rotamers and residues
    def generate_bitstrings(num_res, num_rot):
        # Generate a single bitstring of length num_rot with one '1'
        single_bitstrings = ['0'*i + '1' + '0'*(num_rot-i-1) for i in range(num_rot)]
        
        # Generate all combinations of these single bitstrings
        bitstrings = [''.join(p) for p in product(single_bitstrings, repeat=num_res)]
        
        return bitstrings
    
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    # Calculate the energy of each bitstring

    energies = [calculate_bitstring_energy(bitstring, q_hamiltonian) for bitstring in bitstrings]

    # Find the bitstring with the lowest energy

    min_energy = min(energies)
    min_energy_index = energies.index(min_energy)
    ground_state = bitstrings[min_energy_index]

    return min_energy, ground_state

def get_all_states_efficient(num_res, num_rot):
    ## generate all Hamming weight preserving bitstrings for the given number of rotamers and residues
    def generate_bitstrings(num_res, num_rot):
        # Generate a single bitstring of length num_rot with one '1'
        single_bitstrings = ['0'*i + '1' + '0'*(num_rot-i-1) for i in range(num_rot)]
        
        # Generate all combinations of these single bitstrings
        bitstrings = [''.join(p) for p in product(single_bitstrings, repeat=num_res)]
        
        return bitstrings
    
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    # Calculate the energy of each bitstring

    energies = [calculate_bitstring_energy_efficient(bitstring, q_hamiltonian) for bitstring in bitstrings]

    # Find the bitstring with the lowest energy

    return energies, bitstrings

def get_all_states_estimator(num_res, num_rot):
    ## generate all Hamming weight preserving bitstrings for the given number of rotamers and residues
    def generate_bitstrings(num_res, num_rot):
        # Generate a single bitstring of length num_rot with one '1'
        single_bitstrings = ['0'*i + '1' + '0'*(num_rot-i-1) for i in range(num_rot)]
        
        # Generate all combinations of these single bitstrings
        bitstrings = [''.join(p) for p in product(single_bitstrings, repeat=num_res)]
        
        return bitstrings
    
    bitstrings = generate_bitstrings(num_res, num_rot)

    H = get_hamiltonian(num_rot, num_res)
    num_qubits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits, H)

    # Calculate the energy of each bitstring

    energies = [calculate_bitstring_energy(bitstring, q_hamiltonian) for bitstring in bitstrings]

    # Find the bitstring with the lowest energy

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
    # Number of qubits in the base circuit
    num_qubits = base_circuit.num_qubits

    # Create a new quantum circuit with n times the number of qubits
    product_circuit = QuantumCircuit(num_qubits * n)

    # Append the base circuit to the product circuit n times
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
    

    if num_rot == 2:
        qc = QuantumCircuit(num_rot)
        qc.x(0)
        agate = A_gate(theta=theta)
        qc.compose(agate, [0, 1], inplace=True)
        init_state = create_product_state(qc, num_res)
        return init_state
    elif num_rot == 3:
        qc = QuantumCircuit(num_rot)
        qc.x(1)
        agate = A_gate(theta=theta)
        qc.compose(agate, [0, 1], inplace=True)
        qc.compose(agate, [1, 2], inplace=True)
        
        init_state = create_product_state(qc, num_res)
        return init_state
    else:
        raise ValueError("Number of rotomers not supported.")
    
def aggregate(dist : dict, alpha : float, energies : list) -> float:
            # sort by values
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