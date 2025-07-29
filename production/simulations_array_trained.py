import configparser
import ast
import os
import sys
from dotenv import load_dotenv
from proteinfolding.supporting_functions import get_hyperparameters
from proteinfolding.logging_utils import log_info
import proteinfolding.simulations_production as SP

load_dotenv()

class Simulation:
    def __init__(self, match, simulation_name):
        print(f"Initializing Simulation with simulation_name: {repr(simulation_name)}")
        self.simulation_name = simulation_name.strip()

        # Load configuration
        config_path = "/home/b/aag/proteinfolding/production/config.ini"
        # config_path = "/u/aag/proteinfolding/production/configs/config_1.ini"
        self.config = configparser.ConfigParser()
        self.config.optionxform = str  # Preserve case sensitivity
        self.config.read(config_path)

        # Normalize section names to avoid whitespace mismatches
        normalized_sections = {s.strip(): s for s in self.config.sections()}
        print(f"Normalized Sections: {list(normalized_sections.keys())}")  # Debugging

        if self.simulation_name not in normalized_sections:
            raise ValueError(
                f"Error: Section '{self.simulation_name}' not found in config file.\n"
                f"Available Sections: {list(normalized_sections.keys())}"
            )

        self.simulation_name = normalized_sections[self.simulation_name]

        # Extract parameters safely
        self.params = {}
        for k, v in self.config[self.simulation_name].items():
            try:
                # Use ast.literal_eval only if it looks like a Python literal
                self.params[k] = ast.literal_eval(v)
            except (SyntaxError, ValueError):
                # If it fails, treat it as a string
                self.params[k] = v.strip()

        # try:
        #     self.params = {k: ast.literal_eval(v) for k, v in self.config[self.simulation_name].items()}
        # except (SyntaxError, ValueError) as e:
        #     raise ValueError(f"Error parsing values in section '{self.simulation_name}': {e}")

        self.match = match
        # self.hyperparameters = get_hyperparameters(self.match, *self.params.values())

        # Separate `SIMULATION_ID` to pass it manually
        simulation_id = self.params.pop("SIMULATION_ID", None)  # Extract SIMULATION_ID if present
        self.hyperparameters = get_hyperparameters(self.match, *self.params.values())
        if simulation_id is not None:
            self.hyperparameters += (simulation_id,)

        print(f"Simulation Parameters: {self.params}")  # Debugging
        print(f"Initialized Simulation with match: {match}, simulation_name: {self.simulation_name}")


    def noisy_xy_simulation_parameter_sweep(self):
        num_rot, num_res, shots, alpha, p = self.hyperparameters
        log_info(f"Running noisy_xy_simulation_parameter_sweep with num_rot: {num_rot}, num_res: {num_res}, shots: {shots}, alpha: {alpha}, p: {p}")
        SP.noisy_simulation_XY(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p)

    def noisy_xy_simulation_init_bitstring_scan(self):
        num_rot, num_res, shots, alpha, p, pos = self.hyperparameters
        log_info(f"Running noisy_xy_simulation_init_bitstring_scan with num_rot: {num_rot}, num_res: {num_res}, shots: {shots}, alpha: {alpha}, p: {p}, pos: {pos}")
        SP.noisy_simulation_XY(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p, pos=pos)

    def statevector_simulation_parameter_sweep_with_random_param_init_all_valid_bitstring_init_parallel_new(self):
        num_rot, num_res, shots, alpha, p = self.hyperparameters
        log_info(f"Running statevector_simulation_parameterr_sweep with num_rot: {num_rot}, num_res: {num_res}, shots: {shots}, alpha: {alpha}, p: {p}")
        SP.statevector_simulation_XY_parallel_new(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p, transverse_field=1, pos=0)
          
    def statevector_simulation_parallel(self):
        num_rot, num_res, shots, alpha, p, SIMULATION_ID = self.hyperparameters
        log_info(f"Running statevector_simulation_parallel with num_rot: {num_rot}, num_res: {num_res}, shots: {shots}, alpha: {alpha}, p: {p}")
        if SIMULATION_ID:
            SP.statevector_simulation_XY_parallel_new(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p, simulation_id=SIMULATION_ID)
        else:
            SP.statevector_simulation_XY_parallel_new(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p)

    def statevector_simulation_parallel_trained(self):
        num_rot, num_res, shots, alpha, p, SIMULATION_ID= self.hyperparameters
        log_info(f"Running statevector_simulation_parallel with num_rot: {num_rot}, num_res: {num_res}, shots: {shots}, alpha: {alpha}, p: {p}")
        SP.statevector_simulation_XY_parallel_trained(num_rot=num_rot, num_res=num_res, shots=shots, alpha=alpha, p=p, ignore_shots=True, simulation_id=SIMULATION_ID)
        
        

if __name__ == "__main__":
    log_path = os.path.join(os.getcwd(), 'log.txt')
    
    git_commit = sys.argv[1]
    match = int(sys.argv[2])
    task_id = int(os.getenv("LSF_JOBINDEX", 0))  

    log_info(f"Git commit: {git_commit}")

    simulation_name = "statevector_simulation_parallel_trained"

    # Read the config file
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity
    config.read("/home/b/aag/proteinfolding/production/config.ini")
    # config.read("/u/aag/proteinfolding/production/configs/config_1.ini")

    param_list = []
    
    for num_rot in ast.literal_eval(config[simulation_name]["num_rot_arr"]):
        for num_res in ast.literal_eval(config[simulation_name]["num_res_arr"]):
            for shots in ast.literal_eval(config[simulation_name]["shots_arr"]):
                for alpha in ast.literal_eval(config[simulation_name]["alpha_arr"]):
                    for p in ast.literal_eval(config[simulation_name]["p_arr"]):
                        param_list.append((num_rot, num_res, shots, alpha, p))

    if task_id >= len(param_list):
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID {task_id}, max {len(param_list) - 1}")

    selected_params = param_list[task_id]
    print(f"Running simulation {task_id} with parameters: {selected_params}")

    # Initialize the simulation object with the selected parameters
    simulation = Simulation(match, simulation_name)

    try:
        method = getattr(simulation, simulation_name)  # Get the method
        print(f"Calling method: {method}")  # Debugging
        method() 
        log_info(f"Successfully ran the simulation: {simulation_name}")

    except AttributeError as e:
        print(f"Error: The simulation '{simulation_name}' does not exist.")
        log_info(f"Error: The simulation '{simulation_name}' does not exist.")
        print(f"Actual Exception: {e}")  # Debugging - Print real exception

    except Exception as e:
        print(f"Unexpected error occurred while running simulation '{simulation_name}': {e}")
        log_info(f"Unexpected error: {e}")
