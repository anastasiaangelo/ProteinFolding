from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent # This is your Project Root

# scripts

SCRIPTS_DIR = ROOT_DIR / 'scripts'

# production

PRODUCTION_DIR = ROOT_DIR / 'production'



# data/raw
PYROSETTA_ENERGY_DATA_DIR = ROOT_DIR / 'data/raw/pyrosetta_energy_files_immutable'
PYROSETTA_ENERGY_DATA_DIR_MUTABLE = ROOT_DIR / 'data/raw/pyrosetta_energy_files'
PYROSETTA_PDB_DATA_DIR = ROOT_DIR / 'data/raw/pyrosetta_pdb_files'

QISKT_NOISE_MODEL_DIR = ROOT_DIR / 'data/raw/qiskit_noise_models'

SCP_PRODUCTION_RUNS_DIR = ROOT_DIR / 'data/raw/scp_production_runs'



# data/processed
SCP_PROCESSED_DATA_DIR = ROOT_DIR / 'data/processed/scp_production_runs'
XY_QAOA_DATA_DIR = ROOT_DIR / 'data/processed/xy_qaoa'


EXACT_DATA_DIR = ROOT_DIR / 'data/processed/exact'


# plots

PLOTS_DIR = ROOT_DIR / 'plots'
XY_QAOA_PLOTS_DIR = PLOTS_DIR / 'xy_qaoa'
XY_QAOA_PLOTS_FRACTION_DIR = XY_QAOA_PLOTS_DIR / 'fraction'
XY_QAOA_PLOTS_MIN_SHOTS_TO_GS_DIR = XY_QAOA_PLOTS_DIR / 'min_shots_to_gs'
XY_QAOA_PLOTS_PROBABILITY_DISTRIBUTIONS_DIR = XY_QAOA_PLOTS_DIR / 'probability_distributions'
XY_QAOA_PLOTS_PARAMETERS_EVOLUTION_DIR = XY_QAOA_PLOTS_DIR / 'parameters_evolution'
XY_QAOA_PLOTS_GROUND_STATE_PROBABILITY_DIR = XY_QAOA_PLOTS_DIR / 'ground_state_probability'
XY_QAOA_PLOTS_CVAR_AGGREGATION_DIR = XY_QAOA_PLOTS_DIR / 'cvar_aggregation'

# filenames

HAMMING_WEIGHT_PRESERVING_DATA_FILE = 'hamming_weight_preserving_data.csv'
MIN_SHOTS_TO_FIND_GS_FILE = 'min_shots_to_find_gs.csv'
INITIAL_AND_FINAL_PROBABILITY_DISTRIBUTIONS_FILE = 'initial_and_final_probability_distributions.csv'
GROUND_STATE_PROBABILITY_FILE = 'ground_state_probability.csv'
TAIL_PROBABILITY_FILE = 'tail_probability.csv'
CVAR_AGGREGATION_FILE = 'cvar_aggregation.csv'
PARAMETERS_EVOLUTION_FILE = 'parameters_evolution.csv'
EXACT_DATA_ENERGY_BITSTRING_FILE = EXACT_DATA_DIR / 'exact_energies_and_bitstrings.csv.gz'
CONFIG_FILE = PRODUCTION_DIR / 'config.ini'
SIMULATION_SUMMARY_FILE = SCP_PRODUCTION_RUNS_DIR / 'simulation_summary.txt'
