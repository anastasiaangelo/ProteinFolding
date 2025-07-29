import os
import json
import pandas as pd
from itertools import product
import ast
import re

from proteinfolding.supporting_functions import get_all_states_efficient, int_to_bitstring, bitstring_to_int, check_hamming

def load_json_files(parent_path):
    # Initialize an empty list to store the data from the json files
    data = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            # Check if the file is a json file
            if file.endswith(".json"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                
                # Open the json file and load the data
                with open(file_path, 'r') as f:
                    data.append(json.load(f))

    # Return the loaded data
    return data




def clean_json_data(results):
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(results)

    # Break down the 'params' column into separate columns
    df_params = pd.DataFrame(df['params'].tolist(), index=df.index)
    
    # Check the length of 'params' to handle 'pos'
    if len(df_params.columns) == 7:
        df[['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field']] = df_params
    elif len(df_params.columns) == 6:
        df[['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos']] = df_params
        df['transverse_field'] = None
    else:
        df[['num_res', 'num_rot', 'alpha', 'shots', 'p']] = df_params
        df['transverse_field'] = None
        df['pos'] = None  # or np.nan

    # Drop the original 'params' column
    df = df.drop('params', axis=1)

    # Move the columns to the left side of the DataFrame
    cols = ['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field']  + [col for col in df if col not in ['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field']]
    df = df[cols]

    # Return the cleaned DataFrame
    return df


# def set_multiindex(df):
#     """
#     Set the five columns originated from the params as the key of the DataFrame.

#     Parameters:
#     df (pandas.DataFrame): The DataFrame to modify.

#     Returns:
#     pandas.DataFrame: The modified DataFrame.
#     """
#     # Set the five columns as the DataFrame's multi-index
#     df.set_index(['num_res', 'num_rot', 'alpha', 'shots', 'p'], inplace=True)

#     # Return the modified DataFrame
#     return df




def filter_data(df, **kwargs):
    """
    Filter a DataFrame based on matching column values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to filter.
    **kwargs: Column-value pairs to filter by. For example, filter_data(df, num_res=5, alpha=0.1).

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    # Start with the entire DataFrame
    filtered_df = df

    # For each column-value pair
    for col, value in kwargs.items():
        # Filter the DataFrame
        filtered_df = filtered_df[filtered_df[col] == value]

    # Return the filtered DataFrame
    return filtered_df


import numpy as np

# def clean_intermediate_data_column(df):
#     """
#     Clean the 'intermediate_data' column of a DataFrame by breaking the data into a list of dictionaries.

#     Parameters:
#     df (pandas.DataFrame): The DataFrame to clean.

#     Returns:
#     pandas.DataFrame: The cleaned DataFrame.
#     """
#     # Define a function to flatten the list of lists
#     def flatten(data):
#         return [item for sublist in data for item in sublist]

#     # Apply the flatten function to the 'intermediate_data' column
#     df['intermediate_data'] = df['intermediate_data'].apply(flatten)

#     # Return the cleaned DataFrame
#     return df

import numpy as np

def clean_intermediate_data(df):
    """
    Clean the 'intermediate_data' column of a DataFrame by breaking the data into a list of dictionaries.

    Parameters:
    df (pandas.DataFrame): The DataFrame to clean.

    Returns:
    pandas.DataFrame: The cleaned DataFrame.
    """
    # Define a function to flatten the list of lists
    def flatten(data):
        flattened = [item for sublist in data for item in sublist]
        return flattened if flattened else None

    # Apply the flatten function to the 'intermediate_data' column
    df['intermediate_data'] = df['intermediate_data'].apply(flatten)

    # Return the cleaned DataFrame
    return df

def get_unique_nres_and_nrot_values(df):
    """
    Get the unique values of 'num_res' and 'num_rot' from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to extract unique values from.

    Returns:
    tuple: A tuple containing two NumPy arrays, the unique 'num_res' and 'num_rot' values.
    """
    # Get the unique values of 'num_res' and 'num_rot'
    unique_nres = df['num_res'].unique()
    unique_nrot = df['num_rot'].unique()

    # Return the unique values as NumPy arrays
    return unique_nres, unique_nrot

def get_unique_alpha_and_p_values(df):
    """
    Get the unique values of 'alpha' and 'p' from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to extract unique values from.

    Returns:
    tuple: A tuple containing two NumPy arrays, the unique 'alpha' and 'p' values.
    """
    # Get the unique values of 'alpha' and 'p'
    unique_alpha = df['alpha'].unique()
    unique_p = df['p'].unique()

    # Return the unique values as NumPy arrays
    return unique_alpha, unique_p

def get_unique_alpha_and_shot_values(df):
    """
    Get the unique values of 'alpha' and 'shots' from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to extract unique values from.

    Returns:
    tuple: A tuple containing two NumPy arrays, the unique 'alpha' and 'p' values.
    """
    # Get the unique values of 'alpha' and 'p'
    unique_alpha = df['alpha'].unique()
    unique_shots = df['shots'].unique()

    # Return the unique values as NumPy arrays
    return unique_alpha, unique_shots



def generate_exact_energies_data(res_list, rot_list):
    # Initialize an empty list to store DataFrame rows
    rows = []

    # Iterate over all possible pairs of residues and rotamers
    for num_res, num_rot in product(res_list, rot_list):
        # Get the energies and bitstrings
        energies, bitstrings = get_all_states_efficient(num_res, num_rot)

        # Append the results to the list as a dictionary (which will become a DataFrame row)
        rows.append({'num_res': num_res, 'num_rot': num_rot, 'energies': energies, 'bitstrings': bitstrings})

    # Convert the list of rows to a DataFrame
    df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)

    return df




def find_min_energy_and_bitstring_from_exact_energy_dataframe(df_exact, nres, nrot):
    # Filter the dataframe for the given num_res and num_rot
    df_filtered = df_exact[(df_exact['num_res'] == nres) & (df_exact['num_rot'] == nrot)]
    
    # If there are multiple rows that satisfy the condition, raise an exception
    if len(df_filtered) > 1:
        raise Exception(f"Multiple rows found for num_res = {nres} and num_rot = {nrot}")
    
    # Get the energy and its corresponding bitstring from the row
    # Get the energy and its corresponding bitstring from the row
    energy = ast.literal_eval(df_filtered.iloc[0]['energies'])
    corresponding_bitstring = ast.literal_eval(df_filtered.iloc[0]['bitstrings'])
    
    # Convert the energies to complex numbers and get their real parts
    energy = [complex(e).real for e in energy]

    # If 'energies' or 'bitstrings' is a list, get the minimum energy and its corresponding bitstring
    if isinstance(energy, list):
        min_index = energy.index(min(energy))
        
        energy = energy[min_index]
        corresponding_bitstring = corresponding_bitstring[min_index]
    
    return energy, corresponding_bitstring

def find_max_energy_and_bitstring_from_exact_energy_dataframe(df_exact, nres, nrot):
    # Filter the dataframe for the given num_res and num_rot
    df_filtered = df_exact[(df_exact['num_res'] == nres) & (df_exact['num_rot'] == nrot)]
    
    # If there are multiple rows that satisfy the condition, raise an exception
    if len(df_filtered) > 1:
        raise Exception(f"Multiple rows found for num_res = {nres} and num_rot = {nrot}")
    
    # Get the energy and its corresponding bitstring from the row
    # Get the energy and its corresponding bitstring from the row
    energy = ast.literal_eval(df_filtered.iloc[0]['energies'])
    corresponding_bitstring = ast.literal_eval(df_filtered.iloc[0]['bitstrings'])
    
    # Convert the energies to complex numbers and get their real parts
    energy = [complex(e).real for e in energy]

    # If 'energies' or 'bitstrings' is a list, get the minimum energy and its corresponding bitstring
    if isinstance(energy, list):
        max_index = energy.index(max(energy))
        
        energy = energy[max_index]
        corresponding_bitstring = corresponding_bitstring[max_index]
    
    return energy, corresponding_bitstring

# def find_number_of_iterations_to_find_ground_state(df_filtered, df_exact, **kwargs):
#     # If there are multiple rows that satisfy the condition, raise an exception
#     if len(df_filtered) > 1:
#         raise Exception(f"Multiple rows found for {kwargs}")
    
#     # Call the find_min_energy_and_bitstring_from_exact_energy_dataframe function
#     min_energy, corresponding_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(df_exact, kwargs['num_res'], kwargs['num_rot'])
    
#     # Get the 'intermediate_data' from the row
#     intermediate_data = ast.literal_eval(df_filtered.iloc[0]['intermediate_data'])
    
#     total_bits = kwargs['num_res'] * kwargs['num_rot']
#     # Loop through the list in order and find the index of the dict that contains the correct bitstring
#     for i, data in enumerate(intermediate_data):
#         for key in data.keys():
#             if int_to_bitstring(int(key), total_bits) == corresponding_bitstring:
#                 return i

#     return None

def calculate_approximation_ratio(distribution : dict, min_energy : float, max_energy : float, tail:float):
    """Given a distribution, calculate the approximation ratio."""
    



def find_min_shots_to_AR(df, approximation_ratio=1):
    # Create df_filtered similar to temp_func
    df_filtered = df


    # Sort df_filtered by 'shots'
    df_filtered = df_filtered.sort_values(by='shots')

    return df_filtered


def find_number_of_shots_to_find_ground_state(df, df_exact, **kwargs):
    # Create df_filtered similar to temp_func
    df_filtered = df
    for key, value in kwargs.items():
        if key != 'shots':
            df_filtered = df_filtered[df_filtered[key] == value]
    
    # Sort df_filtered by 'shots'
    df_filtered = df_filtered.sort_values(by='shots')
    
    # Starting from the smallest value of shots, call temp_func
    for nshots in df_filtered['shots'].unique():
        df2 = df_filtered[df_filtered['shots'] == nshots]
        if len(df2) > 1:
            raise Exception(f"Multiple rows found for {kwargs}")
        
        # Call the find_min_energy_and_bitstring_from_exact_energy_dataframe function
        min_energy, corresponding_bitstring = find_min_energy_and_bitstring_from_exact_energy_dataframe(df_exact, kwargs['num_res'], kwargs['num_rot'])

        # Get the 'intermediate_data' from the row
        intermediate_data = ast.literal_eval(df2.iloc[0]['intermediate_data'])
        
        total_bits = kwargs['num_res'] * kwargs['num_rot']

        index = None 

        break_outer = False

        for i, data in enumerate(intermediate_data):
            for key in data.keys():
                if int_to_bitstring(int(key), total_bits) == corresponding_bitstring:
                    index = i
                    break_outer = True
                    break
            if break_outer:
                break


        if index is not None:
            index += 1
            return index, nshots, index * nshots
        
    return None, None, None
    


def generate_minimum_shots_to_ground_state_data(df, df_exact):
    # Initialize an empty DataFrame

    # find unique values of num_res, num_rot, alpha, p
    unique_num_res = df['num_res'].unique()
    unique_num_rot = df['num_rot'].unique()
    unique_alpha = df['alpha'].unique()
    unique_p = df['p'].unique()

    df_min_iter = pd.DataFrame(columns=['num_res', 'num_rot', 'alpha', 'p', 'min_iter', 'shots', 'min_shots'])

    # create a DataFrame with all possible combinations of num_res, num_rot, alpha, p
    for num_res, num_rot, alpha, p in product(unique_num_res, unique_num_rot, unique_alpha, unique_p):
        num_iter, shots, num_shots = find_number_of_shots_to_find_ground_state(df, df_exact, num_res=num_res, num_rot=num_rot, alpha=alpha, p=p)

        row = {'num_res': num_res, 'num_rot': num_rot, 'alpha': alpha, 'p': p, 'min_iter': num_iter, 'shots': shots, 'min_shots': num_shots}

        df_min_iter = pd.concat([df_min_iter, pd.DataFrame([row])])

    # columns=['num_res', 'num_rot', 'alpha', 'p', 'shots', 'intermediate_data']
    # columns_new = ['num_res', 'num_rot', 'alpha', 'p', 'shots', 'min_iter', 'min_shots']

    # df_min_shots = pd.DataFrame(columns=columns_new)

    # # drop all columns of df except the ones specified in columns
    # df = df[columns]

    # for index, row in df.iterrows():
    #     row_dict = row.to_dict()
    #     min_iter = find_number_of_iterations_to_find_ground_state(df, df_exact, **row_dict)
    #     row_dict['min_iter'] = min_iter
    #     if min_iter is not None:
    #         row_dict['min_shots'] = row_dict['shots'] + min_iter
    #     else:
    #         row_dict['min_shots'] = None

    #     df_min_shots = pd.concat([df_min_shots, pd.DataFrame([row_dict])])

    # # drop intermediate_data column
    # df_min_shots = df_min_shots.drop('intermediate_data', axis=1)

    return df_min_iter



def remove_hamming_viloations_from_intermetiate_data(df):
    # Iterate over each row in df
    for index, df_row in df.iterrows():
        # Convert the 'intermediate_data' from string to list of dictionaries
        intermediate_data_list = ast.literal_eval(df_row['intermediate_data'])

        # Create a new list to store the modified dictionaries
        new_intermediate_data_list = []

        for i in range(len(intermediate_data_list)):
            intermediate_data_dict = intermediate_data_list[i]
            # Create a copy of the dictionary to modify while iterating
            intermediate_data_dict_copy = intermediate_data_dict.copy()



            # For each key in the dictionary, check if it passes the check_hamming function
            for int_bitstring in intermediate_data_dict.keys():
                int_bitstr = int(int_bitstring)
                bitstring = int_to_bitstring(int_bitstr, df_row['num_res'] * df_row['num_rot'])
                if not check_hamming(bitstring, df_row['num_rot']):
                    # If the bitstring does not pass the check_hamming function, remove it from the dictionary
                    del intermediate_data_dict_copy[int_bitstring]
                
                    

            # Append the modified dictionary to the new list
            
            new_intermediate_data_list.append(intermediate_data_dict_copy)

        # Update the 'intermediate_data' row in df with the new list of dictionaries
        df.at[index, 'intermediate_data'] = str(new_intermediate_data_list)

    return df


from proteinfolding.supporting_functions import int_to_bitstring



def find_initial_and_final_probability_distributions(df, df_exact):

    def get_init_dist_and_energies(df_exact, df):
        energy_col = []
        init_dist_col = []
        # Iterate over each row in df
        for i, df_row in df.iterrows():
            
        
        # Find the row in df_exact that matches num_res and num_rot
            matching_rows = df_exact[(df_exact['num_res'] == df_row['num_res']) & (df_exact['num_rot'] == df_row['num_rot'])]

            # Check whether there is only one such row, throw an error otherwise
            if len(matching_rows) != 1:
                raise ValueError("There should be exactly one matching row in df_exact")
            
            init_dists = df_row['intermediate_data']

            if isinstance(init_dists, str):
                init_dists = ast.literal_eval(init_dists)

            init_dist = init_dists[0]

            init_energy = []

            exact_row_bitrtrings = np.array(ast.literal_eval(matching_rows.iloc[0].values[3]))

            exact_row_energies = np.array(ast.literal_eval(matching_rows.iloc[0].values[2]))
            
            for int_bitstring in init_dist.keys():
                
                bitstring = int_to_bitstring(int(int_bitstring), df_row['num_res'] * df_row['num_rot'])
                index = np.where(exact_row_bitrtrings == bitstring)
                if len(index[0]) == 0:
                    init_energy.append(None)
                elif len(index[0]) > 1:
                    print(f"bitstring {bitstring} int bitstr {int_bitstring} found multiple times in {exact_row_bitrtrings}")
                    continue
                else:
                    init_energy.append(np.real(exact_row_energies[index[0][0]]))

            energy_col.append(init_energy)
            init_dist_col.append(init_dist)
                
        # convert the energies to complex numbers and get their real parts
        #init_energy = [complex(e).real for e in init_energy] 

        # convert energies to a dataframe column
        init_energy = pd.DataFrame({'init_energy' : energy_col})

        # convert the final distribution to a dataframe column
        init_dist = pd.DataFrame({'init_dist' : init_dist_col})


        return init_dist, init_energy
    
    def get_final_dist_and_energies(df_exact, df):
        
        energy_col = []
        final_dist_col = []
        # Iterate over each row in df
        for i, df_row in df.iterrows():
            
            # Find the row in df_exact that matches num_res and num_rot
            matching_rows = df_exact[(df_exact['num_res'] == df_row['num_res']) & (df_exact['num_rot'] == df_row['num_rot'])]

            # Check whether there is only one such row, throw an error otherwise
            if len(matching_rows) != 1:
                raise ValueError("There should be exactly one matching row in df_exact")
            
            final_dists = df_row['intermediate_data']

            if isinstance(final_dists, str):
                final_dists = ast.literal_eval(final_dists)

            final_dist = final_dists[-1]
            
            

            

            final_energy = []

            exact_row_bitrtrings = np.array(ast.literal_eval(matching_rows.iloc[0].values[3]))
            
            exact_row_energies = np.array(ast.literal_eval(matching_rows.iloc[0].values[2]))
            
            for int_bitstring in final_dist.keys():
                
                bitstring = int_to_bitstring(int(int_bitstring), df_row['num_res'] * df_row['num_rot'])
                index = np.where(exact_row_bitrtrings == bitstring)
                if len(index[0]) == 0:
                    final_energy.append(None)
                elif len(index[0]) > 1:
                    print(f"bitstring {bitstring} int bitstr {int_bitstring} found multiple times in {exact_row_bitrtrings}")
                    continue
                else:
                    final_energy.append(np.real(exact_row_energies[index[0][0]]))

            energy_col.append(final_energy)
            final_dist_col.append(final_dist)

        # convert the energies to complex numbers and get their real parts
        #final_energy = [complex(e).real for e in final_energy]

        # convert energies to a dataframe column
        final_energy = pd.DataFrame({'final_energy' : energy_col})

        # convert the final distribution to a dataframe column
        final_dist = pd.DataFrame({'final_dist' : final_dist_col})
            



        return final_dist, final_energy

    



    
    


    df_new = pd.DataFrame()

    df_new['num_res'] = df['num_res']
    df_new['num_rot'] = df['num_rot']
    df_new['alpha'] = df['alpha']
    df_new['shots'] = df['shots']
    df_new['p'] = df['p']
    if 'pos' in df.columns:
        print("pos column found")
        df_new['pos'] = df['pos']
    if 'transverse_field' in df.columns:
        print("transverse_field column found")
        df_new['transverse_field'] = df['transverse_field']

    df1, df2 = get_init_dist_and_energies(df_exact, df)
    df_new['init_dist'] = df1
    df_new['init_energies'] = df2

    df1, df2 = get_final_dist_and_energies(df_exact, df)
    df_new['final_dist'] = df1
    df_new['final_energies'] = df2


    return df_new

def find_parameter_evolution(df):
    """
    Find the evolution of the parameters for each row in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    pandas.DataFrame: A new DataFrame with the parameter evolution columns.
    """
    # Create a new DataFrame to store the parameter evolution
    df_new = pd.DataFrame()

    # Copy the 'num_res', 'num_rot', 'alpha', 'shots', and 'p' columns to the new DataFrame
    df_new['num_res'] = df['num_res']
    df_new['num_rot'] = df['num_rot']
    df_new['alpha'] = df['alpha']
    df_new['shots'] = df['shots']
    df_new['p'] = df['p']

    # Check if the 'pos' column exists in the DataFrame
    if 'pos' in df.columns:
        df_new['pos'] = df['pos']
    if 'transverse_field' in df.columns:
        df_new['transverse_field'] = df['transverse_field']
    
    df_new['parameters'] = df['parameters']

    def flatten(data):
        flattened = [item for sublist in data for item in sublist]
        return flattened if flattened else None

    df_new['parameters'] = df_new['parameters'].apply(ast.literal_eval)
    # Apply the flatten function to the 'intermediate_data' column
    df_new['parameters'] = df_new['parameters'].apply(flatten)

    return df_new

    #

def find_probability_of_ground_state(df, df_exact, transverse_field_bool=False, pos_bool=False):
    """
    Find the probability of the ground state for each row in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    pandas.DataFrame: A new DataFrame with the probability of the ground state column.
    """
# Group the DataFrame by the combination of 'num_res', 'num_rot', 'alpha', 'shots', 'p'
    if transverse_field_bool and pos_bool:
        grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field'])
    elif pos_bool:
        grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos'])
    else:
        grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p'])

    
    # For each group
    for name, group in grouped:

        for i, row in group.iterrows():
            # Convert the 'init_energies' and 'init_dist' columns from strings to lists/dictionaries
            init_energies = row['init_energies']

            if isinstance(init_energies, str):
                init_energies = ast.literal_eval(init_energies)

            init_dist = row['init_dist']

            if isinstance(init_dist, str):
                init_dist = ast.literal_eval(init_dist)

            final_energies = row['final_energies']

            if isinstance(final_energies, str):
                final_energies = ast.literal_eval(final_energies)

            final_dist = row['final_dist']

            if isinstance(final_dist, str):
                final_dist = ast.literal_eval(final_dist)

            # Filter out None values from the energies and their corresponding probabilities
            init_energies_list, init_dist_list = zip(*[(e, p) for e, p in zip(init_energies, list(init_dist.values())) if e is not None])
            final_energies_list, final_dist_list = zip(*[(e, p) for e, p in zip(final_energies, list(final_dist.values())) if e is not None])

            min_en_key_final = list(final_dist_list)[0]
            min_en_final = final_energies_list[0]

            print(f"nres {row['num_res']} nrot {row['num_rot']} alpha {row['alpha']} shots {row['shots']} p {row['p']} min_en_key_final {min_en_key_final} min_en_final {min_en_final}")

