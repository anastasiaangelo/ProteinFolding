import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast 
import math

from proteinfolding.data_processing import find_number_of_shots_to_find_ground_state, get_unique_nres_and_nrot_values, get_unique_alpha_and_p_values, get_unique_alpha_and_shot_values
from proteinfolding.supporting_functions import int_to_bitstring
from scipy.optimize import curve_fit

class DataPlotter:
    def __init__(self, df):
        self.df = df

    def filter_dataframe(self, num_res, num_rot):
        return self.df[(self.df['num_res'] == num_res) & (self.df['num_rot'] == num_rot)]
    
    def generate_plots_of_fractions_for_all_res_rot_pairs(self, save_path=None, xaxis='shots'):
        # Get unique pairs of num_res and num_rot
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        num_res_values, num_rot_values = get_unique_nres_and_nrot_values(self.df)

        for num_res in num_res_values:
            for num_rot in num_rot_values:
                # Filter the dataframe
                df_filtered = self.filter_dataframe(num_res, num_rot)

                # Check if the filtered dataframe is empty
                if not df_filtered.empty:
                    # Create a new figure
                    plt.figure()
                    
                    # Plot the data
                    if xaxis == 'shots':
                        self.plot_fraction_vs_shots(df_filtered, save_path)
                    elif xaxis == 'p':
                        self.plot_fraction_vs_p(df_filtered, save_path)
                    #self.plot_fraction_vs_shots(df_filtered, save_path)


    def plot_fraction_vs_shots(self, df_filtered, save_path=None):
        # Get unique alpha and p values
        alpha_values, p_values = get_unique_alpha_and_p_values(df_filtered)

        # Define colors, linestyles, and markers
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
        linestyles = ['-', '--', '-.', ':'] * len(p_values)
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(p_values)

        # Create a plot for each combination of alpha and p
        for i, alpha in enumerate(alpha_values):
            for j, p in enumerate(p_values):
                # Filter the dataframe based on alpha and p
                df_filtered_alpha_p = df_filtered[(df_filtered['alpha'] == alpha) & (df_filtered['p'] == p)]
                
                # Sort the dataframe by the 'shots' column
                df_filtered_alpha_p = df_filtered_alpha_p.sort_values(by='shots')

                

                # Plot fraction vs number of shots
                plt.plot(df_filtered_alpha_p['shots'], df_filtered_alpha_p['fraction'], label=f'alpha={alpha}, p={p}', color=colors[i], linestyle=linestyles[j], marker=markers[j])

        # Set plot title and labels
        plt.title('Fraction vs Number of Shots')
        plt.xlabel('Number of Shots')
        plt.ylabel('Fraction')

        # Add gridlines
        plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        # Set the background color of the plot to white
        plt.gca().set_facecolor('white')

        # Set the edge color of the plot to black
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['top'].set_color('black') 
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['left'].set_color('black')

        # Add legend
        plt.legend()

        # Save the plot
        if save_path:
            # Check if the directory exists
            if not os.path.exists(save_path):
                # If the directory doesn't exist, create it
                os.makedirs(save_path)

            plt.savefig(str(save_path) + f"/fraction_vs_shots_num_res_{df_filtered['num_res'].iloc[0]}_num_rot_{df_filtered['num_rot'].iloc[0]}.pdf")
    
    #def generate_plots_of_min_shots_to_find_ground_state(self, save_path=None):


    def plot_p_vs_min_shots_per_structure(self, save_path=None):
        """
        Plots the impact of 'p' on 'min_shots' for each unique structure (num_res, num_rot),
        and arranges the plots in a square grid layout. Skips plots where min_shots has NaN values.
        """

        unique_structures = self.df[['num_res', 'num_rot']].drop_duplicates()
        
        valid_structures = []
        for _, (num_res, num_rot) in unique_structures.iterrows():
            df_filtered = self.df[(self.df['num_res'] == num_res) & (self.df['num_rot'] == num_rot)]
            if df_filtered['min_shots'].notna().any(): 
                valid_structures.append((num_res, num_rot))

        num_plots = len(valid_structures) 

        if num_plots == 0:
            print("No valid structures to plot (all min_shots are NaN).")
            return 

        cols = math.ceil(math.sqrt(num_plots)) 
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows)) 

        axes = axes.flatten() if num_plots > 1 else [axes]

        for ax, (num_res, num_rot) in zip(axes, valid_structures):
            df_filtered = self.df[(self.df['num_res'] == num_res) & (self.df['num_rot'] == num_rot)]
            df_p_analysis = df_filtered.groupby('p')['min_shots'].mean().reset_index()

            if df_p_analysis['min_shots'].isna().all():
                ax.axis('off')
                continue

            ax.plot(df_p_analysis['p'], df_p_analysis['min_shots'], marker='o', linestyle='-')
            ax.set_xlabel("p")
            ax.set_ylabel("Average min_shots")
            ax.grid()

        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "p_vs_minshots_all_structures.pdf"))

        plt.show()
        plt.close()


    def plot_p_vs_min_shots_per_structure_statistics(self, save_path=None):
        """
        Plots the impact of 'p' on 'min_shots' for each unique structure (num_res, num_rot),
        and arranges the plots in a square grid layout. Skips plots where min_shots has NaN values.
        """

        unique_structures = self.df[['num_res', 'num_rot']].drop_duplicates()

        valid_structures = []
        for _, (num_res, num_rot) in unique_structures.iterrows():
            df_filtered = self.df[(self.df['num_res'] == num_res) & (self.df['num_rot'] == num_rot)]
            if df_filtered['min_shots'].notna().any(): 
                valid_structures.append((num_res, num_rot))

        num_plots = len(valid_structures) 

        if num_plots == 0:
            print("No valid structures to plot (all min_shots are NaN).")
            return 

        cols = math.ceil(math.sqrt(num_plots)) 
        rows = math.ceil(num_plots / cols)
        
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 5 * rows)) 

        axes = axes.flatten() if num_plots > 1 else [axes]

        for ax, (num_res, num_rot) in zip(axes, valid_structures):
            df_filtered = self.df[(self.df['num_res'] == num_res) & (self.df['num_rot'] == num_rot)]
            df_p_analysis = df_filtered.groupby('p')['min_shots'].agg(['mean', 'std']).reset_index()

            # if df_p_analysis['min_shots'].isna().all():
            #     ax.axis('off')
            #     continue

            ax.errorbar(
                df_p_analysis['p'], df_p_analysis['mean'], yerr=df_p_analysis['std'],
                fmt='-o', capsize=5, elinewidth=1, markeredgewidth=1
            )

            ax.set_xlabel("p")
            ax.set_ylabel("Average Min Shots")
            ax.grid()

        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "p_vs_minshots_all_structures_statistics.pdf"))

        plt.show()
        plt.close()


    def plot_min_shots_vs_num_qubits_per_p(self, save_path=None):
        """
        Plots min_shots vs num_qubits with different curves for each value of p.

        Parameters:
        - save_path (str, optional): Directory where plots should be saved. If None, the plot is just shown.
        """
        
        plt.figure(figsize=(10, 6))

        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  
        
        unique_p_values = sorted(self.df['p'].unique()) 

        for idx, p in enumerate(unique_p_values):
            df_filtered = self.df[self.df['p'] == p]
            df_qubits_analysis = df_filtered.groupby('num_qubits')['min_shots'].mean().reset_index()

            plt.plot(df_qubits_analysis['num_qubits'], df_qubits_analysis['min_shots'], 
                     marker='o', linestyle=line_styles[idx % len(line_styles)], label=f"p={p}")

        plt.xlabel("Number of Qubits")
        plt.ylabel("Average Min Shots")
        plt.legend(title="p")
        plt.grid()
        
        # Save the plot if a path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "min_shots_vs_num_qubits_per_p.pdf"))

        plt.show()
        plt.close() 


    def plot_min_shots_vs_num_qubits_per_p_statistics(self, save_path=None):
        """
        Plots min_shots vs num_qubits with different curves for each value of p.

        Parameters:
        - save_path (str, optional): Directory where plots should be saved. If None, the plot is just shown.
        """
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        plt.figure(figsize=(11, 10))

        markers = ['o', 's', 'D', '^', 'v', 'p', '*']
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  
        
        unique_p_values = sorted(self.df['p'].unique()) 

        for idx, p in enumerate(unique_p_values):
            df_filtered = self.df[self.df['p'] == p]
            df_qubits_analysis = df_filtered.groupby('num_qubits')['min_shots'].agg(['mean', 'std']).reset_index()

            plt.errorbar(df_qubits_analysis['num_qubits'], df_qubits_analysis['mean'], yerr=df_qubits_analysis['std'],
                     linestyle=line_styles[idx % len(line_styles)], marker=markers[idx % len(markers)], capsize=5, 
                     elinewidth=1, markeredgewidth=1, label=f"p={p}")


        plt.xlabel("Number of Qubits", fontsize=24)
        plt.ylabel("Average Min Shots ± Std Dev", fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(title="p",fontsize=22)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        # Save the plot if a path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "min_shots_vs_num_qubits_per_p.pdf"))

        plt.show()
        plt.close() 

    
    def plot_p_vs_min_shots_per_num_rots_statistics(self, save_path=None):
        """
        Creates separate plots for each unique num_rot value, showing min_shots vs num_qubits 
        with different curves for each value of p.

        Parameters:
        - save_path (str, optional): Directory where plots should be saved. 
        If None, plots are just shown.
        """
        markers = ['o', 's', 'D', '^', 'v', 'p', '*']
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  
        
        unique_num_rot_values = sorted(self.df['num_rot'].unique()) 
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        for num_rot in unique_num_rot_values:
            plt.figure(figsize=(11, 10))

            df_rot_filtered = self.df[self.df['num_rot'] == num_rot] 
            unique_p_values = sorted(df_rot_filtered['p'].unique())  

            for idx, p in enumerate(unique_p_values):
                df_filtered = df_rot_filtered[df_rot_filtered['p'] == p]
                df_qubits_analysis = df_filtered.groupby('num_qubits')['min_shots'].agg(['mean', 'std']).reset_index()

                plt.errorbar(
                    df_qubits_analysis['num_qubits'], df_qubits_analysis['mean'], 
                    yerr=df_qubits_analysis['std'],
                    linestyle=line_styles[idx % len(line_styles)], 
                    marker=markers[idx % len(markers)], 
                    capsize=5, elinewidth=2, markeredgewidth=1, 
                    label=f"p={p}"
                )

            plt.xlabel("Number of Qubits", fontsize=24)
            plt.ylabel(f"Average Min Shots ± Std Dev for {num_rot} rotamers", fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.legend(title="p", fontsize=22, loc='upper left')
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"min_shots_vs_num_qubits_num_rot_{num_rot}.pdf"))

            plt.show()
            plt.close()


    def plot_min_shots_vs_num_qubits_per_p_best_fit(self, save_path=None):
        """
        Plots min_shots vs num_qubits with different curves for each value of p.

        Parameters:
        - save_path (str, optional): Directory where plots should be saved. If None, the plot is just shown.
        """
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        plt.figure(figsize=(11, 10))

        markers = ['o', 's', 'D', '^', 'v', 'p', '*']
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  
        
        unique_p_values = sorted(self.df['p'].unique()) 

        for idx, p in enumerate(unique_p_values):
            df_filtered = self.df[self.df['p'] == p]
            df_qubits_analysis = df_filtered.groupby('num_qubits')['min_shots'].agg(['mean', 'std']).reset_index()
            
            X = df_qubits_analysis["num_qubits"].values
            y = df_qubits_analysis["mean"].values

            num_qubits_smooth = np.linspace(X.min(), X.max(), 100)

            # def log_func(x, a, b, c):
            #     return a * np.log(b * x) + c

            # popt, _ = curve_fit(log_func, X, y)
            # best_fit_curve = log_func(num_qubits_smooth, *popt)

            plt.scatter(
                df_qubits_analysis['num_qubits'], df_qubits_analysis['mean'],
                marker=markers[idx % len(markers)], label=f"p={p}", s=100, alpha=0.8
            )

            if len(df_qubits_analysis) > 2:
                poly_coeffs = np.polyfit(df_qubits_analysis['num_qubits'], df_qubits_analysis['mean'], deg=2)
                poly_fit = np.poly1d(poly_coeffs)

                num_qubits_smooth = np.linspace(df_qubits_analysis['num_qubits'].min(), df_qubits_analysis['num_qubits'].max(), 100)
                best_fit_curve = poly_fit(num_qubits_smooth)

                plt.plot(num_qubits_smooth, best_fit_curve, linestyle=line_styles[idx % len(line_styles)], linewidth=2, alpha=0.7)

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Average Min Shots ± Std Dev", fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(title="p", fontsize=22)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "min_shots_vs_num_qubits_per_p_best_fit.pdf"))

        plt.show()
        plt.close() 
        

    def plot_fraction_vs_p(self, df_filtered, save_path=None):
        # Get unique alpha and p values
        alpha_values, shot_values = get_unique_alpha_and_shot_values(df_filtered)

        # Define colors, linestyles, and markers
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
        linestyles = ['-', '--', '-.', ':'] * len(shot_values)
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(shot_values)

        # Create a plot for each combination of alpha and p
        for i, alpha in enumerate(alpha_values):
            for j, shots in enumerate(shot_values):
                # Filter the dataframe based on alpha and p
                df_filtered_alpha_shots = df_filtered[(df_filtered['alpha'] == alpha) & (df_filtered['shots'] == shots)]
                
                # Sort the dataframe by the 'p' column
                df_filtered_alpha_shots = df_filtered_alpha_shots.sort_values(by='p')

                

                # Plot fraction vs number of shots
                plt.plot(df_filtered_alpha_shots['p'], df_filtered_alpha_shots['fraction'], label=f'alpha={alpha}, shots={shots}', color=colors[i], linestyle=linestyles[j], marker=markers[j])

        # Set plot title and labels
        plt.title('Fraction vs p')
        plt.xlabel('Number of Layers (p)')
        plt.ylabel('Fraction')

        # Add gridlines
        plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        # Set the background color of the plot to white
        plt.gca().set_facecolor('white')

        # Set the edge color of the plot to black
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['top'].set_color('black') 
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['left'].set_color('black')

        # Add legend
        plt.legend()



        # Save the plot
        if save_path:
            # Check if the directory exists
            if not os.path.exists(save_path):
                # If the directory doesn't exist, create it
                os.makedirs(save_path)

            plt.savefig(str(save_path) + f"/fraction_vs_p_num_res_{df_filtered['num_res'].iloc[0]}_num_rot_{df_filtered['num_rot'].iloc[0]}.pdf")
    

    def plot_min_shots(self, df, save_path=None):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']

        # Get unique pairs of alpha and p
        alpha_p_pairs = df[['alpha', 'p']].drop_duplicates().values

        # For each alpha, p pair
        for alpha, p in alpha_p_pairs:
            # Filter the dataframe for the current alpha, p pair
            df_filtered = df[(df['alpha'] == alpha) & (df['p'] == p)]
            
            # Get unique num_rot values for the current alpha, p pair
            num_rot_values = df_filtered['num_rot'].unique()
            
            # Create a new figure
            plt.figure()

            # define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_rot_values)
            linestyles = ['-', '--', '-.', ':'] * len(num_rot_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_rot_values)
            
            # For each num_rot value
            for num_rot in num_rot_values:
                # Filter the dataframe for the current num_rot value
                df_filtered_num_rot = df_filtered[df_filtered['num_rot'] == num_rot]

                # sort the dataframe by the 'N' column
                df_filtered_num_rot = df_filtered_num_rot.sort_values(by='N')
                
                # Plot N vs min_shots
                plt.plot(df_filtered_num_rot['N'], np.log10(df_filtered_num_rot['min_shots']), label=f'num_rot={num_rot}', color=colors[num_rot], linestyle=linestyles[num_rot], marker=markers[num_rot])
            
            # Set title, labels and legend
            plt.title(f'alpha={alpha}, p={p}')
            plt.xlabel('N')
            plt.ylabel('log10(min_shots)')
            plt.legend()
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(str(save_path) + f"/min_shots_vs_N_alpha_{alpha}_p_{p}.pdf")


    def plot_ground_state_probability_vs_N(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
            # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
            df['N'] = df['num_res'] * df['num_rot']

            # Get unique pairs of alpha and p
            alpha_p_shots_pairs = df[['alpha', 'p', 'shots']].drop_duplicates().values

            # For each alpha, p pair
            for alpha, p, shots in alpha_p_shots_pairs:
                # Filter the dataframe for the current alpha, p pair
                df_filtered = df[(df['alpha'] == alpha) & (df['p'] == p ) & (df['shots'] == shots)]
                
                # Get unique num_rot values for the current alpha, p pair
                num_rot_values = df_filtered['num_rot'].unique()
                
                # Create a new figure
                plt.figure()

                # define colors, linestyles, and markers
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_rot_values)
                linestyles = ['-', '--', '-.', ':'] * len(num_rot_values)
                markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_rot_values)
                
                def random_sampling_restricted(num_res, num_rot):
                    return 1/(pow(num_rot, num_res))
                # For each num_rot value
                for num_rot in num_rot_values:
                    # Filter the dataframe for the current num_rot value
                    df_filtered_num_rot = df_filtered[df_filtered['num_rot'] == num_rot]

                    # sort the dataframe by the 'N' column
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by='N')
                    
                    random_sampling_unrestricted = 1/2**df_filtered_num_rot['N']
                    # debug 
                    
                    if num_rot == 2 and alpha == 1.0 and p == 3:
                        print(f'num_rot={num_rot}, alpha={alpha}, p={p}, shots={shots}')
                        print(df_filtered_num_rot)
                        # print(f'ground state probability: {df_filtered_num_rot["ground_state_probability"]}')
                        
                    # Plot N vs min_shots
                    plt.plot(df_filtered_num_rot['N'], np.log10(df_filtered_num_rot['ground_state_probability']), label=f'num_rot={num_rot}', color=colors[num_rot], linestyle=linestyles[num_rot], marker=markers[num_rot])
                    plt.plot(df_filtered_num_rot['N'], np.log10(random_sampling_unrestricted), label=f'random_sampling_unrestricted_num_rot_{num_rot}', color='black', linestyle=linestyles[num_rot], marker=markers[num_rot])
                    plt.plot(df_filtered_num_rot['N'], np.log10(random_sampling_restricted(df_filtered_num_rot['num_res'], df_filtered_num_rot['num_rot'])), label=f'random_sampling_restricted_num_rot_={num_rot}', color='green', linestyle=linestyles[num_rot], marker=markers[num_rot])
                # Set title, labels and legend
                plt.title(f'alpha={alpha}, p={p}, shots={shots}')
                plt.xlabel('N')
                plt.ylabel('lg10(Probability)')
                plt.legend()
                # Add gridlines
                plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

                # Set the background color of the plot to white
                plt.gca().set_facecolor('white')

                # Set the edge color of the plot to black
                plt.gca().spines['bottom'].set_color('black')
                plt.gca().spines['top'].set_color('black') 
                plt.gca().spines['right'].set_color('black')
                plt.gca().spines['left'].set_color('black')
                
                # Save the plot
                if save_path:
                    # Check if the directory exists
                    if not os.path.exists(save_path):
                        # If the directory doesn't exist, create it
                        os.makedirs(save_path)

                    plt.savefig(str(save_path) + f"/ground_state_probability_vs_N_alpha_{alpha}_p_{p}_shots_{shots}.pdf")

                plt.close()

    def plot_tail_probability_vs_N(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
            # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
            df['N'] = df['num_res'] * df['num_rot']

            tail = df['tail'].iloc[0]

            # Get unique pairs of alpha and p
            alpha_p_shots_pairs = df[['alpha', 'p', 'shots']].drop_duplicates().values

            # For each alpha, p pair
            for alpha, p, shots in alpha_p_shots_pairs:
                # Filter the dataframe for the current alpha, p pair
                df_filtered = df[(df['alpha'] == alpha) & (df['p'] == p ) & (df['shots'] == shots)]
                
                # Get unique num_rot values for the current alpha, p pair
                num_rot_values = df_filtered['num_rot'].unique()
                
                # Create a new figure
                plt.figure()

                # define colors, linestyles, and markers
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_rot_values)
                linestyles = ['-', '--', '-.', ':'] * len(num_rot_values)
                markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_rot_values)
                
                # def random_sampling_restricted(num_res, num_rot):
                #     return 1/(pow(num_rot, num_res))
                # For each num_rot value
                for num_rot in num_rot_values:
                    # Filter the dataframe for the current num_rot value
                    df_filtered_num_rot = df_filtered[df_filtered['num_rot'] == num_rot]

                    # sort the dataframe by the 'N' column
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by='N')
                    
                    #random_sampling_unrestricted = 1/2**df_filtered_num_rot['N']
                    # Plot N vs min_shots
                    plt.plot(df_filtered_num_rot['N'], np.log10(df_filtered_num_rot['tail_probability']), label=f'num_rot={num_rot}', color=colors[num_rot], linestyle=linestyles[num_rot], marker=markers[num_rot])
                    #plt.plot(df_filtered_num_rot['N'], np.log10(random_sampling_unrestricted), label=f'random_sampling_unrestricted_num_rot_{num_rot}', color='black', linestyle=linestyles[num_rot], marker=markers[num_rot])
                    #plt.plot(df_filtered_num_rot['N'], np.log10(random_sampling_restricted(df_filtered_num_rot['num_res'], df_filtered_num_rot['num_rot'])), label=f'random_sampling_restricted_num_rot_={num_rot}', color='green', linestyle=linestyles[num_rot], marker=markers[num_rot])
                # Set title, labels and legend
                plt.title(f'tail_{tail}_alpha={alpha}, p={p}, shots={shots}')
                plt.xlabel('N')
                plt.ylabel('lg10(Probability)')
                plt.legend()
                # Add gridlines
                plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

                # Set the background color of the plot to white
                plt.gca().set_facecolor('white')

                # Set the edge color of the plot to black
                plt.gca().spines['bottom'].set_color('black')
                plt.gca().spines['top'].set_color('black') 
                plt.gca().spines['right'].set_color('black')
                plt.gca().spines['left'].set_color('black')
                
                # Save the plot
                if save_path:
                    # Check if the directory exists
                    if not os.path.exists(save_path):
                        # If the directory doesn't exist, create it
                        os.makedirs(save_path)

                    plt.savefig(str(save_path) + f"/tail_{tail}_probability_vs_N_alpha_{alpha}_p_{p}_shots_{shots}.pdf")

                plt.close()


    def plot_ground_state_probability_vs_p(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        

        # Get unique pairs of alpha and p
        nres_nrot_alpha_shots_pairs = df[['num_res', 'num_rot', 'alpha', 'shots']].drop_duplicates().values

        # For each alpha, p pair
        for num_res, num_rot, alpha, shots in nres_nrot_alpha_shots_pairs:
            # Filter the dataframe for the current alpha, p pair
            df_filtered = df[(df['alpha'] == alpha) & (df['shots'] == shots) & (df['num_res'] == num_res) & (df['num_rot'] == num_rot)]
            
            # Get unique num_rot values for the current alpha, p pair
            p_values = df_filtered['p'].unique()
            
            # Create a new figure
            plt.figure()

            # define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(p_values)
            linestyles = ['-', '--', '-.', ':'] * len(p_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(p_values)
            p_vals = df_filtered['p']
            prob_values = df_filtered['ground_state_probability']
            # sort p_vals and prob_values by p values
            p_vals, prob_values = zip(*sorted(zip(p_vals, prob_values)))
            # plot p vs ground_state_probability
            plt.plot(p_vals, prob_values)
            # For each num_rot value
            # for p in p_values:
            #     # Filter the dataframe for the current num_rot value
            #     df_filtered_p = df_filtered[df_filtered['p'] == p]

            #     # sort the dataframe by the 'N' column
            #     df_filtered_p = df_filtered_p.sort_values(by='p')
            #     # Plot N vs min_shots
            #     plt.plot(df_filtered_p['p'], np.log10(df_filtered_p['ground_state_probability']), label=f'p={p}', color=colors[p], linestyle=linestyles[p], marker=markers[p])
            
            # Set title, labels and legend
            plt.title(f'num_res={num_res}, num_rot={num_rot}, alpha={alpha}, shots={shots}')
            plt.xlabel('p')
            plt.ylabel('Probability')
            #plt.legend()
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(str(save_path) + f"/ground_state_probability_vs_p_num_res_{num_res}_num_rot_{num_rot}_alpha_{alpha}_shots_{shots}.pdf")

            plt.close()


    def plot_ground_state_probability_vs_p_pooled_alpha(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']

        # Get unique pairs of num_res, num_rot, and shots
        nres_nrot_shots_pairs = df[['num_res', 'num_rot', 'shots']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_res, num_rot, shots in nres_nrot_shots_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_res'] == num_res) & (df['num_rot'] == num_rot) & (df['shots'] == shots)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            alpha_values = df_filtered['alpha'].unique()
            alpha_values = sorted(df_filtered['alpha'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
            linestyles = ['-', '--', '-.', ':'] * len(alpha_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(alpha_values)

            # For each alpha value
            for i, alpha in enumerate(alpha_values):
                # Filter the dataframe for the current alpha value
                df_alpha = df_filtered[df_filtered['alpha'] == alpha]

                # Get p values and ground state probabilities
                p_vals = df_alpha['p']
                prob_values = df_alpha['ground_state_probability']

                # Sort p_vals and prob_values by p values
                p_vals, prob_values = zip(*sorted(zip(p_vals, prob_values)))

                # Plot p vs ground_state_probability
                plt.plot(p_vals, prob_values, label=f'alpha={alpha}', color=colors[i], linestyle=linestyles[i], marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'num_res={num_res}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('p')
            plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"ground_state_probability_vs_p_num_res_{num_res}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_tail_probability_vs_alpha_pooled_p(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']

        tail = df['tail'].iloc[0]

        # Get unique pairs of num_res, num_rot, and shots
        nres_nrot_shots_pairs = df[['num_res', 'num_rot', 'shots']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_res, num_rot, shots in nres_nrot_shots_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_res'] == num_res) & (df['num_rot'] == num_rot) & (df['shots'] == shots)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            p_values = df_filtered['p'].unique()
            p_values = sorted(df_filtered['p'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(p_values)
            linestyles = ['-', '--', '-.', ':'] * len(p_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(p_values)

            # For each alpha value
            for i, p in enumerate(p_values):
                # Filter the dataframe for the current alpha value
                df_p = df_filtered[df_filtered['p'] == p]

                # Get p values and ground state probabilities
                alpha_vals = df_p['alpha']
                prob_values = df_p['tail_probability']

                # Sort p_vals and prob_values by p values
                alpha_vals, prob_values = zip(*sorted(zip(alpha_vals, prob_values)))

                # Plot p vs ground_state_probability
                plt.plot(alpha_vals, prob_values, label=f'p={p}', color=colors[i], linestyle='-', marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'tail={tail}, num_res={num_res}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('alpha')
            plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"tail_{tail}_probability_vs_alpha_num_res_{num_res}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_tail_probability_vs_alpha_pooled_num_res(self, df, xaxis='log', save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        

        tail = df['tail'].iloc[0]

        # Get unique pairs of num_res, num_rot, and shots
        nrot_shots_p_pairs = df[['num_rot', 'shots','p']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_rot, shots, p in nrot_shots_p_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_rot'] == num_rot) & (df['shots'] == shots) & (df['p'] == p)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            num_res_values = df_filtered['num_res'].unique()
            num_res_values = sorted(df_filtered['num_res'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_res_values)
            linestyles = ['-', '--', '-.', ':'] * len(num_res_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_res_values)

            # For each alpha value
            for i, num_res in enumerate(num_res_values):
                # Filter the dataframe for the current alpha value
                df_res = df_filtered[df_filtered['num_res'] == num_res]

                # Get p values and ground state probabilities
                alpha_vals = df_res['alpha']
                prob_values = df_res['tail_probability']

                # Sort p_vals and prob_values by p values
                alpha_vals, prob_values = zip(*sorted(zip(alpha_vals, prob_values)))

                # Plot p vs ground_state_probability
                if xaxis == 'log':
                    plt.plot(alpha_vals, np.log10(prob_values), label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
                if xaxis == 'linear':
                    plt.plot(alpha_vals, prob_values, label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'tail={tail}, p={p}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('alpha')
            if xaxis == 'log':
                plt.ylabel('lg10(Probability)')
            if xaxis == 'linear':
                plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"tail_{tail}_{xaxis}_probability_vs_alpha_p_{p}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_ground_state_probability_vs_alpha_pooled_num_res(self, df, xaxis='log', save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        

        

        # Get unique pairs of num_res, num_rot, and shots
        nrot_shots_p_pairs = df[['num_rot', 'shots','p']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_rot, shots, p in nrot_shots_p_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_rot'] == num_rot) & (df['shots'] == shots) & (df['p'] == p)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            num_res_values = df_filtered['num_res'].unique()
            num_res_values = sorted(df_filtered['num_res'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_res_values)
            linestyles = ['-', '--', '-.', ':'] * len(num_res_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_res_values)

            # For each alpha value
            for i, num_res in enumerate(num_res_values):
                # Filter the dataframe for the current alpha value
                df_res = df_filtered[df_filtered['num_res'] == num_res]

                # Get p values and ground state probabilities
                alpha_vals = df_res['alpha']
                prob_values = df_res['ground_state_probability']

                # Sort p_vals and prob_values by p values
                alpha_vals, prob_values = zip(*sorted(zip(alpha_vals, prob_values)))

                # Plot p vs ground_state_probability
                if xaxis == 'log':
                    plt.plot(alpha_vals, np.log10(prob_values), label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
                if xaxis == 'linear':
                    plt.plot(alpha_vals, prob_values, label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'p={p}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('alpha')
            if xaxis == 'log':
                plt.ylabel('lg10(Probability)')
            if xaxis == 'linear':
                plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"ground_state_{xaxis}_probability_vs_alpha_p_{p}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_cvar_aggregate_vs_alpha_pooled_num_res(self, df, xaxis = 'linear', save_path=None, pos_bool=False, transverse_field_bool=False, approximation_ratio=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        

        

        # Get unique pairs of num_res, num_rot, and shots
        nrot_shots_p_pairs = df[['num_rot', 'shots','p']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_rot, shots, p in nrot_shots_p_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_rot'] == num_rot) & (df['shots'] == shots) & (df['p'] == p)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            num_res_values = df_filtered['num_res'].unique()
            num_res_values = sorted(df_filtered['num_res'].unique())

            


            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(num_res_values)
            linestyles = ['-', '--', '-.', ':'] * len(num_res_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(num_res_values)

            # For each alpha value
            for i, num_res in enumerate(num_res_values):
                

                # Filter the dataframe for the current alpha value
                df_res = df_filtered[df_filtered['num_res'] == num_res]

                if approximation_ratio:
                    energy = df_res['final_energies']

                    import ast

                    energy = [ast.literal_eval(energy_val) for energy_val in energy]

                # Get p values and ground state probabilities
                alpha_vals = df_res['alpha']
                cvar_values = df_res['cvar_aggregate']

                # Sort p_vals and prob_values by p values
                if approximation_ratio:
                    alpha_vals, cvar_values, energy = zip(*sorted(zip(alpha_vals, cvar_values, energy)))
                else:
                    alpha_vals, cvar_values = zip(*sorted(zip(alpha_vals, cvar_values)))

                if approximation_ratio:
                    min_energy = []
                    for energy_arr in energy:
                        min_energy.append(min(energy_arr))
                    if approximation_ratio:
                        cvar_values = np.array(cvar_values) / np.array(min_energy)

                if xaxis == 'log':
                    plt.plot(alpha_vals, np.log10(cvar_values), label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
                if xaxis == 'linear':
                    plt.plot(alpha_vals, cvar_values, label=f'num_res={num_res}', color=colors[i], linestyle='-', marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'p={p}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('alpha')
 
            if approximation_ratio and xaxis == 'linear':
                plt.ylabel('Approximation ratio')
            elif approximation_ratio and xaxis == 'log':
                plt.ylabel('lg10(Approximation ratio)')
            elif xaxis == 'linear':
                plt.ylabel('cvar_values')
            elif xaxis == 'log':
                plt.ylabel('lg10(cvar_values)')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)
                if approximation_ratio:
                    plt.savefig(os.path.join(save_path, f"{xaxis}_approximation_ratio_vs_alpha_p_{p}_num_rot_{num_rot}_shots_{shots}.pdf"))
                else:
                    plt.savefig(os.path.join(save_path, f"{xaxis}_cvar_vs_alpha_p_{p}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_ground_state_probability_vs_N_pooled_alpha(self, df, xaxis = 'log', save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']
        

        # Get unique pairs of num_res, num_rot, and shots
        p_shots_pairs = df[['p', 'shots']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for p, shots in p_shots_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['p'] == p)  & (df['shots'] == shots)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            alpha_values = df_filtered['alpha'].unique()
            alpha_values = sorted(df_filtered['alpha'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
            linestyles = ['-', '--', '-.', ':'] * len(alpha_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(alpha_values)

            # For each alpha value
            for i, alpha in enumerate(alpha_values):
                # Filter the dataframe for the current alpha value
                df_alpha = df_filtered[df_filtered['alpha'] == alpha]
                

                # Get unique num_rot values for the current alpha, p pair
                num_rot_values = df_alpha['num_rot'].unique()

                # For each num_rot value
                for num_rot in num_rot_values:
                    # Filter the dataframe for the current num_rot value
                    df_filtered_num_rot = df_alpha[df_alpha['num_rot'] == num_rot]

                    # sort the dataframe by the 'N' column
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by='N')

                    # sort by num_res and num_rot
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by=['num_res', 'num_rot'])
                    
                    #random_sampling_unrestricted = 1/2**df_filtered_num_rot['N']
                    # Plot N vs min_shots
                    if xaxis == 'log':
                        plt.plot(df_filtered_num_rot['N'], np.log10(df_filtered_num_rot['ground_state_probability']), label=f'alpha={alpha}_num_rot_{num_rot}', color=colors[i], linestyle=linestyles[num_rot], marker=markers[i])
                    if xaxis == 'linear':
                        plt.plot(df_filtered_num_rot['N'], (df_filtered_num_rot['ground_state_probability']), label=f'alpha={alpha}_num_rot_{num_rot}', color=colors[i], linestyle=linestyles[num_rot], marker=markers[i])
                
                    

                # # Get p values and ground state probabilities
                # p_vals = df_alpha['p']
                # prob_values = df_alpha['ground_state_probability']

                # # Sort p_vals and prob_values by p values
                # p_vals, prob_values = zip(*sorted(zip(p_vals, prob_values)))

                # # Plot p vs ground_state_probability
                # plt.plot(p_vals, prob_values, label=f'alpha={alpha}', color=colors[i], linestyle=linestyles[i], marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'p={p},  shots={shots}')
            plt.xlabel('N')
            if xaxis == 'log':
                plt.ylabel('lg10(Probability)')
            if xaxis == 'linear':
                plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"ground_state_{xaxis}_probability_vs_N_p_{p}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()

    def plot_tail_probability_vs_N_pooled_alpha(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']
        tail = df['tail'].iloc[0]

        # Get unique pairs of num_res, num_rot, and shots
        p_shots_pairs = df[['p', 'shots']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for p, shots in p_shots_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['p'] == p)  & (df['shots'] == shots)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            alpha_values = df_filtered['alpha'].unique()
            alpha_values = sorted(df_filtered['alpha'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
            linestyles = ['-', '--', '-.', ':'] * len(alpha_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(alpha_values)

            # For each alpha value
            for i, alpha in enumerate(alpha_values):
                # Filter the dataframe for the current alpha value
                df_alpha = df_filtered[df_filtered['alpha'] == alpha]
                

                # Get unique num_rot values for the current alpha, p pair
                num_rot_values = df_alpha['num_rot'].unique()

                # For each num_rot value
                for num_rot in num_rot_values:
                    # Filter the dataframe for the current num_rot value
                    df_filtered_num_rot = df_alpha[df_alpha['num_rot'] == num_rot]

                    # sort the dataframe by the 'N' column
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by='N')

                    # sort by num_res and num_rot
                    df_filtered_num_rot = df_filtered_num_rot.sort_values(by=['num_res', 'num_rot'])
                    
                    #random_sampling_unrestricted = 1/2**df_filtered_num_rot['N']
                    # Plot N vs min_shots
                    plt.plot(df_filtered_num_rot['N'], (df_filtered_num_rot['tail_probability']), label=f'alpha={alpha}_num_rot_{num_rot}', color=colors[i], linestyle=linestyles[num_rot], marker=markers[i])
                    

                # # Get p values and ground state probabilities
                # p_vals = df_alpha['p']
                # prob_values = df_alpha['ground_state_probability']

                # # Sort p_vals and prob_values by p values
                # p_vals, prob_values = zip(*sorted(zip(p_vals, prob_values)))

                # # Plot p vs ground_state_probability
                # plt.plot(p_vals, prob_values, label=f'alpha={alpha}', color=colors[i], linestyle=linestyles[i], marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'tail={tail}, p={p},  shots={shots}')
            plt.xlabel('N')
            plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"tail_{tail}_probability_vs_N_p_{p}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()
            

    def plot_tail_probability_vs_p_pooled_alpha(self, df, save_path=None, pos_bool=False, transverse_field_bool=False):
        # Create a new column 'N' which is the product of 'num_res' and 'num_rot'
        df['N'] = df['num_res'] * df['num_rot']

        # Get unique pairs of num_res, num_rot, and shots
        nres_nrot_shots_pairs = df[['num_res', 'num_rot', 'shots']].drop_duplicates().values

        # For each num_res, num_rot, shots pair
        for num_res, num_rot, shots in nres_nrot_shots_pairs:
            # Filter the dataframe for the current num_res, num_rot, shots pair
            df_filtered = df[(df['num_res'] == num_res) & (df['num_rot'] == num_rot) & (df['shots'] == shots)]
            
            # Get unique alpha values for the current num_res, num_rot, shots pair
            alpha_values = df_filtered['alpha'].unique()
            alpha_values = sorted(df_filtered['alpha'].unique())

            
            # Create a new figure
            plt.figure()

            # Define colors, linestyles, and markers
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(alpha_values)
            linestyles = ['-', '--', '-.', ':'] * len(alpha_values)
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * len(alpha_values)

            # For each alpha value
            for i, alpha in enumerate(alpha_values):
                # Filter the dataframe for the current alpha value
                df_alpha = df_filtered[df_filtered['alpha'] == alpha]

                # Get p values and ground state probabilities
                p_vals = df_alpha['p']
                prob_values = df_alpha['tail_probability']

                # Sort p_vals and prob_values by p values
                p_vals, prob_values = zip(*sorted(zip(p_vals, prob_values)))

                # Plot p vs ground_state_probability
                plt.plot(p_vals, prob_values, label=f'alpha={alpha}', color=colors[i], linestyle=linestyles[i], marker=markers[i])
            
            # Set title, labels, and legend
            plt.title(f'num_res={num_res}, num_rot={num_rot}, shots={shots}')
            plt.xlabel('p')
            plt.ylabel('Probability')
            plt.legend()
            
            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            
            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                plt.savefig(os.path.join(save_path, f"tail_probability_vs_p_num_res_{num_res}_num_rot_{num_rot}_shots_{shots}.pdf"))
            
            # Close the plot to free memory
            plt.close()






    def plot_init_and_final_probabilities(self, df, df_exact=None, save_path=None, display=False, transverse_field_bool=False, pos_bool=False):
        # Group the DataFrame by the combination of 'num_res', 'num_rot', 'alpha', 'shots', 'p'
        if transverse_field_bool and pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field'])
        elif pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos'])
        else:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p'])

        
        # For each group
        for name, group in grouped:

            # Initialize the figure and the two axes
            fig, ax = plt.subplots(figsize=(10, 5))
            ax2 = ax.twinx()

            # For each row in the group
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

                # init_energies = ast.literal_eval(row['init_energies'])
                # init_dist = ast.literal_eval(row['init_dist'])

                # # Convert the 'final_energies' and 'final_dist' columns from strings to lists/dictionaries
                # final_energies = ast.literal_eval(row['final_energies'])
                # final_dist = ast.literal_eval(row['final_dist'])

                # Filter out None values from the energies and their corresponding probabilities
                init_energies_list, init_dist_list = zip(*[(e, p) for e, p in zip(init_energies, list(init_dist.values())) if e is not None])
                final_energies_list, final_dist_list = zip(*[(e, p) for e, p in zip(final_energies, list(final_dist.values())) if e is not None])

                # Create the first histogram with the energies from 'init_energies' on the X-axis and the corresponding probabilities from 'init_dist' on the Y-axis
                ax.hist(init_energies_list, weights=list(init_dist_list), bins=1000, alpha=0.5, label='Initial Energies')

                # Create the second histogram with the energies from 'final_energies' on the X-axis and the corresponding probabilities from 'final_dist' on the Y-axis
                ax2.hist(final_energies_list, weights=list(final_dist_list), bins=1000, alpha=0.5, label='Final Energies', color='red')

            # After creating the histogram, find the key that corresponds to the maximum probability
            max_prob_key_init = max(init_dist, key=init_dist.get)
            max_prob_key_final = max(final_dist, key=final_dist.get)
            max_prob_key_init = int(max_prob_key_init)
            max_prob_key_final = int(max_prob_key_final)
            min_en_key_init = list(init_dist_list)[0]
            min_en_key_final = list(final_dist_list)[0]

            max_prob_key_init = int_to_bitstring(max_prob_key_init, name[0] * name[1])
            max_prob_key_final = int_to_bitstring(max_prob_key_final, name[0] * name[1])

            if df_exact is not None:
                from proteinfolding.data_processing import find_min_energy_and_bitstring_from_exact_energy_dataframe
                # filter num_res and num_rot values
                num_res = name[0]
                num_rot = name[1]
                
                min_energy, _ = find_min_energy_and_bitstring_from_exact_energy_dataframe(df_exact, num_res, num_rot)
                # add min_energy to the plot
                ax.axvline(x=min_energy, color='black', linestyle='--', label='Min Energy')

            # Print the key on the plot
           # ax.text(0.6, 0.8, f'Max Prob Key (Init): {max_prob_key_init}', transform=ax.transAxes)
            #ax2.text(0.6, 0.6, f'Max Prob Key (Final): {max_prob_key_final}', transform=ax2.transAxes)

            # ax.text(0.4, 0.4, f'Min Energy Key (Init): {min_en_key_init} with energy {init_energies_list[0]}', transform=ax.transAxes)
            # ax2.text(0.4, 0.2, f'Min Energy Key (Final): {min_en_key_final} with energy {final_energies_list[0]}', transform=ax2.transAxes)

            # Set the title, labels, and legend
            if transverse_field_bool and pos_bool:
                ax.set_title(f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}')
            elif pos_bool:
                ax.set_title(f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}')
            else:
                ax.set_title(f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}')
            ax.set_xlabel('Energy')
            ax.set_ylabel('Initial Probability')
            ax2.set_ylabel('Final Probability')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # Get the limits of the first y axis
            y_lim1 = ax.get_ylim()

            # Get the limits of the second y axis
            y_lim2 = ax2.get_ylim()

            # If the range of the first y axis is larger
            if (y_lim1[1] - y_lim1[0]) > (y_lim2[1] - y_lim2[0]):
                # Set the limits of the second y axis to match the first
                ax2.set_ylim(y_lim1)
            else:
                # Set the limits of the first y axis to match the second
                ax.set_ylim(y_lim2)


            # Set the background color of the plot to white
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')

            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                if transverse_field_bool and pos_bool:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}.pdf")
                elif pos_bool:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}.pdf")
                else:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}.pdf")

            # close fig
            if display:
                plt.show()
            plt.close(fig)

    import os
    import ast
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_parameter_evolution(self, df, save_path=None, display=False, transverse_field_bool=False, pos_bool=False):
        # Group the DataFrame by the combination of 'num_res', 'num_rot', 'alpha', 'shots', 'p'
        if transverse_field_bool and pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field'])
        elif pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos'])
        else:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p'])

        # For each group
        for name, group in grouped:
            # Initialize the figure and the two axes for cost and mixer parameters
            fig, (ax_cost, ax_mixer) = plt.subplots(1, 2, figsize=(25, 10))

            # Adjust the space between the subplots
            fig.subplots_adjust(hspace=0.5)

            # For each row in the group
            for i, row in group.iterrows():
                # Convert the 'parameters' column from strings to lists/dictionaries
                params = row['parameters']
                if isinstance(params, str):
                    params = ast.literal_eval(params)

                # Filter out None values from the energies and their corresponding probabilities
                p = int(row['p'])
                iterations = len(params)

                cost_params = np.zeros((iterations, p), dtype=object)
                mixer_params = np.zeros((iterations, p), dtype=object)

                for current_iter in range(iterations):
                    current_params = np.array(params[current_iter])
                    for rep in range(p):
                        cost_params[current_iter, rep] = current_params[2 * rep]
                        mixer_params[current_iter, rep] = current_params[2 * rep + 1]

                # Plot cost_params
                for rep in range(p):
                    ax_cost.plot(cost_params[:, rep], label=f'cost_params_{rep}')

                # Plot mixer_params
                for rep in range(p):
                    ax_mixer.plot(mixer_params[:, rep], label=f'mixer_params_{rep}')

            # Set the title for cost and mixer parameters plots
            if transverse_field_bool and pos_bool:
                title = f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}'
            elif pos_bool:
                title = f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}'
            else:
                title = f'res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}'

            ax_cost.set_title(f'Cost Parameter Evolution\n{title}')
            ax_mixer.set_title(f'Mixer Parameter Evolution\n{title}')

            # Set the labels and legend for cost parameters plot
            ax_cost.set_xlabel('Iteration')
            ax_cost.set_ylabel('Parameter Value')
            ax_cost.legend(loc='upper left')

            # Set the labels and legend for mixer parameters plot
            ax_mixer.set_xlabel('Iteration')
            ax_mixer.set_ylabel('Parameter Value')
            ax_mixer.legend(loc='upper left')

            # Set the background color of the plots to white
            for ax in [ax_cost, ax_mixer]:
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                pdf_path = os.path.join(save_path, f"{title}.pdf")
                fig.savefig(pdf_path)

            # Display the plot if required
            if display:
                plt.show()
            plt.close(fig)



    import os
    import ast
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_parameter_evolution2(self, df, save_path=None, display=False, transverse_field_bool=False, pos_bool=False):
        # Group the DataFrame by the combination of 'num_res', 'num_rot', 'alpha', 'shots', 'p'
        if transverse_field_bool and pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field'])
        elif pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos'])
        else:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p'])

        # For each group
        for name, group in grouped:
            # Initialize the figures and axes for cost and mixer parameters
            fig_cost, ax_cost = plt.subplots(figsize=(10, 5))
            fig_mixer, ax_mixer = plt.subplots(figsize=(10, 5))

            # For each row in the group
            for i, row in group.iterrows():
                # Convert the 'parameters' column from strings to lists/dictionaries
                params = row['parameters']
                if isinstance(params, str):
                    params = ast.literal_eval(params)

                # Filter out None values from the energies and their corresponding probabilities
                p = int(row['p'])
                iterations = len(params)

                cost_params = np.zeros((iterations, p), dtype=object)
                mixer_params = np.zeros((iterations, p), dtype=object)

                for current_iter in range(iterations):
                    current_params = params[current_iter]

                    # Ensure it's a flat list
                    if isinstance(current_params, list) and len(current_params) == 1 and isinstance(current_params[0], list):
                        # Unwrap one more layer if it's wrapped like [[...]]
                        current_params = current_params[0]

                    current_params = np.array(current_params).flatten()

                    if len(current_params) != 2 * p:
                        print(f"Warning: Parameter length mismatch at iteration {current_iter}. Expected {2 * p}, got {len(current_params)}")
                        continue  # Skip this iteration if something's wrong

                    for rep in range(p):
                        cost_params[current_iter, rep] = current_params[2 * rep]
                        mixer_params[current_iter, rep] = current_params[2 * rep + 1]


                # Plot cost_params
                for rep in range(p):
                    ax_cost.plot(cost_params[:, rep], label=f'cost_params_{rep}')

                # Plot mixer_params
                for rep in range(p):
                    ax_mixer.plot(mixer_params[:, rep], label=f'mixer_params_{rep}')

            # Set the title for cost parameters plot
            if transverse_field_bool and pos_bool:
                ax_cost.set_title(f'cost_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}')
                ax_mixer.set_title(f'mixer_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}')
            elif pos_bool:
                ax_cost.set_title(f'cost_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}')
                ax_mixer.set_title(f'mixer_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}')
            else:
                ax_cost.set_title(f'cost_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}')
                ax_mixer.set_title(f'mixer_params_res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}')

            # Set the labels and legend for cost parameters plot
            ax_cost.set_xlabel('Iteration')
            ax_cost.set_ylabel('Parameter Value')
            ax_cost.legend(loc='upper left')

            # Set the labels and legend for mixer parameters plot
            ax_mixer.set_xlabel('Iteration')
            ax_mixer.set_ylabel('Parameter Value')
            ax_mixer.legend(loc='upper left')

            # Set the background color of the plots to white
            for ax in [ax_cost, ax_mixer]:
                ax.set_facecolor('white')
                ax.spines['bottom'].set_color('black')
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.spines['left'].set_color('black')
                ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Save the plots
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                if transverse_field_bool and pos_bool:
                    cost_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}_cost.pdf")
                    mixer_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}_mixer.pdf")
                elif pos_bool:
                    cost_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_cost.pdf")
                    mixer_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_mixer.pdf")
                else:
                    cost_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_cost.pdf")
                    mixer_path = os.path.join(save_path, f"res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_mixer.pdf")

                fig_cost.savefig(cost_path)
                fig_mixer.savefig(mixer_path)

            # Display the plots if required
            if display:
                plt.show()
            plt.close(fig_cost)
            plt.close(fig_mixer)





    def plot_parameter_evolution2(self, df, save_path=None, display=False, transverse_field_bool=False, pos_bool=False):
        # Group the DataFrame by the combination of 'num_res', 'num_rot', 'alpha', 'shots', 'p'
        if transverse_field_bool and pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos', 'transverse_field'])
        elif pos_bool:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p', 'pos'])
        else:
            grouped = df.groupby(['num_res', 'num_rot', 'alpha', 'shots', 'p'])

        for name, group in grouped:

            # Initialize the figure and the two axes
            fig, ax = plt.subplots(figsize=(10, 5))
            

            # For each row in the group
             
            for i, row in group.iterrows():
                # Convert the 'init_energies' and 'init_dist' columns from strings to lists/dictionaries
                params = row['parameters']
                
                if isinstance(params, str):
                    params = ast.literal_eval(params)

                # Filter out None values from the energies and their corresponding probabilities
                p = int(row['p'])
                iterations = len(params)
                

                cost_params = np.zeros(iterations, dtype=object)
                mixer_params = np.zeros(iterations, dtype=object)

                # assert len(params) == 2*p
                
                for j in range(iterations):
                    current_params = np.array(params[j])
                    print("current_params: ", current_params)
                    cost_params[j] = current_params[::2]
                    mixer_params[j] = current_params[1::2]
                    

                print("cost_params: ", cost_params)
                print("mixer_params: ", mixer_params)
                

                

        


        # plot cost_params
        # for i in range(cost_param_arr.shape[1]):
        #     ax.plot(cost_param_arr[:, i], label=f'cost_param_{i}')

        # # plot mixer_params
        # for i in range(mixer_param_arr.shape[1]):
        #     ax.plot(mixer_param_arr[:, i], label=f'mixer_param_{i}')
            print("cost para")
            for i in range(p):
                print("cost_params[:,i]: ", cost_params[:,i])
                print("mixer_params[:,i]: ", mixer_params[:,i])
                # ax.plot(cost_params[:,i], label=f'cost_params_{i}')
                # ax.plot(mixer_params[:,i], label=f'mixer_params_{i}')

            return None

            # Set the title, labels, and legend
            ax.set_title(f'Cost and Mixer Parameter Evolution')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')

            ax.legend(loc='upper left')

            # Set the background color of the plot to white
            
            plt.gca().set_facecolor('white')

            # Set the edge color of the plot to black
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['top'].set_color('black') 
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['left'].set_color('black')

            # Add gridlines
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Save the plot
            if save_path:
                # Check if the directory exists
                if not os.path.exists(save_path):
                    # If the directory doesn't exist, create it
                    os.makedirs(save_path)

                if transverse_field_bool and pos_bool:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}_transverse_field_{name[6]}.pdf")
                elif pos_bool:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}_pos_{name[5]}.pdf")
                else:
                    plt.savefig(str(save_path) + f"/res_{name[0]}_rot_{name[1]}_alpha_{name[2]}_shots_{name[3]}_p_{name[4]}.pdf")

            # close fig
            if display:
                plt.show()
            plt.close(fig)

                


    
