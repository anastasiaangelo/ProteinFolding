import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class DataPlotter:
    def __init__(self, df_cpu, df_qpu):
        self.df_cpu = df_cpu
        self.df_qpu = df_qpu

    def exp_fit(self, x, A, B):
        return A * np.exp(B * x)

    def plot_scaling_linear_scale(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 

        plt.figure(figsize=(12, 10))

        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(num_qubits_smooth, cpu_fit, '--', label=f'Polynomial Fit (Deg. {degree})', color='orange')

        plt.scatter(qpu_stats["num_qubits"], y_qpu, label="QAOA", color='royalblue')
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label="Exp. Fit", color="royalblue", linewidth=2)

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=19)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, f"cpu_vs_qpu_SA_linear_scale.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_multiple_p(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit and seperating values of p

        """
        if 'p' not in self.df_qpu.columns:
            raise ValueError("Column 'p' not found in QPU data. Make sure 'p' exists in the dataset.")
        
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_p_values = sorted(df["p"].dropna().unique())

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)
        
        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, marker='s', label="SA", color='darkorange')
        plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label="Poly. Fit (Deg. 3)", color="darkorange", linewidth=2)
        
        for p in unique_p_values:
            df_p = df[df["p"] == p]
            
            qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
            qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
            qpu_stats.dropna(inplace=True)
            
            print(f"Processing p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)
            
            X_qpu = qpu_stats["num_qubits"].values
            y_qpu = qpu_stats["qpu_mean"].values
            
            if len(X_qpu) > 2:  
                popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
                qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)
                
                # scaling_factor = (y_cpu[0] / y_qpu[0]) 
                # qpu_best_fit_scaled = qpu_best_fit * scaling_factor
                # y_qpu_scaled = y_qpu * scaling_factor
                
                plt.scatter(X_qpu, y_qpu, label=f"QAOA p={p}", s=60)
                plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Ex.p Fit p={p}", linewidth=2)
        
        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_p.pdf"))
        
        plt.show()
        plt.close()

 
    def plot_scaling_per_p(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit with one plot for each value of p

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_p_values = sorted(df["p"].dropna().unique())

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        for p in unique_p_values:
            df_p = df[df["p"] == p]

            qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
            qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
            qpu_stats.dropna(inplace=True)

            print(f"Processing p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)

            X_qpu = qpu_stats["num_qubits"].values
            y_qpu = qpu_stats["qpu_mean"].values

            plt.figure(figsize=(12, 10))
            plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
            plt.plot(X_cpu, cpu_fit, marker='s', linestyle='--', label="SA Poly. Fit", color='darkorange')

            plt.scatter(X_qpu, y_qpu, label=f"QAOA p={p}", color='royalblue')

            if len(X_qpu) > 1:
                popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu)
                num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)
                qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 
                plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Exp. Fit", color="royalblue", linewidth=2)

            plt.xlabel("Number of Qubits", fontsize=14)
            plt.ylabel("Computational Complexity", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(os.path.join(save_path, f"cpu_vs_qpu_SA_p_{p}.pdf"))

            plt.show()
            plt.close()

    
    def plot_scaling_per_num_rot_p(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit with one plot for each 
        unique value of num_rot and seperate values of p on each plot

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_rotations = sorted(df["num_rot_qpu"].dropna().unique())
        unique_p_values = sorted(df["p"].dropna().unique())
        
        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        for num_rot in unique_rotations:
            df_rot = df[df["num_rot_qpu"] == num_rot]  
            
            plt.figure(figsize=(12, 10))
            plt.scatter(X_cpu, y_cpu, marker='s', label="CPU Calls", color='darkorange', s=80)
            plt.plot(X_cpu, y_cpu, marker='s', linestyle='--', label="SA", color='darkorange')
            plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label="Poly. Fit (Deg. 3)", color="darkorange", linewidth=2)
            
            for p in unique_p_values:
                df_p = df_rot[df_rot["p"] == p]
                
                qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
                qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
                qpu_stats.dropna(inplace=True)
                
                print(f"Processing num_rot={num_rot}, p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)
                
                X_qpu = qpu_stats["num_qubits"].values
                y_qpu = qpu_stats["qpu_mean"].values
                
                if len(X_qpu) > 2: 
                    popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
                    qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)
                                        
                    plt.scatter(X_qpu, y_qpu, label=f"QAOA p={p}", s=60)
                    plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Exp. Fit", linewidth=2)
            
            plt.xlabel("Number of Qubits", fontsize=22)
            plt.ylabel("Computational Complexity", fontsize=22)
            plt.title(f"num_rot={num_rot}", fontsize=24)
            plt.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(os.path.join(save_path, f"cpu_vs_qpu_SA_num_rot_{num_rot}.pdf"))
            
            plt.show()
            plt.close()


    def plot_scaling_semilog_exp_fit(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a semilog scale with an exponential fit

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]
        df_cpu_valid = df[df["cpu_calls"] > 0]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        cpu_stats = df_cpu_valid.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()
        
        X_cpu = cpu_stats["num_qubits"].values
        y_cpu = cpu_stats["mean"].values
        cpu_mask = (X_cpu >= 10) & (X_cpu <= 65)
        X_cpu_fit = X_cpu[cpu_mask]
        y_cpu_fit = y_cpu[cpu_mask]
        
        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        log_y_cpu_fit = np.log(y_cpu_fit)
        coeffs_cpu = np.polyfit(X_cpu_fit, log_y_cpu_fit, deg=1)  
        cpu_fit = np.exp(coeffs_cpu[1]) * np.exp(coeffs_cpu[0] * num_qubits_smooth) 

        # For R² in log space:
        log_y_pred = coeffs_cpu[0] * X_cpu_fit + coeffs_cpu[1]
        r2_exp = r2_score(log_y_cpu_fit, log_y_pred)

        print(f"CPU Exp Fit (log-fit): a={np.exp(coeffs_cpu[1]):.3e}, b={coeffs_cpu[0]:.3f}, R²={r2_exp:.4f}")

        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)

        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label=f"Exp. Fit", color="darkorange", linewidth=2)

        plt.scatter(qpu_stats["num_qubits"], y_qpu, label="QAOA", color='royalblue')
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label="Exp. Fit", color="royalblue", linewidth=2)

        plt.yscale("log")
        plt.xlabel("Number of Qubits", fontsize=24)
        plt.ylabel("Computational Complexity (log)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_semi_log.pdf"))

        plt.show()
        plt.close()

    
    def plot_scaling_loglog_poly_vs_exp(self, save_path=None):
        """
        Plot CPU (polynomial) vs QPU (exponential) scaling on log-log scale.
        """

        # Merge and group QPU stats
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]
        df_cpu_valid = df[df["cpu_calls"] > 0]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        cpu_stats = df_cpu_valid.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()

        # CPU data
        # X_cpu = df_cpu_valid["num_qubits"].values
        # y_cpu = df_cpu_valid["cpu_calls"].values

        X_cpu = cpu_stats["num_qubits"].values
        y_cpu = cpu_stats["mean"].values

        cpu_mask = (X_cpu >= 10) & (X_cpu <= 65)
        X_cpu_fit = X_cpu[cpu_mask]
        y_cpu_fit = y_cpu[cpu_mask]
        
        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 200)

        log_y = np.log(y_cpu_fit)
        coeffs_exp = np.polyfit(X_cpu_fit, log_y, deg=1)
        a_exp = np.exp(coeffs_exp[1])
        b_exp = coeffs_exp[0]

        exp_fit_vals = a_exp * np.exp(b_exp * num_qubits_smooth)

        print(f"Exponential Fit: y = {a_exp:.2e} * exp({b_exp:.2f} * x)")


        # QPU data (filter zero)
        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        mask = y_qpu > 0
        X_qpu = X_qpu[mask]
        y_qpu = y_qpu[mask]

        # Exponential fit (QPU)
        log_y_qpu = np.log(y_qpu)
        coeffs_qpu = np.polyfit(X_qpu, log_y_qpu, deg=1)
        a_qpu = np.exp(coeffs_qpu[1])
        b_qpu = coeffs_qpu[0]

        exp_fit = a_qpu * np.exp(b_qpu * num_qubits_smooth)

        # Scaling CPU/QPU
        scaling_ratio = exp_fit_vals / exp_fit

        # Plot
        plt.figure(figsize=(12, 10))
        plt.loglog(X_cpu, y_cpu, 'o', color='darkorange', label="SA")
        plt.loglog(num_qubits_smooth, exp_fit_vals, '--', color='darkorange', label=f"Poly. Fit (Deg. )")

        plt.loglog(X_qpu, y_qpu, 'o', color='royalblue', label="QAOA")
        plt.loglog(num_qubits_smooth, exp_fit, '-', color='royalblue', label = "Exp. Fit")

        plt.plot(num_qubits_smooth, scaling_ratio, label="Poly / Exp Ratio", color='purple')
        
        n_vals = num_qubits_smooth
        O_n2 = n_vals ** 2
        O_n3 = n_vals ** 3
        O_2n = 2 ** n_vals

        # Normalize for visual comparison (scale to match SA/QAOA range)
        scale_O_n2 = y_cpu[0] / O_n2[0]
        scale_O_n3 = y_cpu[0] / O_n3[0]
        scale_O_2n = y_cpu[0] / O_2n[0]

        plt.plot(n_vals, scale_O_n2 * O_n2, linestyle=":", color="gray", label=r"$\mathcal{O}(n^2)$")
        plt.plot(n_vals, scale_O_n3 * O_n3, linestyle="--", color="gray", label=r"$\mathcal{O}(n^3)$")
        plt.plot(n_vals, scale_O_2n * O_2n, linestyle="-.", color="gray", label=r"$\mathcal{O}(2^n)$")
        
        plt.xlabel("Number of Qubits (log)", fontsize=20)
        plt.ylabel("Computational Complexity (log)", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, which="both", linestyle="--", linewidth=0.2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_loglog_poly_vs_exp_scaling.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_loglog_poly_vs_exp_normalised(self, save_path=None):
        """
        Plot CPU (polynomial) vs QPU (exponential) scaling on log-log scale.
        """

        # Merge and group QPU stats
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]
        df_cpu_valid = df[df["cpu_calls"] > 0]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        cpu_stats = df_cpu_valid.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()

        # CPU data
        # X_cpu = df_cpu_valid["num_qubits"].values
        # y_cpu = df_cpu_valid["cpu_calls"].values

        X_cpu = cpu_stats["num_qubits"].values
        y_cpu = cpu_stats["mean"].values
        
        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 200)

        log_x = np.log(X_cpu)
        log_y = np.log(y_cpu)

        # Try degree 1, 2, 3 for comparison
        deg = 4
        coeffs = np.polyfit(log_x, log_y, deg=deg)
        poly_fit_log = np.poly1d(coeffs)

        # Predict and evaluate
        log_y_fit = poly_fit_log(log_x)
        r2_poly = r2_score(log_y, log_y_fit)
        print(f"Polynomial (deg={deg}) R² in log-log space: {r2_poly:.4f}")

        # Evaluate on smooth range
        log_x_smooth = np.log(num_qubits_smooth)
        log_y_smooth = poly_fit_log(log_x_smooth)
        y_poly_fit = np.exp(log_y_smooth)  # back-transform

        # QPU data (filter zero)
        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        mask = y_qpu > 0
        X_qpu = X_qpu[mask]
        y_qpu = y_qpu[mask]

        # Exponential fit (QPU)
        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.05))
        A_qpu, B_qpu = popt_qpu
        exp_fit = A_qpu * np.exp(B_qpu * num_qubits_smooth)

        # Normalisation curves
        cpu_norm = y_poly_fit / y_poly_fit[0]
        qpu_norm = exp_fit / exp_fit[0] 

        y_cpu_norm = y_cpu / y_cpu[0]
        y_qpu_norm = y_qpu / y_qpu[0]


        # Plot
        plt.figure(figsize=(12, 10))
        plt.loglog(X_cpu, y_cpu_norm, 'o', color='darkorange', label="SA")
        plt.loglog(num_qubits_smooth, cpu_norm, '--', color='darkorange', label=f"Poly. Fit (Deg. {deg})")

        plt.loglog(X_qpu, y_qpu_norm, 'o', color='royalblue', label="QAOA")
        plt.loglog(num_qubits_smooth, qpu_norm, '-', color='royalblue', label = "Exp. Fit")

        plt.xlabel("Number of Qubits (log)", fontsize=20)
        plt.ylabel("Computational Complexity (log)", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, which="both", linestyle="--", linewidth=0.2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_loglog_poly_vs_exp.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_semilog_exp_fit_NN(self, cpu_df_alt=None, label_alt="CPU (Alt)", save_path=None): 
        """
        Plot the scaling of SA vs QPU calls on a semilog scale, with exponential fit.
        Optionally overlay a second CPU dataset.
        """
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]
        df_cpu_valid = df[df["cpu_calls"] > 0]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        cpu_stats = df_cpu_valid.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()

        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]
        
        X_cpu = cpu_stats["num_qubits"].values
        y_cpu = cpu_stats["mean"].values
        num_qubits_smooth = np.linspace(min(X_cpu.min(), X_qpu.min()), max(X_cpu.max(), X_qpu.max()), 200)

        # === CPU Dataset 1 ===
        df_cpu_valid = df[df["cpu_calls"] > 0]
        cpu_mask = (X_cpu >= 10) & (X_cpu <= 65)
        X_cpu_fit = X_cpu[cpu_mask]
        y_cpu_fit = y_cpu[cpu_mask]

        log_y_cpu_fit = np.log(y_cpu_fit)
        coeffs_cpu = np.polyfit(X_cpu_fit, log_y_cpu_fit, deg=1)  
        cpu_fit = np.exp(coeffs_cpu[1]) * np.exp(coeffs_cpu[0] * num_qubits_smooth) 


        # === Optional Second CPU Dataset ===
        if cpu_df_alt is not None:
            df_alt = cpu_df_alt[cpu_df_alt["cpu_calls"] > 0]
            cpu_stats_alt = df_alt.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()

            X_cpu_alt = cpu_stats_alt["num_qubits"].values
            y_cpu_alt = cpu_stats_alt["mean"].values

            cpu_mask_alt = (X_cpu_alt >= 10) & (X_cpu_alt <= 65)
            X_cpu_fit_alt = X_cpu_alt[cpu_mask_alt]
            y_cpu_fit_alt = y_cpu_alt[cpu_mask_alt]

     
            log_y_cpu_fit_alt = np.log(y_cpu_fit_alt)
            coeffs_cpu_alt = np.polyfit(X_cpu_fit_alt, log_y_cpu_fit_alt, deg=1)  
            cpu_fit_alt = np.exp(coeffs_cpu_alt[1]) * np.exp(coeffs_cpu_alt[0] * num_qubits_smooth) 


        # === QPU ===
        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)

        # === Plot ===
        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA Full H", color='darkorange')
        plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label="Exp. Fit", color="darkorange", linewidth=2)

        if cpu_df_alt is not None:
            plt.scatter(X_cpu_alt, y_cpu_alt, label="SA NN", color='seagreen')
            plt.plot(num_qubits_smooth, cpu_fit_alt, linestyle="--", label="Exp. Fit", color="seagreen", linewidth=2)

        plt.scatter(qpu_stats["num_qubits"], y_qpu, label="QAOA", color='royalblue')
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=" Exp. Fit", color="royalblue", linewidth=2)

        plt.yscale("log")
        plt.xlabel("Number of Qubits", fontsize=24)
        plt.ylabel("Computational Complexity (log scale)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_semi_log.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_num_rot_division(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        dividing by vlaue of number of rotamers

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]

        qpu_stats = df_qpu_valid.groupby(["num_qubits", "num_rot_qpu"])["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        num_rot = qpu_stats["num_rot_qpu"].values

        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)        
        
        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 


        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(X_cpu, y_cpu, marker='s', linestyle='--', label="Poly. Fit (Deg. 3)", color='darkorange')

        sc = plt.scatter(qpu_stats["num_qubits"], y_qpu, c=num_rot, cmap="viridis", label="QAOA", s=60)
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label="Exp. Fit", color="royalblue", linewidth=2)

        cbar = plt.colorbar(sc)
        cbar.set_label("Number of Rotations (num_rot)", fontsize=19)

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=19)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_rot_division.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_all_rotations(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        dividing by value of number of rotamers absolute values rather than mean

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values

        X_qpu = df_qpu_valid["num_qubits"].values
        y_qpu = df_qpu_valid["qpu_calls"].values
        num_rot = df_qpu_valid["num_rot_qpu"].values

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)


        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(X_cpu, cpu_fit, marker='s', linestyle='--', label="Poly. Fit (Deg. 3)", color='darkorange')

        unique_rotations = np.unique(num_rot)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rotations)))

        for i, rot in enumerate(unique_rotations):
            mask = num_rot == rot
            X_qpu_subset = X_qpu[mask]
            y_qpu_subset = y_qpu[mask]

            if len(X_qpu_subset) > 1:
                popt_qpu, _ = curve_fit(self.exp_fit, X_qpu_subset, y_qpu_subset, p0=(1, 0.1))
                qpu_fit_curve = self.exp_fit(num_qubits_smooth, *popt_qpu) 
                plt.plot(num_qubits_smooth, qpu_fit_curve, linestyle="-", label=f"QAOA Exp. Fit (num_rot={rot})", color=colors[i], linewidth=2)

            plt.scatter(X_qpu_subset, y_qpu[mask], c=[colors[i]], label=f"num_rot={rot}", s=60)

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_all_rotations.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_linear_exp_fit_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit after training

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        
        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]
        
        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)        
        
        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 

        plt.figure(figsize=(12, 10))

        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(X_cpu, cpu_fit, marker='s', linestyle='--', label="CPU Calls", color='darkorange')

        plt.scatter(qpu_stats["num_qubits"], y_qpu, label="QAOA after Training", color='royalblue')
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", 
                 label= "Exp. Fit", 
                 color="royalblue", linewidth=2)
        

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computaitonal Complexity", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=19)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_linear_exp_trained.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_multiple_p_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit and seperating values of p
        after training
        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_p_values = sorted(df["p"].dropna().unique()) 

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values

        num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, marker='s', label="SA", color='darkorange')
        plt.plot(X_cpu, cpu_fit, marker='s', linestyle='--', label="Poly. Fit (Deg. 3)", color='darkorange')
        
        for p in unique_p_values:
            df_p = df[df["p"] == p]
            
            qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
            qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
            qpu_stats.dropna(inplace=True)
            
            print(f"Processing p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)
            
            X_qpu = qpu_stats["num_qubits"].values
            y_qpu = qpu_stats["qpu_mean"].values
            
            if len(X_qpu) > 2:  
                popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
                qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)
                                
                plt.scatter(X_qpu, y_qpu, label=f"QAOA after Training p={p}", s=60)
                plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Exp. Fit p={p}", linewidth=2)
        
        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_p_trained.pdf"))

        plt.show()
        plt.close()

 
    def plot_scaling_per_p_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit with one plot for each value of p
        after training
        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_p_values = sorted(df["p"].dropna().unique())

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        for p in unique_p_values:
            df_p = df[df["p"] == p]

            qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
            qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
            qpu_stats.dropna(inplace=True)

            print(f"Processing p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)

            X_qpu = qpu_stats["num_qubits"].values
            y_qpu = qpu_stats["qpu_mean"].values

            plt.figure(figsize=(12, 10))
            plt.scatter(X_cpu, y_cpu, marker='s', label="SA", color='darkorange')
            plt.plot(num_qubits_smooth, cpu_fit, '--', label=f'Poly. Fit (Deg. {degree})', color='orange')

            plt.scatter(X_qpu, y_qpu, label=f"QAOA after Training p={p}", color='royalblue')

            if len(X_qpu) > 1:
                popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu)
                num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)
                qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 
                plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Exp. Fit p={p}", color="royalblue", linewidth=2)

            plt.xlabel("Number of Qubits", fontsize=14)
            plt.ylabel("Computational Complexity", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(os.path.join(save_path, f"cpu_vs_qpu_SA_trained_p_{p}.pdf"))

            plt.show()
            plt.close()

    
    def plot_scaling_per_num_rot_p_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a linear scale with an exponential fit with one plot for each 
        unique value of num_rot and seperate values of p on each plot
        after training

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        unique_rotations = sorted(df["num_rot_qpu"].dropna().unique())
        unique_p_values = sorted(df["p"].dropna().unique())
        
        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values
        
        X_cpu = X_cpu[~np.isnan(y_cpu)]  
        y_cpu = y_cpu[~np.isnan(y_cpu)]
        
        num_qubits_smooth = np.linspace(df["num_qubits"].min(), df["num_qubits"].max(), 100)

        degree = 3
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)    

        for num_rot in unique_rotations:
            df_rot = df[df["num_rot_qpu"] == num_rot]  
            
            plt.figure(figsize=(12, 10))
            plt.scatter(X_cpu, y_cpu, marker='s', label="SA", color='darkorange', s=80)
            plt.plot(num_qubits_smooth, cpu_fit, marker='s', linestyle='--', label="Poly. Fit (Deg. 3)", color='darkorange')

            
            for p in unique_p_values:
                df_p = df_rot[df_rot["p"] == p]
                
                qpu_stats = df_p.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
                qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
                qpu_stats.dropna(inplace=True)
                
                print(f"Processing num_rot={num_rot}, p={p}: Unique num_qubits values:", qpu_stats["num_qubits"].values)
                
                X_qpu = qpu_stats["num_qubits"].values
                y_qpu = qpu_stats["qpu_mean"].values
                
                if len(X_qpu) > 2: 
                    popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
                    qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)
                    
                    plt.scatter(X_qpu, y_qpu, label=f"QAOA after Training (Mean) p={p}", s=60)
                    plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label=f"Exp. Fit p={p}", linewidth=2)
            
            plt.xlabel("Number of Qubits", fontsize=22)
            plt.ylabel("Computational Complexity", fontsize=22)
            plt.title(f"CPU vs QPU Calls for num_rot={num_rot}", fontsize=24)
            plt.legend(fontsize=15, loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(os.path.join(save_path, f"cpu_vs_qpu_SA_trained_num_rot_{num_rot}.pdf"))
            
            plt.show()
            plt.close()



    def plot_scaling_semilog_exp_fit_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        on a semilog scale with an exponential fit for trained circuit

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]
        df_cpu_valid = df[df["cpu_calls"] > 0]

        qpu_stats = df_qpu_valid.groupby("num_qubits")["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)
        # cpu_stats = df_cpu_valid.groupby("num_qubits")["cpu_calls"].agg(["mean", "std"]).reset_index()

        # X_cpu = cpu_stats["num_qubits"].values
        # y_cpu = cpu_stats["mean"].values

        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls_mean"].values
        cpu_mask = (X_cpu >= 10) & (X_cpu <= 65)
        X_cpu_fit = X_cpu[cpu_mask]
        y_cpu_fit = y_cpu[cpu_mask]


        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        log_y_cpu_fit = np.log(y_cpu_fit)
        coeffs_cpu = np.polyfit(X_cpu_fit, log_y_cpu_fit, deg=1)  
        cpu_fit = np.exp(coeffs_cpu[1]) * np.exp(coeffs_cpu[0] * num_qubits_smooth) 

        # For R² in log space:
        log_y_pred = coeffs_cpu[0] * X_cpu_fit + coeffs_cpu[1]
        r2_exp = r2_score(log_y_cpu_fit, log_y_pred)

        print(f"CPU Exp Fit (log-fit): a={np.exp(coeffs_cpu[1]):.3e}, b={coeffs_cpu[0]:.3f}, R²={r2_exp:.4f}")

        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu)

        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.errorbar(
            df_cpu_valid["num_qubits"], df_cpu_valid["cpu_mean"],
            yerr=df_cpu_valid["cpu_std"], fmt='o', color='darkorange',
            ecolor='navajowhite', elinewidth=2, capsize=4, label="SA ± std"
        )
        plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label="Exp. Fit", color="darkorange", linewidth=2)

        plt.errorbar(
            qpu_stats["num_qubits"], qpu_stats["qpu_mean"], 
            yerr=qpu_stats["qpu_std"], fmt='o', color='royalblue', 
            ecolor='lightblue', elinewidth=2, capsize=4, label="QAOA Std Dev"
        )

        plt.scatter(qpu_stats["num_qubits"], y_qpu, label="QAOA after Training", color='royalblue')
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label="Exp. Fit", color="royalblue", linewidth=2)

        plt.yscale("log")
        plt.xlabel("Number of Qubits", fontsize=24)
        plt.ylabel("Computational Complexity (log)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_semi_log_trained.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_num_rot_division_trained(self, save_path=None):
        """"
        Function to plot the scaling of the SA vs the calls to the QPU
        dividing by vlaue of number of rotamers

        """
        df = pd.merge(self.df_cpu, self.df_qpu, on="num_qubits", how="outer", suffixes=('_cpu', '_qpu'))
        df_qpu_valid = df[df["qpu_calls"].notna()]

        qpu_stats = df_qpu_valid.groupby(["num_qubits", "num_rot_qpu"])["qpu_calls"].agg(["mean", "std"]).reset_index()
        qpu_stats.rename(columns={"mean": "qpu_mean", "std": "qpu_std"}, inplace=True)

        df_cpu_valid = df[df["cpu_calls"] > 0]
        X_cpu = df_cpu_valid["num_qubits"].values
        y_cpu = df_cpu_valid["cpu_calls"].values

        num_rot = qpu_stats["num_rot_qpu"].values

        X_qpu = qpu_stats["num_qubits"].values
        y_qpu = qpu_stats["qpu_mean"].values
        y_qpu = y_qpu[y_qpu > 0]

        num_qubits_smooth = np.linspace(X_cpu.min(), X_cpu.max(), 100)

        degree = 5
        coeffs = np.polyfit(X_cpu, y_cpu, deg=degree)
        poly_fit = np.poly1d(coeffs)
        cpu_fit = poly_fit(num_qubits_smooth)

        popt_qpu, _ = curve_fit(self.exp_fit, X_qpu, y_qpu, p0=(1, 0.1))
        qpu_best_fit = self.exp_fit(num_qubits_smooth, *popt_qpu) 

        plt.figure(figsize=(12, 10))
        plt.scatter(X_cpu, y_cpu, label="SA", color='darkorange')
        plt.plot(num_qubits_smooth, cpu_fit, linestyle="--", label="Poly. Fit (Deg. 3)", color="darkorange", linewidth=2)

        sc = plt.scatter(qpu_stats["num_qubits"], y_qpu, c=num_rot, cmap="viridis", label="QAOA after Training", s=60)
        plt.plot(num_qubits_smooth, qpu_best_fit, linestyle="-", label="Exp. Fit", color="royalblue", linewidth=2)

        cbar = plt.colorbar(sc)
        cbar.set_label("Number of Rotations (num_rot)", fontsize=19)

        plt.xlabel("Number of Qubits", fontsize=22)
        plt.ylabel("Computational Complexity", fontsize=22)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        plt.legend(fontsize=19)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_rot_division_trained.pdf"))

        plt.show()
        plt.close()
