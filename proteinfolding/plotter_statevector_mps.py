import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def pooled_std(std_values):
    std_values = std_values.dropna()
    n = len(std_values)
    if n <= 1:
        return np.nan
    return np.sqrt((std_values**2).sum() / n)

def aggregate_stats(df, mean_col, std_col, samples_per_row):
    def std_or_pooled(group):
        if len(group) == 1:
            return group[std_col].iloc[0]
        else:
            # Convert std to variance, average, then back to std
            pooled_var = (group[std_col] ** 2).mean()
            return np.sqrt(pooled_var)

    result = df.groupby("num_qubits", group_keys=False).apply(
        lambda group: pd.Series({
            "num_qubits": group.name,  # explicitly include it
            mean_col: group[mean_col].mean(),
            std_col.replace("mean", "std"): std_or_pooled(group)
        })
    ).reset_index(drop=True)
    return result

def aggregate_stats_conv(df, mean_col, std_col, samples_per_row=1000):
    df = df.copy()
    df.columns = df.columns.map(str).str.strip()  # normalize
    return df.groupby(['num_res', 'num_rot', 'p', 'num_qubits']).agg({
        mean_col: 'mean',
        std_col: 'std'
    }).reset_index()


def aggregate_stats_with_n(df, mean_col, std_col, n_col):
    def pooled_std_weighted(group):
        if len(group) == 1:
            return group[std_col].iloc[0]
        else:
            dof = group[n_col] - 1  # degrees of freedom per row
            numerator = (dof * group[std_col] ** 2).sum()
            denominator = dof.sum()
            return np.sqrt(numerator / denominator) if denominator > 0 else np.nan

    return df.groupby("num_qubits").apply(
        lambda group: pd.Series({
            mean_col: group[mean_col].mean(),
            std_col.replace("mean", "std"): pooled_std_weighted(group)
        })
    ).reset_index()


def leave_one_out_fit_stability(x, y, exp_func):
    slopes = []
    for i in range(len(x)):
        x_cv = np.delete(x, i)
        y_cv = np.delete(y, i)

        try:
            A0 = y_cv[0]
            B0 = (np.log(y_cv[-1]) - np.log(y_cv[0])) / (x_cv[-1] - x_cv[0])
            popt, _ = curve_fit(exp_func, x_cv, y_cv, p0=(A0, B0))
            slopes.append(popt[1])
        except RuntimeError:
            continue  # skip failed fits

    slopes = np.array(slopes)
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    return mean_slope, std_slope, slopes

def apply_convergence_correction(stats_df, success_df, method, join_key='num_qubits', ratio_col='mean_convergence_ratio'):
    """Apply convergence correction to the computational cost by dividing by the success ratio."""
    corrected_df = stats_df.copy()
    if method == 'SA':
        # Merge by num_qubits and correct the mean and std
        corrected_df = corrected_df.merge(success_df[[join_key, ratio_col]], on=join_key, how='left')
        corrected_df[ratio_col] = corrected_df[ratio_col].fillna(1.0)
        corrected_df['corrected_mean'] = corrected_df['cpu_calls_mean'] / corrected_df[ratio_col]
        corrected_df['corrected_std'] = corrected_df['std_cpu_calls'] / corrected_df[ratio_col]
    elif method == 'MPS':
        print("Checking MPS success_df columns:", success_df.columns.tolist())
        print("First few rows of success_df:")
        print(success_df.head())
            # Merge by Res, Rot, p to align
        corrected_df = corrected_df.merge(success_df[['num_res', 'num_rot', 'p', 'ratio']], on=['num_res', 'num_rot', 'p'], how='left')
        corrected_df['ratio'] = corrected_df['ratio'].fillna(1.0)
        corrected_df['corrected_mean'] = corrected_df['qpu_calls_mean'] / corrected_df['ratio']
        corrected_df['corrected_std'] = corrected_df['std_qpu_calls'] / corrected_df['ratio']
    else:
        # For SV, no correction needed
        corrected_df['corrected_mean'] = corrected_df['qpu_calls_mean']
        corrected_df['corrected_std'] = corrected_df['std_qpu_calls']
    return corrected_df


class DataPlotter:
    def __init__(self, df_cpu, df_qpu_sv, df_qpu_mps):
        self.df_cpu = df_cpu
        self.df_qpu_sv = df_qpu_sv
        self.df_qpu_mps = df_qpu_mps
        self.df_qpu_mps.columns = self.df_qpu_mps.columns.map(str).str.strip()

        self.fits = []

    def exp_fit(self, x, A, B):
        return A * np.exp(B * x)

    def fit_and_plot(self, df, x_col, y_col, std_col, label, color, linestyle, fit_color,   
                 num_qubits_smooth, fit_mask_min, ax, marker_size=50):
        x = df[x_col].values
        y = df[y_col].values
        yerr = df[std_col].values

        y = np.clip(y, 1e-12, None)
        yerr = np.clip(yerr, 1e-12, None)

        rel_err = yerr / y
        log_lower = y / np.exp(rel_err)
        log_upper = y * np.exp(rel_err)
        asym_err = np.vstack([y - log_lower, log_upper - y])

        # Mask for fit region
        fit_mask = x >= fit_mask_min
        x_fit = x[fit_mask]
        y_fit = y[fit_mask]

        if len(x_fit) < 2:
            print(f"❌ Not enough points to fit ({len(x_fit)} < 2) for {label}")
            return None, None


        ax.errorbar(x[fit_mask], y[fit_mask], yerr=asym_err[:, fit_mask], fmt='o',
            color=color, ecolor=color, elinewidth=1, capsize=2, alpha=0.8,
            markersize=marker_size / 5, markerfacecolor=color, markeredgecolor=color)

        # ax.scatter(x[fit_mask], y[fit_mask], label=label, color=color, s=marker_size)
        ax.errorbar(x[~fit_mask], y[~fit_mask], yerr=asym_err[:, ~fit_mask], fmt='o',
                    markerfacecolor='white', markeredgecolor=color, ecolor=color,
                    elinewidth=1, capsize=2, alpha=0.8,
                    markersize=marker_size / 5)

        # ax.scatter(x[~fit_mask], y[~fit_mask], facecolors='none', edgecolors=color, s=marker_size)

        try:
            A0 = y_fit[0]
            B0 = (np.log(y_fit[-1]) - np.log(y_fit[0])) / (x_fit[-1] - x_fit[0])
            popt, pcov = curve_fit(self.exp_fit, x_fit, y_fit, p0=(A0, B0))

            log_y = np.log(y_fit)
            log_y_pred = popt[1] * x_fit + np.log(popt[0])
            r2 = r2_score(log_y, log_y_pred)
            loocv_std = None
            if len(x_fit) >= 4:
                loocv_mean, loocv_std, _ = leave_one_out_fit_stability(x_fit, y_fit, self.exp_fit)
                print(f"{label} LOOCV Slope: {loocv_mean:.3f} ± {loocv_std:.3f}")

            full_fit_y = self.exp_fit(num_qubits_smooth, *popt)
            ax.plot(num_qubits_smooth, full_fit_y, linestyle=':', color=fit_color, linewidth=1.0, alpha=0.6)

            solid_fit_x = np.linspace(x_fit.min(), x_fit.max(), 200)
            solid_fit_y = self.exp_fit(solid_fit_x, *popt)
            ax.plot(solid_fit_x, solid_fit_y, linestyle='-', color=fit_color, linewidth=1.0)

            J = np.vstack([
                np.exp(popt[1] * solid_fit_x),
                popt[0] * solid_fit_x * np.exp(popt[1] * solid_fit_x)
            ]).T
            fit_var = np.sum(J @ pcov * J, axis=1)
            fit_std = np.sqrt(fit_var)
            ax.fill_between(solid_fit_x, solid_fit_y - fit_std, solid_fit_y + fit_std,
                            color=fit_color, alpha=0.2, linewidth=0)

            slope = popt[1]
            slope_err = np.sqrt(pcov[1, 1])
            print(f"{label} Exp Fit: A = {popt[0]:.2e}, B = {slope:.3f} ± {slope_err:.3f}, R² = {r2:.4f}")

            self.fits.append({
                "label": label,
                "A": popt[0],
                "B": popt[1],
                "B_err": slope_err,
                "B_loocv_std": loocv_std,
                "color": fit_color,
                "range": (x_fit.min(), x_fit.max())
            })

            return slope, slope_err

        except RuntimeError:
            print(f"Fit failed for {label}")
            return None, None
    
    def plot_qpu_calls_by_p(self, df, save_path=None):
        df["num_qubits"] = pd.to_numeric(df["num_qubits"], errors="coerce")
        df["qpu_calls"] = pd.to_numeric(df["qpu_calls"], errors="coerce")
        df["num_res"] = pd.to_numeric(df["num_res"], errors="coerce")
        df["num_rot"] = pd.to_numeric(df["num_rot"], errors="coerce")
        df["p"] = pd.to_numeric(df["p"], errors="coerce")
        df = df.dropna(subset=["p", "num_qubits", "qpu_calls"])

        # Define p ranges
        p_low = [4]

        def plot_for_ps(p_list, suffix):
            ps = sorted([p for p in df["p"].unique() if p in p_list])
            colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))
            all_qubits = np.sort(df["num_qubits"].unique())
            num_qubits_smooth = np.linspace(all_qubits.min(), all_qubits.max(), 200)

            plt.figure(figsize=(11, 10))

            for i, p_val in enumerate(ps):
                df_p = df[df["p"] == p_val]

                grouped = df_p.groupby("num_qubits")["qpu_calls"]
                stats = grouped.agg(qpu_calls_mean="mean", std_qpu_calls="std").reset_index()

                p_val_float = float(p_val)
                label = f"p = {int(p_val_float)}" if p_val_float.is_integer() else f"p = {p_val_float}"
                self.fit_and_plot_semilog(
                    stats,
                    x_col="num_qubits",
                    y_col="qpu_calls_mean",
                    std_col="std_qpu_calls",
                    label=label,
                    color=colors[i],
                    linestyle='-',
                    fit_color=colors[i],
                    num_qubits_smooth=num_qubits_smooth,
                    fit_mask_min=10  # fixed for more stable fits
                )

            plt.yscale("log")
            plt.xlabel("Number of Qubits (M)", fontsize=24)
            plt.ylabel("QPU Calls (log scale)", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=18, loc='lower right')
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(os.path.join(save_path, f"qpu_calls_by_p_{suffix}.pdf"))

            plt.show()
            plt.close()

        # Plot for selected p values
        plot_for_ps(p_low, "p1_3")


    def fit_and_plot_semilog(self, df, x_col, y_col, std_col, label, color, linestyle, fit_color, num_qubits_smooth, fit_mask_min):
        # Prepare values
        x = df[x_col].values
        y = df[y_col].values
        yerr = df[std_col].values

        # Avoid zeros
        y = np.clip(y, 1e-12, None)
        yerr = np.clip(yerr, 1e-12, None)

        # Log-scale consistent error bars (multiplicative)
        rel_err = yerr / y
        log_lower = y / np.exp(rel_err)
        log_upper = y * np.exp(rel_err)
        asym_err = np.vstack([y - log_lower, log_upper - y])

        # Mask for fit region
        fit_mask = x >= fit_mask_min
        x_fit = x[fit_mask]
        y_fit = y[fit_mask]
        if len(x_fit) < 2:
            print(f"Not enough points to fit for {label}")
            return None, None
        yerr_fit = yerr[fit_mask]

        # For the fit region
        log_y_fit = np.log(y_fit)
        log_yerr_fit = yerr_fit / y_fit
        yerr_fit_plot = [
            y_fit - np.exp(log_y_fit - log_yerr_fit),
            np.exp(log_y_fit + log_yerr_fit) - y_fit
        ]

        x_unfit = x[~fit_mask]
        y_unfit = y[~fit_mask]
        yerr_unfit = yerr[~fit_mask]

        # For the unfit region
        log_y_unfit = np.log(y_unfit)
        log_yerr_unfit = yerr_unfit / y_unfit
        yerr_unfit_plot = [
            y_unfit - np.exp(log_y_unfit - log_yerr_unfit),
            np.exp(log_y_unfit + log_yerr_unfit) - y_unfit
        ]


        # Plot data points
        plt.errorbar(x[fit_mask], y[fit_mask], yerr=yerr_fit_plot, fmt='o',
                    color=color, ecolor=color, elinewidth=2, capsize=4, alpha=0.5)
        plt.scatter(x[fit_mask], y[fit_mask], label=label, color=color)

        plt.errorbar(x[~fit_mask], y[~fit_mask], yerr=yerr_unfit_plot, fmt='o',
                    markerfacecolor='none', markeredgecolor=color, ecolor=color,
                    elinewidth=2, capsize=4, alpha=0.5)
        plt.scatter(x[~fit_mask], y[~fit_mask], facecolors='none', edgecolors=color)

        try:
            # Fit
            A0 = y_fit[0]
            B0 = (np.log(y_fit[-1]) - np.log(y_fit[0])) / (x_fit[-1] - x_fit[0])
            popt, pcov = curve_fit(self.exp_fit, x_fit, y_fit, p0=(A0, B0))

            # popt, pcov = curve_fit(self.exp_fit, x_fit, y_fit, p0=(1, 0.1))
            log_y = np.log(y_fit)
            log_y_pred = popt[1] * x_fit + np.log(popt[0])
            r2 = r2_score(log_y, log_y_pred)
            loocv_std = None 
            if len(x_fit) >= 4:  # At least 4 points to do LOOCV
                loocv_mean, loocv_std, loocv_all = leave_one_out_fit_stability(x_fit, y_fit, self.exp_fit)
                print(f"{label} LOOCV Slope: {loocv_mean:.3f} ± {loocv_std:.3f}")

           # Dotted full-range fit using full smooth range
            full_fit_y = self.exp_fit(num_qubits_smooth, *popt)
            plt.plot(num_qubits_smooth, full_fit_y, linestyle=':', color=fit_color, linewidth=2, alpha=0.6)

            # Solid fit only in fit region
            solid_fit_x = np.linspace(x_fit.min(), x_fit.max(), 200)
            solid_fit_y = self.exp_fit(solid_fit_x, *popt)
            plt.plot(solid_fit_x, solid_fit_y, linestyle='-', color=fit_color, linewidth=2)
            
            # Error bar on fit (± 1σ confidence interval)
            # Get predicted fit and calculate standard error
            # Error bar (±1σ) ONLY in fit region
            fit_range_x = np.linspace(x_fit.min(), x_fit.max(), 200)
            fit_range_y = self.exp_fit(fit_range_x, *popt)

            # Jacobian for error band
            J = np.vstack([
                np.exp(popt[1] * fit_range_x),                        # ∂f/∂A
                popt[0] * fit_range_x * np.exp(popt[1] * fit_range_x) # ∂f/∂B
            ]).T

            # Standard deviation of prediction
            fit_var = np.sum(J @ pcov * J, axis=1)
            fit_std = np.sqrt(fit_var)

            # Plot the confidence band only in the fit region
            plt.fill_between(fit_range_x,
                            fit_range_y - fit_std,
                            fit_range_y + fit_std,
                            color=fit_color,
                            alpha=0.2,
                            linewidth=0)
            
            slope = popt[1]
            slope_err = np.sqrt(pcov[1, 1]) 
            print(f"{label} Exp Fit: A = {popt[0]:.2e}, B = {slope:.3f} ± {slope_err:.3f}, R² = {r2:.4f}")

            # Store fit parameters for crossover calculation
            self.fits.append({
                "label": label,
                "A": popt[0],
                "B": popt[1],
                "B_err": slope_err, 
                "B_loocv_std": loocv_std,
                "color": fit_color,
                "range": (x_fit.min(), x_fit.max())
            })

            return slope, slope_err

        except RuntimeError:
            print(f"Fit failed for {label}")
            return None, None 

    
           

    def plot_scaling_semilog_exp_fit(self, save_path=None):
         
        cpu_stats = aggregate_stats(self.df_cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100)
        sv_stats = aggregate_stats_with_n(self.df_qpu_sv, "qpu_calls_mean", "std_qpu_calls", "n_runs")
        mps_stats = aggregate_stats(self.df_qpu_mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000)

        # Smooth x range for exponential fit
        min_q = min(cpu_stats["num_qubits"].min(), sv_stats["num_qubits"].min(), mps_stats["num_qubits"].min())
        max_q = max(cpu_stats["num_qubits"].max(), sv_stats["num_qubits"].max(), mps_stats["num_qubits"].max())
        num_qubits_smooth = np.linspace(min_q, max_q, 200)

        # Plot
        plt.figure(figsize=(11, 10))
        self.fit_and_plot_semilog(cpu_stats, "num_qubits", "cpu_calls_mean", "std_cpu_calls",
                "SA", "darkorange", "--", "darkorange", num_qubits_smooth, fit_mask_min=1)

        self.fit_and_plot_semilog(sv_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        "SV QAOA", "royalblue", "-", "royalblue", num_qubits_smooth, fit_mask_min=15)

        self.fit_and_plot_semilog(mps_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        "MPS QAOA", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, fit_mask_min=1)
        
        plt.style.use('/Users/aag/Documents/proteinfolding/proteinfolding/molecular.mplstyle')

        plt.yscale("log")
        plt.xlabel("Number of Qubits (M)", fontsize=24)
        plt.ylabel("Computational Cost (log scale)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20, loc='lower right')
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_SA_semi_log_statevector+mps.pdf"))

        plt.show()
        plt.close()


    def plot_scaling_normalized(self, save_path=None):
        self.fits = []

        cpu_stats = aggregate_stats(self.df_cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100)
        sv_stats = aggregate_stats_with_n(self.df_qpu_sv, "qpu_calls_mean", "std_qpu_calls", "n_runs")
        mps_stats = aggregate_stats(self.df_qpu_mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000)

        # Normalize
        cpu_stats["cpu_calls_mean"] /= 1e9
        cpu_stats["std_cpu_calls"] /= 1e9
        sv_stats["qpu_calls_mean"] /= 1e3
        sv_stats["std_qpu_calls"] /= 1e3
        mps_stats["qpu_calls_mean"] /= 1e4
        mps_stats["std_qpu_calls"] /= 1e4

        min_q = min(cpu_stats["num_qubits"].min(), sv_stats["num_qubits"].min())
        num_qubits_smooth = np.linspace(min_q, 160, 160)

        # Make the figure square-ish
        fig, ax = plt.subplots(figsize=(11, 10))

        # # Inset axis with better proportions
        # axins = inset_axes(
        #     ax,
        #     width=3, height=3,     # inches
        #     loc='upper left',
        #     bbox_to_anchor=(0.08, 0.95),
        #     bbox_transform=ax.transAxes,
        #     borderpad=0
        # )

        # Plot main fits
        self.fit_and_plot(cpu_stats, "num_qubits", "cpu_calls_mean", "std_cpu_calls",
                        r"SA (/ 10^9 Hz)", "darkorange", "--", "darkorange", num_qubits_smooth, 18, ax, marker_size=32)
        self.fit_and_plot(sv_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        r"SV QAOA (/ 10^3 Hz)", "royalblue", "-", "royalblue", num_qubits_smooth, 15, ax, marker_size=32)
        self.fit_and_plot(mps_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        r"MPS QAOA (/ 10^4 Hz)", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, 12, ax, marker_size=32)

        # # Inset fits (no legend, smaller points)
        # self.fit_and_plot(cpu_stats, "num_qubits", "cpu_calls_mean", "std_cpu_calls",
        #                 "", "darkorange", "--", "darkorange", num_qubits_smooth, 18, axins, marker_size=10)
        # self.fit_and_plot(sv_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
        #                 "", "royalblue", "-", "royalblue", num_qubits_smooth, 10, axins, marker_size=10)
        # self.fit_and_plot(mps_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
        #                 "", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, 12, axins, marker_size=10)

        # Error band MPS
        mps_fit = self.fits[2]
        A, B = mps_fit['A'], mps_fit['B']
        B_err = mps_fit.get('B_err', 0.0)
        mps_lower = A * np.exp((B - B_err) * num_qubits_smooth) 
        mps_upper = A * np.exp((B + B_err) * num_qubits_smooth)

        ax.fill_between(num_qubits_smooth, mps_lower, mps_upper, color='mediumseagreen', alpha=0.1, label='MPS ±1σ')
        # axins.fill_between(num_qubits_smooth, mps_lower, mps_upper, color='royalblue', alpha=0.1)

         # Error band for SA
        sa_fit = self.fits[0]
        A_sa, B_sa = sa_fit['A'], sa_fit['B']
        B_sa_err = sa_fit.get('B_err', 0.0)
        sa_lower = A_sa * np.exp((B_sa - B_sa_err) * num_qubits_smooth)
        sa_upper = A_sa * np.exp((B_sa + B_sa_err) * num_qubits_smooth)

        ax.fill_between(num_qubits_smooth, sa_lower, sa_upper, color='darkorange', alpha=0.1, label='SA ±1σ')

        # Main plot config
        ax.set_yscale("log")
        ax.set_xlabel("Number of Qubits (M)", fontsize=24)
        ax.set_ylabel("Estimated Runtime (log scale)", fontsize=24)
        ax.tick_params(labelsize=20)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=20, loc='lower right')

        # # Inset limits and style
        # axins.set_xlim(min_q, 150)
        # axins.set_ylim(1e-5, 2e2)
        # axins.set_yscale("log")
        # axins.tick_params(labelsize=12)
        # axins.grid(True, linestyle="--", alpha=0.5)
        # --- CROSSOVER COMPUTATION ---
        def compute_crossover_point(A1, B1, A2, B2):
            if B1 == B2:
                return None, None
            x_cross = np.log(A1 / A2) / (B2 - B1)
            y_cross = A1 * np.exp(B1 * x_cross)
            return x_cross, y_cross

        sa_fit = self.fits[0]
        sv_fit = self.fits[1]
        mps_fit = self.fits[2]

        A_sa, B_sa, B_sa_err = sa_fit["A"], sa_fit["B"], sa_fit.get("B_err", 0.0)
        A_sv, B_sv = sv_fit["A"], sv_fit["B"]
        A_mps, B_mps, B_mps_err = mps_fit["A"], mps_fit["B"], mps_fit.get("B_err", 0.0)

        B_sa_minus = B_sa - B_sa_err
        B_sa_plus = B_sa + B_sa_err
        B_mps_minus = B_mps - B_mps_err
        B_mps_plus = B_mps + B_mps_err
        A_mps_upper = A_mps 
        A_mps_lower = A_mps

        crossovers = {
            "SA vs SV central": compute_crossover_point(A_sa, B_sa, A_sv, B_sv),
            "SA vs MPS central": compute_crossover_point(A_sa, B_sa, A_mps, B_mps),
            "SA lower vs MPS upper": compute_crossover_point(A_sa, B_sa_minus, A_mps_upper, B_mps_plus),
            "SA upper vs MPS lower": compute_crossover_point(A_sa, B_sa_plus, A_mps_upper, B_mps_plus),
        }

        print("\n📈 Central Fit Crossovers")
        if crossovers["SA vs SV central"][0]:
            print(f"- SA vs SV: x = {crossovers['SA vs SV central'][0]:.2f} qubits, "
                f"runtime ≈ {crossovers['SA vs SV central'][1]:.2e} s")
        if crossovers["SA vs MPS central"][0]:
            print(f"- SA vs MPS: x = {crossovers['SA vs MPS central'][0]:.2f} qubits, "
                f"runtime ≈ {crossovers['SA vs MPS central'][1]:.2e} s")

        print("\n📉 Error Band Crossovers")
        if crossovers["SA lower vs MPS upper"][0]:
            print(f"- SA lower vs MPS upper: x = {crossovers['SA lower vs MPS upper'][0]:.2f} qubits")
        if crossovers["SA upper vs MPS lower"][0]:
            print(f"- SA upper vs MPS lower: x = {crossovers['SA upper vs MPS lower'][0]:.2f} qubits")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_sv_normalized_inset.pdf"))

        plt.show()
        plt.close()

    def plot_scaling_semilog_exp_fit_per_num_rot(self, save_path=None):
        # Union instead of intersection
        unique_num_rots = sorted(
            set(self.df_cpu['num_rot']).union(
                self.df_qpu_sv['num_rot']).union(
                self.df_qpu_mps['num_rot'])
        )

        for num_rot in unique_num_rots:
            print(f"\n📊 Plotting semilog scaling for num_rot = {num_rot}")
            self.fits = []

            # Filter
            cpu = self.df_cpu[self.df_cpu['num_rot'] == num_rot]
            sv = self.df_qpu_sv[self.df_qpu_sv['num_rot'] == num_rot]
            mps = self.df_qpu_mps[self.df_qpu_mps['num_rot'] == num_rot]

            # Skip completely empty
            if cpu.empty and sv.empty and mps.empty:
                print(f"❌ Skipping num_rot = {num_rot}, no data at all")
                continue

            # Aggregate
            cpu_stats = aggregate_stats(cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100) if not cpu.empty else pd.DataFrame()
            sv_stats  = aggregate_stats_with_n(sv, "qpu_calls_mean", "std_qpu_calls", "n_runs") if not sv.empty else pd.DataFrame()
            mps_stats = aggregate_stats(mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000) if not mps.empty else pd.DataFrame()

            # Compute min and max qubit range from existing stats
            q_vals = []
            for df in [cpu_stats, sv_stats, mps_stats]:
                if not df.empty:
                    q_vals.append(df["num_qubits"].min())
                    q_vals.append(df["num_qubits"].max())
            if not q_vals:
                print(f"⚠️ Skipping num_rot = {num_rot}, no valid qubit data")
                continue

            min_q, max_q = min(q_vals), max(q_vals)
            num_qubits_smooth = np.linspace(min_q, max_q, 200)

            # Plot
            plt.figure(figsize=(11, 10))

            if not cpu_stats.empty:
                self.fit_and_plot_semilog(cpu_stats, "num_qubits", "cpu_calls_mean", "std_cpu_calls",
                        "SA", "darkorange", "--", "darkorange", num_qubits_smooth, fit_mask_min=18)

            if not sv_stats.empty:
                self.fit_and_plot_semilog(sv_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        "SV QAOA", "royalblue", "-", "royalblue", num_qubits_smooth, fit_mask_min=14)

            if not mps_stats.empty:
                self.fit_and_plot_semilog(mps_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                        "MPS QAOA", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, fit_mask_min=18)

            if len(self.fits) == 0:
                print(f"⚠️ Skipping plot for num_rot = {num_rot}, no successful fits")
                plt.close()
                continue

            # Format
            plt.yscale("log")
            plt.xlabel("Number of Qubits (M)", fontsize=24)
            plt.ylabel("Computational Complexity (log scale)", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20, loc='lower right')
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                filename = f"cpu_vs_qpu_semilog_numrot_{num_rot}.pdf"
                plt.savefig(os.path.join(save_path, filename))

            plt.show()
            plt.close()


    def plot_scaling_normalized_per_num_rot(self, save_path=None):
        unique_num_rots = sorted(set(self.df_cpu['num_rot']) | 
                                set(self.df_qpu_sv['num_rot']) | 
                                set(self.df_qpu_mps['num_rot']))

        for num_rot in unique_num_rots:
            print(f"\n📊 Plotting normalized scaling for num_rot = {num_rot}")
            self.fits = []

            cpu = self.df_cpu[self.df_cpu['num_rot'] == num_rot]
            sv  = self.df_qpu_sv[self.df_qpu_sv['num_rot'] == num_rot]
            mps = self.df_qpu_mps[self.df_qpu_mps['num_rot'] == num_rot]

            if cpu.empty and sv.empty and mps.empty:
                print(f"❌ Skipping num_rot = {num_rot}, no data available")
                continue

            # Aggregate
            cpu_stats = aggregate_stats(cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100) if not cpu.empty else pd.DataFrame()
            sv_stats  = aggregate_stats_with_n(sv, "qpu_calls_mean", "std_qpu_calls", "n_runs") if not sv.empty else pd.DataFrame()
            mps_stats = aggregate_stats(mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000) if not mps.empty else pd.DataFrame()

            # Normalize if non-empty
            if not cpu_stats.empty:
                cpu_stats["cpu_calls_mean"] /= 1e9
                cpu_stats["std_cpu_calls"] /= 1e9
            if not sv_stats.empty:
                sv_stats["qpu_calls_mean"] /= 1e3
                sv_stats["std_qpu_calls"] /= 1e3
            if not mps_stats.empty:
                mps_stats["qpu_calls_mean"] /= 1e4
                mps_stats["std_qpu_calls"] /= 1e4

            # Determine num_qubits range for fitting
            q_vals = []
            for df in [cpu_stats, sv_stats, mps_stats]:
                if not df.empty:
                    q_vals.append(df["num_qubits"].min())
                    q_vals.append(df["num_qubits"].max())
            if not q_vals:
                print(f"⚠️ Skipping num_rot = {num_rot}, no valid qubit range")
                continue
            min_q = min(q_vals)
            num_qubits_smooth = np.linspace(min_q, 160, 160)

            # Plot
            fig, ax = plt.subplots(figsize=(11, 10))

            if not cpu_stats.empty:
                self.fit_and_plot(cpu_stats, "num_qubits", "cpu_calls_mean", "std_cpu_calls",
                                r"SA (/ 10^9 Hz)", "darkorange", "--", "darkorange", num_qubits_smooth, 18, ax, marker_size=32)
            if not sv_stats.empty:
                self.fit_and_plot(sv_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                                r"SV QAOA (/ 10^3 Hz)", "royalblue", "-", "royalblue", num_qubits_smooth, 14, ax, marker_size=32)
            if not mps_stats.empty:
                self.fit_and_plot(mps_stats, "num_qubits", "qpu_calls_mean", "std_qpu_calls",
                                r"MPS QAOA (/ 10^4 Hz)", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, 12, ax, marker_size=32)

            # If no successful fit, skip
            if len(self.fits) == 0:
                print(f"⚠️ Skipping plot for num_rot = {num_rot}, no successful fits")
                plt.close()
                continue

            # Error bands (only if available)
            for i, fit in enumerate(self.fits):
                label = fit["label"]
                A, B = fit["A"], fit["B"]
                B_err = fit.get("B_err", 0.0)
                color = fit["color"]
                ax.fill_between(num_qubits_smooth,
                                A * np.exp((B - B_err) * num_qubits_smooth),
                                A * np.exp((B + B_err) * num_qubits_smooth),
                                color=color,
                                alpha=0.1,
                                label=f"{label} ±1σ (slope)")

            ax.set_yscale("log")
            ax.set_xlabel("Number of Qubits (M)", fontsize=24)
            ax.set_ylabel("Estimated Runtime (log scale)", fontsize=24)
            ax.tick_params(labelsize=20)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=20, loc='lower right')

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                filename = f"cpu_vs_sv_normalized_numrot_{num_rot}.pdf"
                plt.savefig(os.path.join(save_path, filename))

            plt.show()
            plt.close()


    
    def plot_scaling_with_convergence_correction(self, sa_success, mps_success, save_path=None):
        print("df_qpu_mps columns:", self.df_qpu_mps.columns.tolist())
        cpu_stats = aggregate_stats(self.df_cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100)
        sv_stats = aggregate_stats_with_n(self.df_qpu_sv, "qpu_calls_mean", "std_qpu_calls", "n_runs")
        mps_stats = aggregate_stats_conv(self.df_qpu_mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000)

        corrected_cpu = apply_convergence_correction(cpu_stats, sa_success, method='SA')
        corrected_sv = apply_convergence_correction(sv_stats, None, method='SV')
        corrected_mps = apply_convergence_correction(mps_stats, mps_success, method='MPS')

        min_q = min(corrected_cpu["num_qubits"].min(), corrected_sv["num_qubits"].min(), corrected_mps["num_qubits"].min())
        max_q = max(corrected_cpu["num_qubits"].max(), corrected_sv["num_qubits"].max(), corrected_mps["num_qubits"].max())
        num_qubits_smooth = np.linspace(min_q, max_q, 200)

        plt.figure(figsize=(11, 10))

        self.fit_and_plot_semilog(corrected_cpu, "num_qubits", "corrected_mean", "corrected_std",
                            "SA (corrected)", "darkorange", "--", "darkorange", num_qubits_smooth, fit_mask_min=1)

        self.fit_and_plot_semilog(corrected_sv, "num_qubits", "corrected_mean", "corrected_std",
                            "SV QAOA", "royalblue", "-", "royalblue", num_qubits_smooth, fit_mask_min=15)

        self.fit_and_plot_semilog(corrected_mps, "num_qubits", "corrected_mean", "corrected_std",
                            "MPS QAOA (corrected)", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, fit_mask_min=1)

        plt.yscale("log")
        plt.xlabel("Number of Qubits (M)", fontsize=24)
        plt.ylabel("Computational Cost (log scale)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20, loc='lower right')
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_corrected_semilog.pdf"))

        plt.show()
        plt.close()

    def plot_scaling_normalized_with_convergence_correction(self, sa_success, mps_success, save_path=None):
        self.fits = []

        # 1. Compute aggregate stats
        cpu_stats = aggregate_stats(self.df_cpu, "cpu_calls_mean", "std_cpu_calls", samples_per_row=100)
        sv_stats = aggregate_stats_with_n(self.df_qpu_sv, "qpu_calls_mean", "std_qpu_calls", "n_runs")
        mps_stats = aggregate_stats_conv(self.df_qpu_mps, "qpu_calls_mean", "std_qpu_calls", samples_per_row=1000)

        # 2. Apply convergence correction
        corrected_cpu = apply_convergence_correction(cpu_stats, sa_success, method='SA')
        corrected_sv = apply_convergence_correction(sv_stats, None, method='SV')
        corrected_mps = apply_convergence_correction(mps_stats, mps_success, method='MPS')

        # 3. Normalize corrected values
        corrected_cpu["corrected_mean"] /= 1e9
        corrected_cpu["corrected_std"] /= 1e9
        corrected_sv["corrected_mean"] /= 1e3
        corrected_sv["corrected_std"] /= 1e3
        corrected_mps["corrected_mean"] /= 1e4
        corrected_mps["corrected_std"] /= 1e4

        min_q = min(corrected_cpu["num_qubits"].min(), corrected_sv["num_qubits"].min())
        num_qubits_smooth = np.linspace(min_q, 160, 160)

        fig, ax = plt.subplots(figsize=(11, 10))

        # 4. Plot corrected and normalized curves
        self.fit_and_plot(corrected_cpu, "num_qubits", "corrected_mean", "corrected_std",
                        r"SA (/ 10^9 Hz)", "darkorange", "--", "darkorange", num_qubits_smooth, 18, ax, marker_size=32)
        self.fit_and_plot(corrected_sv, "num_qubits", "corrected_mean", "corrected_std",
                        r"QAOA Statevector (/ 10^3 Hz)", "royalblue", "-", "royalblue", num_qubits_smooth, 15, ax, marker_size=32)
        self.fit_and_plot(corrected_mps, "num_qubits", "corrected_mean", "corrected_std",
                        r"QAOA MPS (/ 10^4 Hz)", "mediumseagreen", "-", "mediumseagreen", num_qubits_smooth, 12, ax, marker_size=32)

        # Error bands for MPS
        mps_fit = self.fits[2]
        A, B = mps_fit['A'], mps_fit['B']
        B_err = mps_fit.get('B_err', 0.0)
        mps_lower = A * np.exp((B - B_err) * num_qubits_smooth)
        mps_upper = A * np.exp((B + B_err) * num_qubits_smooth)
        ax.fill_between(num_qubits_smooth, mps_lower, mps_upper, color='mediumseagreen', alpha=0.1, label='MPS ±1σ')

        # Error bands for SA
        sa_fit = self.fits[0]
        A_sa, B_sa = sa_fit['A'], sa_fit['B']
        B_sa_err = sa_fit.get('B_err', 0.0)
        sa_lower = A_sa * np.exp((B_sa - B_sa_err) * num_qubits_smooth)
        sa_upper = A_sa * np.exp((B_sa + B_sa_err) * num_qubits_smooth)
        ax.fill_between(num_qubits_smooth, sa_lower, sa_upper, color='darkorange', alpha=0.1, label='SA ±1σ')

        # Plot config
        ax.set_yscale("log")
        ax.set_xlabel("Number of Qubit (M)", fontsize=24)
        ax.set_ylabel("Estimated Runtime (log scale)", fontsize=24)
        ax.tick_params(labelsize=20)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=20, loc='lower right')

        # --- CROSSOVER POINTS ---
        def compute_crossover_point(A1, B1, A2, B2):
            if B1 == B2:
                return None, None
            x_cross = np.log(A1 / A2) / (B2 - B1)
            y_cross = A1 * np.exp(B1 * x_cross)
            return x_cross, y_cross

        sv_fit = self.fits[1]
        A_sv, B_sv = sv_fit["A"], sv_fit["B"]
        A_mps, B_mps, B_mps_err = mps_fit["A"], mps_fit["B"], mps_fit.get("B_err", 0.0)
        B_sa_minus, B_sa_plus = B_sa - B_sa_err, B_sa + B_sa_err
        B_mps_plus = B_mps + B_mps_err

        crossovers = {
            "SA vs SV central": compute_crossover_point(A_sa, B_sa, A_sv, B_sv),
            "SA vs MPS central": compute_crossover_point(A_sa, B_sa, A_mps, B_mps),
            "SA lower vs MPS upper": compute_crossover_point(A_sa, B_sa_minus, A_mps, B_mps_plus),
            "SA upper vs MPS lower": compute_crossover_point(A_sa, B_sa_plus, A_mps, B_mps),
        }

        print("\n📈 Central Fit Crossovers")
        if crossovers["SA vs SV central"][0]:
            print(f"- SA vs SV: x = {crossovers['SA vs SV central'][0]:.2f} qubits, "
                f"runtime ≈ {crossovers['SA vs SV central'][1]:.2e} s")
        if crossovers["SA vs MPS central"][0]:
            print(f"- SA vs MPS: x = {crossovers['SA vs MPS central'][0]:.2f} qubits, "
                f"runtime ≈ {crossovers['SA vs MPS central'][1]:.2e} s")

        print("\n📉 Error Band Crossovers")
        if crossovers["SA lower vs MPS upper"][0]:
            print(f"- SA lower vs MPS upper: x = {crossovers['SA lower vs MPS upper'][0]:.2f} qubits")
        if crossovers["SA upper vs MPS lower"][0]:
            print(f"- SA upper vs MPS lower: x = {crossovers['SA upper vs MPS lower'][0]:.2f} qubits")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, "cpu_vs_qpu_normalized_corrected.pdf"))

        plt.show()
        plt.close()


