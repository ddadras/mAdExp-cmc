import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import os

"""
Plotting Dashboard for Fitted Network Simulations.
"""

# Set plotting aesthetics
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_data(filename):
    """Load the results dictionary from pickle file."""
    print(f"Loading data from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_population_fano_factor(times, indices, N, t_start, t_end, bin_size=0.005, window_size=0.1):
    """
    Calculates the Fano Factor of the population spike count in sliding windows.
    Fano Factor = Var(Count) / Mean(Count).
    High Fano Factor -> Bursty/Synchronous.
    Low Fano Factor (~1) -> Poisson/Asynchronous.
    """

    print(f"Calculating Population Fano Factor... (N={N})")
    
    if len(times) == 0:
        return np.array([]), np.array([])
        
    # 1. Bin spikes
    bins = np.arange(t_start, t_end, bin_size)
    counts, _ = np.histogram(times, bins=bins)
    
    # 2. Sliding window statistics on counts
    window_bins = int(window_size / bin_size)
    if window_bins < 2: window_bins = 2
    
    conv_win = np.ones(window_bins) / window_bins
    
    # Rolling Mean
    mean_counts = np.convolve(counts, conv_win, mode='valid')
    
    # Rolling Var: E[X^2] - (E[X])^2
    mean_sq_counts = np.convolve(counts**2, conv_win, mode='valid')
    var_counts = mean_sq_counts - mean_counts**2
    
    # Fano Factor (Var/Mean)
    # Add epsilon to mean to avoid div by zero
    ff = var_counts / (mean_counts + 1e-9)
    
    # Time vector (centered)
    t_valid = bins[:-1][:len(ff)] + (window_size / 2)
    
    return t_valid, ff

def simple_lfp(data, layer='L23', include_inh=True, include_inh_pops=True):
    """
    Calculate LFP for a specific layer based on simple current summation.
    """
    t = data['t']
    pops = [f'{layer}e']
    if include_inh_pops:
        pops.append(f'{layer}i')
        
    lfp_out = np.zeros(len(t))
    
    for pop in pops:
        curr_exc = data['I_exc'][pop]
        if curr_exc.ndim == 2:
            lfp_out += np.sum(curr_exc, axis=0)
        else:
            lfp_out += curr_exc

        if include_inh:
            curr_inh = data['I_inh'][pop]
            if curr_inh.ndim == 2:
                lfp_out += np.sum(curr_inh, axis=0)
            else:
                lfp_out += curr_inh
            
    return t, lfp_out

def generate_dashboard(data_path, output_path):
    data = load_data(data_path)
    
    t = data['t']
    
    # ---------------------------------------------------------
    # 1. Compute Metrics
    # ---------------------------------------------------------
    
    # A. Synchronicity (Fano Factor)
    spikes = data['spikes']
    colors = {'L23': '#1f77b4', 'L4': '#2ca02c', 'L5': '#d62728', 'L6': '#9467bd'}
    exc_pops = ['L23e', 'L4e', 'L5e', 'L6e']
    fano_data = {}
    
    # Per-population Fano Factor
    for pop in exc_pops:
        t_f, ff = calculate_population_fano_factor(
            spikes[f'{pop}_times'], spikes[f'{pop}_indices'], 
            N=np.max(spikes[f'{pop}_indices'])+1 if len(spikes[f'{pop}_indices'])>0 else 1, 
            t_start=t[0], t_end=t[-1],
            bin_size=0.005, window_size=0.1
        )
        fano_data[pop] = (t_f, ff)

    # Global Fano Factor (combining all excitatory spikes)
    all_exc_times = []
    all_exc_indices = []
    offset = 0
    for pop in exc_pops:
        t_p = spikes[f'{pop}_times']
        i_p = spikes[f'{pop}_indices']
        if len(i_p) > 0:
            all_exc_times.append(t_p)
            all_exc_indices.append(i_p + offset)
            offset += np.max(i_p) + 1
    
    t_fano_glob, fano_factor_glob = np.array([]), np.array([])
    if len(all_exc_times) > 0:
        all_exc_times = np.concatenate(all_exc_times)
        all_exc_indices = np.concatenate(all_exc_indices)
        t_fano_glob, fano_factor_glob = calculate_population_fano_factor(
            all_exc_times, all_exc_indices, N=offset, t_start=t[0], t_end=t[-1],
            bin_size=0.005, window_size=0.1
        )
    
    # B. Aggregate LFP (Method 2: Exc+Inh, Excitatory Populations Only, Summed across layers)
    fs = 1 / (t[1] - t[0])
    sos = signal.butter(4, [1.5, 100], 'bandpass', fs=fs, output='sos')
    
    aggregate_lfp_raw = np.zeros(len(t))
    for layer in ['L23', 'L4', 'L5', 'L6']:
        _, lfp_layer = simple_lfp(data, layer=layer, include_inh=True, include_inh_pops=False)
        aggregate_lfp_raw += lfp_layer
        
    aggregate_lfp_filt = signal.sosfiltfilt(sos, aggregate_lfp_raw)
    
    # ---------------------------------------------------------
    # 2. Setup Plot
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1.5, 1, 1.5, 1], hspace=0.3)
    
    # Row 1: Synchronicity (Fano Factor)
    ax1 = plt.subplot(gs[0])
    
    if len(t_fano_glob) > 0:
        ax1.plot(t_fano_glob, fano_factor_glob, 'k-', lw=1.5, alpha=0.9, label='Global')
        
    for pop in exc_pops:
        t_f, ff = fano_data[pop]
        if len(t_f) > 0:
            layer = pop[:-1] # Remove 'e'
            ax1.plot(t_f, ff, color=colors[layer], lw=1.0, alpha=0.7, label=layer)
    
    ax1.set_ylabel('Fano Factor')
    ax1.set_title('A  Network Synchronicity (Fano Factor)', loc='left', fontweight='bold')
    
    ax1.grid(alpha=0.3)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim([0, 500])
    ax1.legend(loc='upper left', ncol=2, frameon=False, fontsize=8)
    
    # Row 2: Firing Rates (Grouped by Layer)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    for layer in ['L23', 'L4', 'L5', 'L6']:
        rate_e = data['rates'][f'{layer}e']
        rate_i = data['rates'][f'{layer}i']
        
        window = int(0.05 * fs)
        rate_e_smooth = np.convolve(rate_e, np.ones(window)/window, mode='same')
        
        ax2.plot(t, rate_e_smooth, label=layer, color=colors[layer], lw=1.5)

    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('B  Population Firing Rates', loc='left', fontweight='bold')
    ax2.legend(loc='lower right', ncol=1, frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Row 3: Energy Levels
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    for layer in ['L23', 'L4', 'L5', 'L6']:
        eps_e = data['epsilon'][f'{layer}e']
        ax3.plot(t, eps_e, color=colors[layer], label=layer)
        
    ax3.set_ylabel(r'Energy Level ($\epsilon$)')
    ax3.set_title('C  Energy Levels', loc='left', fontweight='bold')
    ax3.legend(loc='lower right', ncol=1, frameon=False, fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Row 4: Spike Raster
    ax4 = plt.subplot(gs[3], sharex=ax1)
    
    current_offset = 0
    pop_order = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
    raster_colors = ['k', 'gray'] * 4
    
    tick_locs = []
    tick_labels = []
    
    for i, pop in enumerate(pop_order):
        ts = spikes[f'{pop}_times']
        ids = spikes[f'{pop}_indices']
        
        if len(ids) == 0: continue
            
        max_id = np.max(ids) if len(ids) > 0 else 0
        
        if len(ids) > 10000:
             mask = np.random.choice(len(ids), 10000, replace=False)
             ts = ts[mask]
             ids = ids[mask]
        
        ax4.scatter(ts, ids + current_offset, s=0.5, color=raster_colors[i], alpha=0.7)
        
        tick_locs.append(current_offset + max_id/2)
        tick_labels.append(pop)
        
        current_offset += max_id + 100
        
    ax4.set_yticks(tick_locs)
    ax4.set_yticklabels(tick_labels, fontsize=8)
    ax4.set_ylabel('Neuron Index')
    ax4.set_title('D Spike Raster Plot', loc='left', fontweight='bold')
    
    # Row 5: Aggregate LFP
    ax5 = plt.subplot(gs[4], sharex=ax1)
    ax5.plot(t, aggregate_lfp_filt, color='black', lw=1.2, label='Global Sum (Method 2)')
        
    ax5.set_ylabel(r'$\mu$A')
    ax5.set_title('E Local Field Potential', loc='left', fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.grid(alpha=0.3)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    disruption_times = [3, 8, 13]
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for dt in disruption_times:
            ax.axvline(x=dt, color='gray', linestyle='--', alpha=0.5, lw=1.5, zorder=1)

        ax.axvspan(3, 8, color='#ffffcc', alpha=0.3, zorder=0, label='ATP Disruption' if ax == ax1 else None)
        ax.axvspan(8, 13, color='#ffcccc', alpha=0.3, zorder=0, label='Combined Disruption' if ax == ax1 else None)

            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    pkl_path = os.path.join(os.path.dirname(__file__), 'fitted_network_disruption.pkl')
    out_path = os.path.join(os.path.dirname(__file__), 'fitted_network_dashboard.png')
    
    if os.path.exists(pkl_path):
        generate_dashboard(pkl_path, out_path)
    else:
        print(f"Error: {pkl_path} not found.")


