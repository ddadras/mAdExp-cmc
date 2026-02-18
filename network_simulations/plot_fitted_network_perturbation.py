import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

results_path = os.path.join(os.path.dirname(__file__), 'fitted_network_disruption_mem_optimized.pkl')


if not os.path.exists(results_path):
    print(f"Error: Results file not found at {results_path}")
    print("Please run perturbation_network.py first to generate results.")
    exit(1)

print("Loading results...")
with open(results_path, 'rb') as f:
    data = pickle.load(f)

results = data['results']
multipliers = data['multipliers']
pop_names = data['pop_names']
params = data['params_perturbed']

param_labels = {
    'energy_factor': r'$d_{\epsilon}$',
    'pump_k': r'$\alpha_P$',
    'ATP_k': r'$\alpha_{ATP}$',
    'gamma': r'$\gamma$',
    'g_NMDA': r'$g_{NMDA}$',
    'g_GABA': r'$g_{GABA}$',
}

metrics = ['rates', 'energy', 'cvs', 'entropy', 'kuramoto', 'mutual_info']
metric_labels = ['Firing Rate', r'$\int \epsilon$', 'CV', 'SE', 'Kuramoto', 'MI']

print(f"Loaded results for {len(params)} parameters: {params}")
print(f"Multipliers: {multipliers}")
print(f"Populations: {pop_names}")


# ============================================================================
# PLOT 1: Per-Population Line Plots (4x8 Grid)
# ============================================================================
def plot_per_population():
    print("\nGenerating Plot 1: Per-Population Line Plots...")

    fig, axes = plt.subplots(4, 4, figsize=(7.5, 8.75))
    
    for i, param in enumerate(params):
        for j, pop in enumerate(pop_names[:4]):
            ax = axes[i, j]
            
            for k, metric in enumerate(metrics):
                values = results[param][pop][metric]
                baseline_idx = multipliers.index(1.0)
                baseline = values[baseline_idx]
                if baseline != 0:
                    normalized = [v / baseline for v in values]
                else:
                    normalized = values
                
                ax.plot(multipliers, normalized, 'o-', label=metric_labels[k], markersize=3)
            
            ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            
            if i == 0:
                ax.set_title(pop, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{param}\nNormalized', fontsize=9)
            if i == 3:
                ax.set_xlabel('Multiplier', fontsize=9)
            
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.99), fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_per_population_1.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()
    
    fig, axes = plt.subplots(4, 4, figsize=(7.5, 8.75))
    
    for i, param in enumerate(params):
        for j, pop in enumerate(pop_names[4:]):
            ax = axes[i, j]
            
            for k, metric in enumerate(metrics):
                values = results[param][pop][metric]
                baseline_idx = multipliers.index(1.0)
                baseline = values[baseline_idx]
                if baseline != 0:
                    normalized = [v / baseline for v in values]
                else:
                    normalized = values
                
                ax.plot(multipliers, normalized, 'o-', label=metric_labels[k], markersize=3)
            
            ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axvline(1.0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            
            if i == 0:
                ax.set_title(pop, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{param}\nNormalized', fontsize=9)
            if i == 3:
                ax.set_xlabel('Multiplier', fontsize=9)
            
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.99), fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_per_population_2.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 2: Heatmap Visualization (Parameter × Population)
# ============================================================================
def plot_heatmaps():
    print("\nGenerating Plot 2: Heatmap Visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 8.0))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        matrix = np.zeros((len(params), len(pop_names)))
        
        baseline_idx = multipliers.index(1.0)
        perturb_idx = multipliers.index(1.2)
        
        for i, param in enumerate(params):
            for j, pop in enumerate(pop_names):
                baseline_val = results[param][pop][metric][baseline_idx]
                perturb_val = results[param][pop][metric][perturb_idx]
                
                if baseline_val != 0:
                    pct_change = ((perturb_val - baseline_val) / baseline_val) * 100
                else:
                    pct_change = 0
                
                matrix[i, j] = pct_change
        
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
        
        ax.set_xticks(range(len(pop_names)))
        ax.set_xticklabels(pop_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=8)
        ax.set_title(f'{metric_labels[idx]}\n(% change at 1.2x)', fontsize=9)
        
        for i in range(len(params)):
            for j in range(len(pop_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=7)
        
        plt.colorbar(im, ax=ax, label='% Change', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_heatmap.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 3: Network-Wide Summary Plot
# ============================================================================
def plot_network_summary():
    print("\nGenerating Plot 3: Network-Wide Summary...")
    
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), sharex=True, sharey=True)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for param_idx, param in enumerate(params):
            if metric == 'rates':
                key = 'total_rate'
            elif metric == 'cvs':
                key = 'mean_cv'
            elif metric == 'kuramoto':
                key = 'mean_kuramoto'
            else:
                key = f'mean_{metric}'
                
            values = results[param]['network'][key]
            
            baseline_idx = multipliers.index(1.0)
            baseline = values[baseline_idx]
            if baseline != 0:
                normalized = [v / baseline for v in values]
            else:
                normalized = values
            
            ax.plot(multipliers, normalized, 'o-', linewidth=1.5, markersize=6, 
                   label=param_labels[param], color=colors[param_idx])
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Perturbation', fontsize=9)
        ax.set_ylabel('Deviation', fontsize=9)
        ax.set_title(f'{metric_labels[idx]}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        axes[0].set_xlim([0.3, 2.0])
    axes[0].set_ylim([0.0, 1.6])
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, fontsize=8, 
           bbox_to_anchor=(0.5, 0.03), framealpha=0.9, borderaxespad=0)

    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_network_summary.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 4: E/I Balance Analysis
# ============================================================================
def plot_ei_balance():
    print("\nGenerating Plot 4: E/I Balance Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    
    exc_pops = ['L23e', 'L4e', 'L5e', 'L6e']
    inh_pops = ['L23i', 'L4i', 'L5i', 'L6i']
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        exc_rates = np.mean([results[param][pop]['rates'] for pop in exc_pops], axis=0)
        inh_rates = np.mean([results[param][pop]['rates'] for pop in inh_pops], axis=0)
        
        baseline_idx = multipliers.index(1.0)
        exc_norm = exc_rates / exc_rates[baseline_idx]
        inh_norm = inh_rates / inh_rates[baseline_idx]
        
        ax.plot(multipliers, exc_norm, 'o-', linewidth=1.5, markersize=6, 
                color='red', label='Excitatory')
        ax.plot(multipliers, inh_norm, 's-', linewidth=1.5, markersize=6, 
                color='blue', label='Inhibitory')
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Perturbation Multiplier', fontsize=10)
        ax.set_ylabel('Normalized Firing Rate', fontsize=10)
        ax.set_title(f'{param} Perturbation', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_ei_balance.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 5: Layer-Specific Analysis
# ============================================================================
def plot_layer_comparison():
    print("\nGenerating Plot 5: Layer-Specific Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    
    layers = {
        'L2/3': ['L23e', 'L23i'],
        'L4': ['L4e', 'L4i'],
        'L5': ['L5e', 'L5i'],
        'L6': ['L6e', 'L6i']
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        for layer_idx, (layer_name, pops) in enumerate(layers.items()):
            layer_rates = np.mean([results[param][pop]['rates'] for pop in pops], axis=0)
            
            baseline_idx = multipliers.index(1.0)
            layer_norm = layer_rates / layer_rates[baseline_idx]
            
            ax.plot(multipliers, layer_norm, 'o-', linewidth=1.5, markersize=6, 
                   label=layer_name, color=colors[layer_idx])
        
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Perturbation Multiplier', fontsize=10)
        ax.set_ylabel('Normalized Firing Rate', fontsize=10)
        ax.set_title(f'{param} Perturbation', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_layer_comparison.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOT 6: Detailed Sensitivity Heatmaps
# ============================================================================
def plot_detailed_heatmaps():
    print("\nGenerating Plot 6: Detailed Sensitivity Heatmaps (Combined)...")
    
    neuron_counts = {
        'L23e': 20683, 'L23i': 5834,
        'L4e': 21915, 'L4i': 5479,
        'L5e': 4850, 'L5i': 1065,
        'L6e': 14395, 'L6i': 2948
    }
    
    exc_pops = ['L23e', 'L4e', 'L5e', 'L6e']
    inh_pops = ['L23i', 'L4i', 'L5i', 'L6i']
    
    vis_metric_labels = ['Firing Rate', r'$\int \epsilon$', 'CV', 'SE', 'Synchrony', 'MI']
    row_labels = pop_names + ['Exc (W. Mean)', 'Inh (W. Mean)', 'Network']
    n_rows = len(row_labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 8.75))
    axes = axes.flatten()
    
    for idx, param in enumerate(params):
        ax = axes[idx]
        
        sensitivity_matrix = np.zeros((n_rows, len(metrics)))
        
        baseline_idx = multipliers.index(1.0)
        perturb_idxs = [i for i in range(len(multipliers)) if i != baseline_idx]
        
        pop_sensitivities = np.zeros((len(pop_names), len(metrics)))
        
        for j, pop in enumerate(pop_names):
            for k, metric in enumerate(metrics):
                values = results[param][pop][metric]
                base = values[baseline_idx]
                
                sens = 0
                if base != 0:
                    for p_idx in perturb_idxs:
                        sens += abs(values[p_idx] - base) / abs(base)
                    sens /= len(perturb_idxs)
                else:
                    sens = 0 
                
                pop_sensitivities[j, k] = sens

        sensitivity_matrix[:8, :] = pop_sensitivities
        
        exc_indices = [pop_names.index(p) for p in exc_pops]
        exc_weights = np.array([neuron_counts[p] for p in exc_pops])
        exc_vals = pop_sensitivities[exc_indices, :]
        sensitivity_matrix[8, :] = np.average(exc_vals, axis=0, weights=exc_weights)
        
        inh_indices = [pop_names.index(p) for p in inh_pops]
        inh_weights = np.array([neuron_counts[p] for p in inh_pops])
        inh_vals = pop_sensitivities[inh_indices, :]
        sensitivity_matrix[9, :] = np.average(inh_vals, axis=0, weights=inh_weights)
        
        for k, metric in enumerate(metrics):
            if metric == 'rates': key = 'total_rate'
            elif metric == 'cvs': key = 'mean_cv'
            elif metric == 'kuramoto': key = 'mean_kuramoto'
            else: key = f'mean_{metric}'
            
            values = results[param]['network'][key]
            base = values[baseline_idx]
            
            sens = 0
            if base != 0:
                for p_idx in perturb_idxs:
                    sens += abs(values[p_idx] - base) / abs(base)
                sens /= len(perturb_idxs)
            
            sensitivity_matrix[10, k] = sens
        
        vmax = np.max(sensitivity_matrix) if np.max(sensitivity_matrix) > 0 else 1.0
        im = ax.imshow(sensitivity_matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=vmax)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(vis_metric_labels, fontsize=7, rotation=30, ha='right')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        
        for i, label in enumerate(ax.get_yticklabels()):
            if i >= 8:
                label.set_fontweight('bold')
        
        ax.set_title(f'{param} Sensitivity', fontsize=9, fontweight='bold')
        
        for i in range(n_rows):
            for j in range(len(metrics)):
                val = sensitivity_matrix[i, j]
                txt = f'{val:.2f}' if val >= 0.005 else '0'
                color = 'white' if val > vmax/2 else 'black'
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=5)
        
        ax.axhline(7.5, color='black', linewidth=1.0)
        ax.axhline(9.5, color='black', linewidth=1.0)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Parameter Sensitivity (Mean Relative Change)", fontsize=11, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    for fmt in ['png', 'tif']:
        output_path = os.path.join(os.path.dirname(__file__), f'figures\perturbation_sensitivity_combined.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Network Perturbation Visualization Suite")
    print("="*70)
    
    # Generate all plots
    plot_per_population()
    plot_heatmaps()
    plot_network_summary()
    plot_ei_balance()
    plot_layer_comparison()
    plot_detailed_heatmaps()
    
    print("\n" + "="*70)
    print("All plots generated successfully!")