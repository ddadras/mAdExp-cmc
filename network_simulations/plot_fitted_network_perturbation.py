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
# PLOT : Network-Wide Summary Plot
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
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Network Perturbation Visualization Suite")
    print("="*70)
    
    plot_network_summary()
    
    print("\n" + "="*70)
    print("All plots generated successfully!")