import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8

# Load the data
with open('network_simulations/fitted_network_performance.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data
t = data['t']
spikes = data['spikes']
rates = data['rates']
epsilon = data['epsilon']

layers = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
layer_labels = ['L2/3e', 'L2/3i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

colors = [
    '#E69F00',  # L23e - Orange
    '#56B4E9',  # L23i - Sky Blue
    '#009E73',  # L4e - Bluish Green
    '#F0E442',  # L4i - Yellow
    '#0072B2',  # L5e - Blue
    '#D55E00',  # L5i - Vermillion
    '#CC79A7',  # L6e - Reddish Purple
    '#999999',  # L6i - Gray
]

line_styles = ['-', '--', '-', '--', '-', '--', '-', '--']
line_widths = [1.5, 1.2, 1.5, 1.2, 1.5, 1.2, 1.5, 1.2]

fig = plt.figure(figsize=(7.5, 8.75), dpi=300)
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.08)

t_ms = t * 1000
x_min, x_max = t_ms[0], 2000

# ============================================================================
# 1. FIRING RATES
# ============================================================================
ax1 = fig.add_subplot(gs[0])

for i, layer in enumerate(layers):
    window = 10
    if len(rates[layer]) > window:
        smoothed_rate = np.convolve(rates[layer], np.ones(window)/window, mode='same')
    else:
        smoothed_rate = rates[layer]
    
    ax1.plot(t_ms, smoothed_rate, label=layer_labels[i], color=colors[i], 
             linestyle=line_styles[i], linewidth=line_widths[i], alpha=0.85)

ax1.set_ylabel('Firing Rate (Hz)', fontsize=11)
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim([x_min, x_max])

# ============================================================================
# 2. ENERGY LEVELS (EPSILON)
# ============================================================================
ax2 = fig.add_subplot(gs[1])

for i, layer in enumerate(layers):
    ax2.plot(t_ms, epsilon[layer], label=layer_labels[i], color=colors[i], 
             linestyle=line_styles[i], linewidth=line_widths[i], alpha=0.85)

ax2.set_ylabel(r'$\epsilon$', fontsize=11)
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim([x_min, x_max])

# ============================================================================
# 3. SPIKE RASTER PLOT
# ============================================================================
ax3 = fig.add_subplot(gs[2])

N_scale = 0.1
full_counts = {
    'L23e': 20683, 'L23i': 5834,
    'L4e': 21915, 'L4i': 5479,
    'L5e': 4850, 'L5i': 1065,
    'L6e': 14395, 'L6i': 2948
}
num_neurons = {k: int(v * N_scale) for k, v in full_counts.items()}

cumulative_neurons = {}
offset = 0
for layer in layers:
    cumulative_neurons[layer] = offset
    offset += num_neurons[layer]

for i, layer in enumerate(layers):
    spike_times_key = f'{layer}_times'
    spike_indices_key = f'{layer}_indices'
    
    if spike_times_key in spikes and spike_indices_key in spikes:
        spike_times = spikes[spike_times_key]
        spike_indices = spikes[spike_indices_key]
        
        if len(spike_times) > 0:
            y_positions = cumulative_neurons[layer] + spike_indices
            
            ax3.scatter(spike_times * 1000, y_positions, s=0.3, color=colors[i], 
                       alpha=0.5, rasterized=True, marker='|', linewidths=0.5)

for i, layer in enumerate(layers[:-1]):
    y_sep = cumulative_neurons[layers[i+1]]
    ax3.axhline(y=y_sep, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)

for i, layer in enumerate(layers):
    y_center = cumulative_neurons[layer] + num_neurons[layer] / 2
    ax3.text(x_max + (x_max - x_min) * 0.02, y_center, layer_labels[i], 
             verticalalignment='center', fontsize=9, color=colors[i], 
             fontweight='bold', ha='left')

ax3.set_xlabel('ms', fontsize=11)
ax3.set_xlim([x_min, x_max])
ax3.set_ylim([0, offset])
ax3.grid(True, alpha=0.25, axis='x', linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=8, fontsize=8, 
           bbox_to_anchor=(0.5, 0.03), framealpha=0.9, borderaxespad=0)

plt.savefig('fitted_network_performance.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig('fitted_network_performance.tif', dpi=300, bbox_inches='tight', format='tiff')
print("Plots saved as 'fitted_network_performance.png' and 'fitted_network_performance.tif'")
print(f"Figure size: 7.5 x 8.75 inches at 300 dpi")
print(f"Font: Arial, 8-12 point")