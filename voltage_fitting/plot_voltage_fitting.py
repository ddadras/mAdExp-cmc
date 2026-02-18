import matplotlib.pyplot as plt
from brian2.monitors.spikemonitor import SpikeMonitor
import numpy as np
import matplotlib.pyplot as pltfrom
from brian2 import *
from fit_module import load_and_downsample
import pickle 
import os

plt.rcParams.update({'font.size': 8})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Robust path handling
base_dir = os.path.dirname(os.path.abspath(__file__))

stim_file_layer_3 = os.path.join(base_dir, "data/stimulus_Layer 3.txt")
resp_file_layer_3 = os.path.join(base_dir, "data/response_Layer 3.txt")

stim_file_layer_4 = os.path.join(base_dir, "data/stimulus_Layer 4.txt")
resp_file_layer_4 = os.path.join(base_dir, "data/response_Layer 4.txt")

stim_file_layer_5 = os.path.join(base_dir, "data/stimulus_Layer 5.txt")
resp_file_layer_5 = os.path.join(base_dir, "data/response_Layer 5.txt")

stim_file_layer_6 = os.path.join(base_dir, "data/stimulus_Layer 6.txt")
resp_file_layer_6 = os.path.join(base_dir, "data/response_Layer 6.txt")

# Load data again for comparison
layer_labels = ["layer_3", "layer_4", "layer_5", "layer_6"]
stim_files = [stim_file_layer_3, stim_file_layer_4, stim_file_layer_5, stim_file_layer_6]
resp_files = [resp_file_layer_3, resp_file_layer_4, resp_file_layer_5, resp_file_layer_6]
tracing_data = {}
model_data = {}
print("Loading data...")
for layer in layer_labels:
    data = load_and_downsample(stim_files[layer_labels.index(layer)], resp_files[layer_labels.index(layer)], downsample_factor=10)
    tracing_data[layer] = {}
    tracing_data[layer]['t'] = data['t']
    tracing_data[layer]['stim'] = data['stim']
    tracing_data[layer]['resp'] = data['resp']
    tracing_data[layer]['dt'] = data['dt']
    
    pkl_path = os.path.join(base_dir, f'model_params/good_enough/{layer}.pkl')
    with open(pkl_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    model_data[layer] = loaded_dict
print("Data loaded.")

print("Starting simulations...")

voltages = {}

for layer in layer_labels:
    print(f"Simulating {layer}...")
    
    stim_waveform = TimedArray(tracing_data[layer]['stim'], dt=tracing_data[layer]['dt'])

    model_data[layer]['I_stim'] = stim_waveform

    eqs = """
    E_L = E_0 * (1 - energy_factor * ((epsilon_0-epsilon)/(epsilon_0 - epsilon_c))) : volt
    dV/dt  = (g_L*(E_L-V) + g_L*Delta_T*(epsilon - epsilon_c)*exp((V-V_th)/Delta_T)/(epsilon_0) + I_stim(t) - w) / C_m : volt
    dw/dt   = (a*(V-E_L) - w + I_KATP*((epsilon_0-epsilon)/(epsilon_0 - epsilon_c))) / tau_w : amp
    depsilon/dt = (ATP - pump - w/gamma) / tau_e : 1
    ATP = ATP_k * (1 - epsilon / epsilon_0) : 1
    pump = pump_k * (V - E_0) * (1 / (1 + exp(-(V - E_0) / (1*mV)))) * 1 / (1 + exp(-10*(epsilon - epsilon_c))) : 1
    """

    neuron = NeuronGroup(1, model=eqs, 
                            threshold='V > V_th and epsilon > epsilon_c',
                            reset="V = V_reset; w += b; epsilon -= delta",
                            method='heun',
                            namespace = model_data[layer],
                            dt=tracing_data[layer]['dt'], refractory=2 * ms)
                            
    neuron.V = model_data[layer]['E_0']
    neuron.epsilon = model_data[layer]['epsilon_0']
    neuron.w = 0

    statemon = StateMonitor(neuron, ['V', 'epsilon', 'w'], record=0, dt=tracing_data[layer]['dt'])
    spikemon = SpikeMonitor(neuron)

    print("Running simulation...")
    run(20*second, report='text')

    pos = np.digitize(spikemon.t / ms, statemon.t / ms) - 1
    t_sim, v_sim, eps_sim, we_sim, pos = statemon.t/second, statemon.V[0]/volt, statemon.epsilon[0], statemon.w[0]/pA, pos

    v_sim[pos] = 40.*mV
    voltages[layer] = v_sim



fig2, axes2 = plt.subplots(4, 2, figsize=(15, 12), sharex=True)

for i, layer in enumerate(layer_labels):
    ax_real = axes2[i, 0]
    ax_sim = axes2[i, 1]
    
    t_real = tracing_data[layer]['t'] / second
    v_real = tracing_data[layer]['resp'] / mV
    v_sim = voltages[layer] * 1000.0
    
    n_points = min(len(t_real), len(v_sim))
    t_plot = t_real[:n_points]
    v_real_plot = v_real[:n_points]
    v_sim_plot = v_sim[:n_points]

    layer_name = layer.replace('_', ' ').title()

    ax_real.plot(t_plot, v_real_plot, color='black', alpha=0.8, linewidth=1)
    ax_real.set_ylabel(f"{layer_name}\nVoltage (mV)")
    
    ax_sim.plot(t_plot, v_sim_plot, color='#d62728', alpha=0.8, linewidth=1) # Tab:red
    
    v_min = min(v_real_plot.min(), v_sim_plot.min())
    v_max = max(v_real_plot.max(), v_sim_plot.max())
    margin = (v_max - v_min) * 0.1 if (v_max != v_min) else 10.0
    ylim = (v_min - margin, v_max + margin)
    
    ax_real.set_ylim(ylim)
    ax_sim.set_ylim(ylim)
    
    ax_real.spines['top'].set_visible(False)
    ax_real.spines['right'].set_visible(False)
    ax_sim.spines['top'].set_visible(False)
    ax_sim.spines['right'].set_visible(False)

    if i < 3:
        ax_real.xaxis.set_visible(False)
        ax_sim.xaxis.set_visible(False)

    if i == 3:
        ax_real.set_xlabel("Time (s)")
        ax_sim.set_xlabel("Time (s)")
        ax_real.set_xticks([0, 5, 10, 15, 20])
        ax_sim.set_xticks([0, 5, 10, 15, 20])


# Column titles
axes2[0, 0].set_title("Real Data", fontsize=12, fontweight='bold')
axes2[0, 1].set_title("Simulated Data", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

fig_path = os.path.join(base_dir, "fitted_traces")
plt.savefig(fig_path + ".png", dpi=300)
plt.savefig(fig_path + ".tif", dpi=300)
plt.show()
