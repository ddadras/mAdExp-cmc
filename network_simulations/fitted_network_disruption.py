import sys
import os
import pickle
from brian2 import *
from brian2 import devices
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %% Load Parameters
def load_layer_params(layer_num):
    param_path = os.path.join(os.path.dirname(__file__), 'layer_parameters', f'layer_{layer_num}.pkl')
    with open(param_path, 'rb') as f:
        return pickle.load(f)


params_L3 = load_layer_params(3)
params_L4 = load_layer_params(4)
params_L5 = load_layer_params(5)
params_L6 = load_layer_params(6)

# Map pop names to params
layer_params_map = {
    'L23e': params_L3, 'L23i': params_L3,
    'L4e': params_L4, 'L4i': params_L4,
    'L5e': params_L5, 'L5i': params_L5,
    'L6e': params_L6, 'L6i': params_L6,
}

# %% Model parameters
devices.device.seed(1)  # for reproducibility
defaultclock.dt = 0.1 * ms

# Constant parameters (if not in pickle) or default values
V_E = 0. * mV
V_I = -70. * mV

# Noise parameters
tau = 1 * ms ** -1
sigma = 0.01 * mV

# Connectivity parameters
N_scale = 0.1
connectivity_scale = 1.0

# Neuron Counts (Potjans & Diesmann 2014)
full_counts = {
    'L23e': 20683, 'L23i': 5834,
    'L4e': 21915, 'L4i': 5479,
    'L5e': 4850, 'L5i': 1065,
    'L6e': 14395, 'L6i': 2948
}

# Layer depths (μm from pia, approximate for cortical column) [4]
layer_depths = {
    'L23e': 200, 'L23i': 200,  # L2/3: 0-500 μm
    'L4e': 500, 'L4i': 500,  # L4: 500-750 μm
    'L5e': 900, 'L5i': 900,  # L5: 750-1200 μm
    'L6e': 1400, 'L6i': 1400  # L6: 1200-1800 μm
}

dendrite_depths = {
    'L23e': 50,
    'L4e': 350,
    'L5e': 400,
    'L6e': 800
}

electrode_depth = 600
sigma_ext = 0.3 * siemens / meter  # Extracellular conductivity [3]

pop_names = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
num_neurons = {k: int(v * N_scale) for k, v in full_counts.items()}
total_neurons = int(np.sum(list(num_neurons.values())))

# Connectivity Probabilities (Potjans & Diesmann 2014)
conn_probs = np.array([
    # 23e    23i    4e     4i     5e     5i     6e     6i
    [0.101, 0.169, 0.044, 0.082, 0.032, 0.0, 0.008, 0.0],  # L23e
    [0.135, 0.137, 0.032, 0.052, 0.075, 0.0, 0.004, 0.0],  # L23i
    [0.008, 0.006, 0.050, 0.135, 0.007, 0.0, 0.045, 0.0],  # L4e
    [0.069, 0.003, 0.079, 0.160, 0.003, 0.0, 0.046, 0.0],  # L4i
    [0.100, 0.062, 0.050, 0.006, 0.083, 0.373, 0.020, 0.0],  # L5e
    [0.016, 0.007, 0.003, 0.001, 0.060, 0.316, 0.009, 0.0],  # L5i
    [0.030, 0.004, 0.020, 0.001, 0.020, 0.0, 0.024, 0.225],  # L6e
    [0.030, 0.002, 0.010, 0.001, 0.010, 0.0, 0.019, 0.144]  # L6i
]) * connectivity_scale

# AMPA (excitatory)
tau_AMPA = 2. * ms

# NMDA (excitatory)
tau_NMDA_rise = 2. * ms
tau_NMDA_decay = 100. * ms
alpha = 0.5 / ms
Mg2 = 1

# GABAergic (inhibitory)
tau_GABA = 10 * ms

# %% Model definition
# Parameters that vary per neuron (fitted) are declared as constants/params
eqs = '''
dV/dt  = (g_L*(E_L-V) + g_L*Delta_T*(epsilon - epsilon_c)*exp((V-V_th)/Delta_T)/epsilon_0 + I - I_syn -we) / C_m + xi*sigma*tau**0.5: volt 
I : amp

E_L = E_0 * (1 - energy_factor*((epsilon_0-epsilon)/(epsilon_0 - epsilon_c))) : volt
dwe/dt   = (a*(V-E_L) - we + I_KATP*epsilon_c/(epsilon_c + 2*epsilon)) / tau_we : amp
depsilon/dt = (ATP - pump - we/gamma) / tau_e: 1
ATP = ATP_k * (1 - epsilon / epsilon_0) : 1
pump = pump_k * (V - E_0) * (1 / (1 + exp(-(V - E_0) / (1*mV)))) * 1 / (1 + exp(-10*(epsilon - epsilon_c))) : 1

I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp
I_exc = I_AMPA_rec + I_NMDA_rec : amp
I_inh = I_GABA_rec : amp

I_AMPA_rec = g_AMPA_rec * (V - V_E) * s_AMPA : amp
ds_AMPA / dt = - s_AMPA / tau_AMPA : 1
g_AMPA_rec : siemens

I_AMPA_ext = g_AMPA_ext * (V - V_E) * s_AMPA_ext : amp
ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
g_AMPA_ext : siemens

I_NMDA_rec = g_NMDA * (V - V_E) / (1 + Mg2 * exp(-0.062 * V / mV) / 3.57) * s_NMDA_tot : amp
s_NMDA_tot : 1
g_NMDA : siemens

I_GABA_rec = g_GABA * (V - V_I) * s_GABA : amp
ds_GABA / dt = - s_GABA / tau_GABA : 1
g_GABA : siemens

# Fitted Parameters
g_L : siemens (constant)
C_m : farad (constant)
Delta_T : volt (constant)
epsilon_c : 1 (constant)
epsilon_0 : 1 (constant)
V_th : volt (constant)
E_0 : volt (constant)
energy_factor : 1 (constant)
a : siemens (constant)
tau_we : second (constant)
I_KATP : amp (constant)
gamma : amp (constant)
tau_e : second (constant)
pump_k : 1/volt (constant)
b : amp (constant)
delta : 1 (constant)
V_reset : volt (constant)
ATP_k : 1
'''

print("Creating Neurons...")

neurons = NeuronGroup(total_neurons, eqs, method='heun', threshold='V > V_th and epsilon > epsilon_c',
                      reset="V = V_reset; epsilon -= delta; we += b", refractory=2 * ms)

# Create Populations and Set Parameters
start_idx = 0
populations = {}

E_scaling = (800 / 61843 / N_scale)
I_scaling = (200 / 15326 / N_scale)
Ext_scaling_E = (800 / 2000)
Ext_scaling_I = (800 / 1850)

for name in pop_names:
    end_idx = start_idx + num_neurons[name]
    pop = neurons[start_idx:end_idx]
    populations[name] = pop

    # Set parameters from pickle
    p = layer_params_map[name]

    pop.g_L = p['g_L']
    pop.C_m = p['C_m']
    pop.Delta_T = p['Delta_T']
    pop.epsilon = p['epsilon_0']
    pop.epsilon_c = p['epsilon_c']
    pop.epsilon_0 = p['epsilon_0']
    pop.V_th = p['V_th']
    pop.E_0 = p['E_0']
    pop.energy_factor = p['energy_factor'] if p['energy_factor'] <= 0.15 else 0.15
    pop.a = p['a']
    pop.tau_we = p['tau_w']
    pop.I_KATP = p['I_KATP']
    pop.gamma = p['gamma']
    pop.tau_e = p['tau_e']
    pop.pump_k = p['pump_k'] if p['pump_k'] <= 1 / (60 * mV) else 1 / (60 * mV)
    pop.b = p['b']
    pop.delta = p['delta']
    pop.V_reset = p['V_reset'] if p['V_th'] - p['V_reset'] > 2*mV else p['V_th'] - 2*mV
    pop.ATP_k = 1
    pop.V = p['E_0']

    # Assign default (Excitatory) values to all first
    neurons.g_NMDA = 0.327 * nS * E_scaling
    neurons.g_GABA = 1.25 * nS * I_scaling * 1
    neurons.g_AMPA_rec = 0.104 * nS * E_scaling
    neurons.g_AMPA_ext = 2.08 * nS * Ext_scaling_E * 0.25

    start_idx = end_idx

print("Neurons created and parameters set.")

# Extract populations into variables for convenience (and namespace)
pop_L23e = populations['L23e']
pop_L23i = populations['L23i']
pop_L4e = populations['L4e']
pop_L4i = populations['L4i']
pop_L5e = populations['L5e']
pop_L5i = populations['L5i']
pop_L6e = populations['L6e']
pop_L6i = populations['L6i']

inh_pops = [pop_L23i, pop_L4i, pop_L5i, pop_L6i]

for pop in inh_pops:
    pop.g_NMDA = 0.258 * nS * E_scaling
    pop.g_GABA = 0.973 * nS * I_scaling * 1
    pop.g_AMPA_rec = 0.081 * nS * E_scaling
    pop.g_AMPA_ext = 1.62 * nS * Ext_scaling_I * 0.25

print(f"Populations variables assigned.")

eqs_glut = '''
s_NMDA_tot_post = w * s_NMDA : 1 (summed)
ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * rise * (1 - s_NMDA) : 1 (clock-driven)
drise / dt = - rise / tau_NMDA_rise : 1 (clock-driven)
w : 1
'''

eqs_pre_glut = '''
s_AMPA += w
rise += 1
'''

eqs_pre_gaba = '''
s_GABA += w
'''

print("Creating synapses...")

# Create individual Synapses for each connection (Brian2 namespace requirement)
# Naming convention: S_{source}_{target}

# L23e sources
S_L23e_L23e = Synapses(pop_L23e, pop_L23e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L23e.connect(p=conn_probs[0, 0])
S_L23e_L23e.w[:] = 1

S_L23e_L23i = Synapses(pop_L23e, pop_L23i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L23i.connect(p=conn_probs[1, 0])
S_L23e_L23i.w[:] = 1

S_L23e_L4e = Synapses(pop_L23e, pop_L4e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L4e.connect(p=conn_probs[2, 0])
S_L23e_L4e.w[:] = 1

S_L23e_L4i = Synapses(pop_L23e, pop_L4i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L4i.connect(p=conn_probs[3, 0])
S_L23e_L4i.w[:] = 1

S_L23e_L5e = Synapses(pop_L23e, pop_L5e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L5e.connect(p=conn_probs[4, 0])
S_L23e_L5e.w[:] = 1

S_L23e_L6e = Synapses(pop_L23e, pop_L6e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L23e_L6e.connect(p=conn_probs[6, 0])
S_L23e_L6e.w[:] = 1

print(f"L23e synapses created. {8 / 64 * 100}% finished.")

# L23i sources
S_L23i_L23e = Synapses(pop_L23i, pop_L23e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L23e.connect(p=conn_probs[0, 1])
S_L23i_L23e.w = 1

S_L23i_L23i = Synapses(pop_L23i, pop_L23i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L23i.connect(p=conn_probs[1, 1])
S_L23i_L23i.w = 1

S_L23i_L4e = Synapses(pop_L23i, pop_L4e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L4e.connect(p=conn_probs[2, 1])
S_L23i_L4e.w = 1

S_L23i_L4i = Synapses(pop_L23i, pop_L4i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L4i.connect(p=conn_probs[3, 1])
S_L23i_L4i.w = 1

S_L23i_L5e = Synapses(pop_L23i, pop_L5e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L5e.connect(p=conn_probs[4, 1])
S_L23i_L5e.w = 1

S_L23i_L6e = Synapses(pop_L23i, pop_L6e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L23i_L6e.connect(p=conn_probs[6, 1])
S_L23i_L6e.w = 1

print(f"L23i synapses created. {16 / 64 * 100}% finished.")

# L4e sources
S_L4e_L23e = Synapses(pop_L4e, pop_L23e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L23e.connect(p=conn_probs[0, 2])
S_L4e_L23e.w[:] = 1

S_L4e_L23i = Synapses(pop_L4e, pop_L23i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L23i.connect(p=conn_probs[1, 2])
S_L4e_L23i.w[:] = 1

S_L4e_L4e = Synapses(pop_L4e, pop_L4e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L4e.connect(p=conn_probs[2, 2])
S_L4e_L4e.w[:] = 1

S_L4e_L4i = Synapses(pop_L4e, pop_L4i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L4i.connect(p=conn_probs[3, 2])
S_L4e_L4i.w[:] = 1

S_L4e_L5e = Synapses(pop_L4e, pop_L5e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L5e.connect(p=conn_probs[4, 2])
S_L4e_L5e.w[:] = 1

S_L4e_L6e = Synapses(pop_L4e, pop_L6e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L4e_L6e.connect(p=conn_probs[6, 2])
S_L4e_L6e.w[:] = 1

print(f"L4e synapses created. {24 / 64 * 100}% finished.")

# L4i sources
S_L4i_L23e = Synapses(pop_L4i, pop_L23e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L23e.connect(p=conn_probs[0, 3])
S_L4i_L23e.w = 1

S_L4i_L23i = Synapses(pop_L4i, pop_L23i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L23i.connect(p=conn_probs[1, 3])
S_L4i_L23i.w = 1

S_L4i_L4e = Synapses(pop_L4i, pop_L4e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L4e.connect(p=conn_probs[2, 3])
S_L4i_L4e.w = 1

S_L4i_L4i = Synapses(pop_L4i, pop_L4i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L4i.connect(p=conn_probs[3, 3])
S_L4i_L4i.w = 1

S_L4i_L5e = Synapses(pop_L4i, pop_L5e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L5e.connect(p=conn_probs[4, 3])
S_L4i_L5e.w = 1

S_L4i_L6e = Synapses(pop_L4i, pop_L6e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L4i_L6e.connect(p=conn_probs[6, 3])
S_L4i_L6e.w = 1

print(f"L4i synapses created. {32 / 64 * 100}% finished.")

# L5e sources
S_L5e_L23e = Synapses(pop_L5e, pop_L23e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L23e.connect(p=conn_probs[0, 4])
S_L5e_L23e.w[:] = 1

S_L5e_L23i = Synapses(pop_L5e, pop_L23i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L23i.connect(p=conn_probs[1, 4])
S_L5e_L23i.w[:] = 1

S_L5e_L4e = Synapses(pop_L5e, pop_L4e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L4e.connect(p=conn_probs[2, 4])
S_L5e_L4e.w[:] = 1

S_L5e_L4i = Synapses(pop_L5e, pop_L4i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L4i.connect(p=conn_probs[3, 4])
S_L5e_L4i.w[:] = 1

S_L5e_L5e = Synapses(pop_L5e, pop_L5e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L5e.connect(p=conn_probs[4, 4])
S_L5e_L5e.w[:] = 1

S_L5e_L5i = Synapses(pop_L5e, pop_L5i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L5i.connect(p=conn_probs[5, 4])
S_L5e_L5i.w[:] = 1

S_L5e_L6e = Synapses(pop_L5e, pop_L6e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L5e_L6e.connect(p=conn_probs[6, 4])
S_L5e_L6e.w[:] = 1

print(f"L5e synapses created. {40 / 64 * 100}% finished.")

# L5i sources
S_L5i_L23e = Synapses(pop_L5i, pop_L23e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L23e.connect(p=conn_probs[0, 5])
S_L5i_L23e.w = 1

S_L5i_L23i = Synapses(pop_L5i, pop_L23i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L23i.connect(p=conn_probs[1, 5])
S_L5i_L23i.w = 1

S_L5i_L4e = Synapses(pop_L5i, pop_L4e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L4e.connect(p=conn_probs[2, 5])
S_L5i_L4e.w = 1

S_L5i_L4i = Synapses(pop_L5i, pop_L4i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L4i.connect(p=conn_probs[3, 5])
S_L5i_L4i.w = 1

S_L5i_L5e = Synapses(pop_L5i, pop_L5e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L5e.connect(p=conn_probs[4, 5])
S_L5i_L5e.w = 1

S_L5i_L5i = Synapses(pop_L5i, pop_L5i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L5i.connect(p=conn_probs[5, 5])
S_L5i_L5i.w = 1

S_L5i_L6e = Synapses(pop_L5i, pop_L6e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L5i_L6e.connect(p=conn_probs[6, 5])
S_L5i_L6e.w = 1

print(f"L5i synapses created. {48 / 64 * 100}% finished.")

# L6e sources
S_L6e_L23e = Synapses(pop_L6e, pop_L23e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L23e.connect(p=conn_probs[0, 6])
S_L6e_L23e.w[:] = 1

S_L6e_L23i = Synapses(pop_L6e, pop_L23i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L23i.connect(p=conn_probs[1, 6])
S_L6e_L23i.w[:] = 1

S_L6e_L4e = Synapses(pop_L6e, pop_L4e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L4e.connect(p=conn_probs[2, 6])
S_L6e_L4e.w[:] = 1

S_L6e_L4i = Synapses(pop_L6e, pop_L4i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L4i.connect(p=conn_probs[3, 6])
S_L6e_L4i.w[:] = 1

S_L6e_L5e = Synapses(pop_L6e, pop_L5e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L5e.connect(p=conn_probs[4, 6])
S_L6e_L5e.w[:] = 1

S_L6e_L6e = Synapses(pop_L6e, pop_L6e, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L6e.connect(p=conn_probs[6, 6])
S_L6e_L6e.w[:] = 1

S_L6e_L6i = Synapses(pop_L6e, pop_L6i, model=eqs_glut, on_pre=eqs_pre_glut, delay=2 * ms, method=euler)
S_L6e_L6i.connect(p=conn_probs[7, 6])
S_L6e_L6i.w[:] = 1

print(f"L6e synapses created. {56 / 64 * 100}% finished.")

# L6i sources
S_L6i_L23e = Synapses(pop_L6i, pop_L23e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L23e.connect(p=conn_probs[0, 7])
S_L6i_L23e.w = 1

S_L6i_L23i = Synapses(pop_L6i, pop_L23i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L23i.connect(p=conn_probs[1, 7])
S_L6i_L23i.w = 1

S_L6i_L4e = Synapses(pop_L6i, pop_L4e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L4e.connect(p=conn_probs[2, 7])
S_L6i_L4e.w = 1

S_L6i_L4i = Synapses(pop_L6i, pop_L4i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L4i.connect(p=conn_probs[3, 7])
S_L6i_L4i.w = 1

S_L6i_L5e = Synapses(pop_L6i, pop_L5e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L5e.connect(p=conn_probs[4, 7])
S_L6i_L5e.w = 1

S_L6i_L6e = Synapses(pop_L6i, pop_L6e, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L6e.connect(p=conn_probs[6, 7])
S_L6i_L6e.w = 1

S_L6i_L6i = Synapses(pop_L6i, pop_L6i, model="w:1", on_pre=eqs_pre_gaba, delay=2 * ms, method=euler)
S_L6i_L6i.connect(p=conn_probs[7, 7])
S_L6i_L6i.w = 1

print(f"L6i synapses created. {64 / 64 * 100}% finished.")

print("Synapses created.")


# %% Poisson Input / External Input

base_poisson = 3 * Hz

full_counts_poisson = {
    'L23e': 2000, 'L23i': 1850,
    'L4e': 2000, 'L4i': 1850,
    'L5e': 2000, 'L5i': 1850,
    'L6e': 2000, 'L6i': 1850
}

P_L23_e = PoissonInput(pop_L23e, 's_AMPA_ext', full_counts_poisson["L23e"], base_poisson, weight=1)
P_L23_i = PoissonInput(pop_L23i, 's_AMPA_ext', full_counts_poisson["L23i"], base_poisson, weight=1)
P_L4e = PoissonInput(pop_L4e, 's_AMPA_ext', full_counts_poisson["L4e"], base_poisson, weight=1)
P_L4i = PoissonInput(pop_L4i, 's_AMPA_ext', full_counts_poisson["L4i"], base_poisson, weight=1)
P_L5e = PoissonInput(pop_L5e, 's_AMPA_ext', full_counts_poisson["L5e"], base_poisson, weight=1)
P_L5_i = PoissonInput(pop_L5i, 's_AMPA_ext', full_counts_poisson["L5i"], base_poisson, weight=1)
P_L6e = PoissonInput(pop_L6e, 's_AMPA_ext', full_counts_poisson["L6e"], base_poisson, weight=1)
P_L6i = PoissonInput(pop_L6i, 's_AMPA_ext', full_counts_poisson["L6i"], base_poisson, weight=1)

if __name__ == "__main__":

    init_time = 1
    run(init_time * second, report='text')

    print("Creating monitors...")

    # Monitors
    spikes_L23e = SpikeMonitor(pop_L23e, record=True)
    spikes_L23i = SpikeMonitor(pop_L23i, record=True)
    spikes_L4e = SpikeMonitor(pop_L4e, record=True)
    spikes_L4i = SpikeMonitor(pop_L4i, record=True)
    spikes_L5e = SpikeMonitor(pop_L5e, record=True)
    spikes_L5i = SpikeMonitor(pop_L5i, record=True)
    spikes_L6e = SpikeMonitor(pop_L6e, record=True)
    spikes_L6i = SpikeMonitor(pop_L6i, record=True)

    n_rec = 50
    states_L23e = StateMonitor(pop_L23e, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L23e))))
    states_L23i = StateMonitor(pop_L23i, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L23i))))
    states_L4e = StateMonitor(pop_L4e, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L4e))))
    states_L4i = StateMonitor(pop_L4i, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L4i))))
    states_L5e = StateMonitor(pop_L5e, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L5e))))
    states_L5i = StateMonitor(pop_L5i, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L5i))))
    states_L6e = StateMonitor(pop_L6e, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L6e))))
    states_L6i = StateMonitor(pop_L6i, ['V', 'epsilon', 'I_exc', 'I_inh'], record=np.arange(min(n_rec, len(pop_L6i))))

    rate_L23e = PopulationRateMonitor(pop_L23e)
    rate_L23i = PopulationRateMonitor(pop_L23i)
    rate_L4e = PopulationRateMonitor(pop_L4e)
    rate_L4i = PopulationRateMonitor(pop_L4i)
    rate_L5e = PopulationRateMonitor(pop_L5e)
    rate_L5i = PopulationRateMonitor(pop_L5i)
    rate_L6e = PopulationRateMonitor(pop_L6e)
    rate_L6i = PopulationRateMonitor(pop_L6i)

    print("Monitors created.")

    # %% Simulate

    run_time = 2
    stim_time = 5

    disruption_factor = 0.3

    # Run
    run(run_time * second, report='text')
    pop_L23e.ATP_k *= disruption_factor
    pop_L23i.ATP_k *= disruption_factor
    pop_L4e.ATP_k *= disruption_factor
    pop_L4i.ATP_k *= disruption_factor
    pop_L5e.ATP_k *= 0.5
    pop_L5i.ATP_k *= 0.5
    pop_L6e.ATP_k *= 0.5
    pop_L6i.ATP_k *= 0.5
    run(stim_time * second, report='text')
    neurons.g_GABA *= 2
    run(stim_time * second, report='text')
    neurons.g_GABA /= 2
    pop_L23e.ATP_k /= disruption_factor
    pop_L23i.ATP_k /= disruption_factor
    pop_L4e.ATP_k /= disruption_factor
    pop_L4i.ATP_k /= disruption_factor
    pop_L5e.ATP_k /= 0.5
    pop_L5i.ATP_k /= 0.5
    pop_L6e.ATP_k /= 0.5
    pop_L6i.ATP_k /= 0.5
    run(run_time * second, report='text')

    # %% Save Data

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].plot(states_L23e.t / second, np.mean(states_L23e.V / mV, axis=0), color="black")
    axs[1, 0].plot(states_L23i.t / second, np.mean(states_L23i.V / mV, axis=0), color="red")
    axs[0, 1].plot(states_L4e.t / second, np.mean(states_L4e.V / mV, axis=0), color="green")
    axs[1, 1].plot(states_L4i.t / second, np.mean(states_L4i.V / mV, axis=0), color="yellow")
    axs[0, 2].plot(states_L5e.t / second, np.mean(states_L5e.V / mV, axis=0), color="blue")
    axs[1, 2].plot(states_L5i.t / second, np.mean(states_L5i.V / mV, axis=0), color="orange")
    axs[0, 3].plot(states_L6e.t / second, np.mean(states_L6e.V / mV, axis=0), color="purple")
    axs[1, 3].plot(states_L6i.t / second, np.mean(states_L6i.V / mV, axis=0), color="pink")
    plt.savefig('states.png')

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].plot(states_L23e.t / second, np.mean(states_L23e.epsilon, axis=0), color="black")
    axs[1, 0].plot(states_L23i.t / second, np.mean(states_L23i.epsilon, axis=0), color="red")
    axs[0, 1].plot(states_L4e.t / second, np.mean(states_L4e.epsilon, axis=0), color="green")
    axs[1, 1].plot(states_L4i.t / second, np.mean(states_L4i.epsilon, axis=0), color="yellow")
    axs[0, 2].plot(states_L5e.t / second, np.mean(states_L5e.epsilon, axis=0), color="blue")
    axs[1, 2].plot(states_L5i.t / second, np.mean(states_L5i.epsilon, axis=0), color="orange")
    axs[0, 3].plot(states_L6e.t / second, np.mean(states_L6e.epsilon, axis=0), color="purple")
    axs[1, 3].plot(states_L6i.t / second, np.mean(states_L6i.epsilon, axis=0), color="pink")
    plt.savefig('epsilon.png')

    fig, axs = plt.subplots(2, 4)
    axs[0, 0].plot(states_L23e.t / second, rate_L23e.rate / Hz, color="black")
    axs[1, 0].plot(states_L23i.t / second, rate_L23i.rate / Hz, color="red")
    axs[0, 1].plot(states_L4e.t / second, rate_L4e.rate / Hz, color="green")
    axs[1, 1].plot(states_L4i.t / second, rate_L4i.rate / Hz, color="yellow")
    axs[0, 2].plot(states_L5e.t / second, rate_L5e.rate / Hz, color="blue")
    axs[1, 2].plot(states_L5i.t / second, rate_L5i.rate / Hz, color="orange")
    axs[0, 3].plot(states_L6e.t / second, rate_L6e.rate / Hz, color="purple")
    axs[1, 3].plot(states_L6i.t / second, rate_L6i.rate / Hz, color="pink")
    plt.savefig('rates.png')

    spike_dict = {
        'L23e_times': spikes_L23e.t / second,
        'L23e_indices': np.array(spikes_L23e.i),
        'L23i_times': spikes_L23i.t / second,
        'L23i_indices': np.array(spikes_L23i.i),
        'L4e_times': spikes_L4e.t / second,
        'L4e_indices': np.array(spikes_L4e.i),
        'L4i_times': spikes_L4i.t / second,
        'L4i_indices': np.array(spikes_L4i.i),
        'L5e_times': spikes_L5e.t / second,
        'L5e_indices': np.array(spikes_L5e.i),
        'L5i_times': spikes_L5i.t / second,
        'L5i_indices': np.array(spikes_L5i.i),
        'L6e_times': spikes_L6e.t / second,
        'L6e_indices': np.array(spikes_L6e.i),
        'L6i_times': spikes_L6i.t / second,
        'L6i_indices': np.array(spikes_L6i.i),
    }

    rate_dict = {
        'L23e': rate_L23e.rate / Hz,
        'L23i': rate_L23i.rate / Hz,
        'L4e': rate_L4e.rate / Hz,
        'L4i': rate_L4i.rate / Hz,
        'L5e': rate_L5e.rate / Hz,
        'L5i': rate_L5i.rate / Hz,
        'L6e': rate_L6e.rate / Hz,
        'L6i': rate_L6i.rate / Hz,
    }

    def V_spiked(spike_monitor, state_monitor):
        v_mean = np.mean(state_monitor.V / mV, axis=0)
        
        spike_times = spike_monitor.t / ms
        spike_indices = spike_monitor.i
        # Times index in the state monitor
        times = state_monitor.t / ms
        
        if len(spike_times) > 0:
            num_neurons_pop = state_monitor.V.shape[0]
            pos = np.digitize(spike_times, times) - 1
            valid = (pos >= 0) & (pos < len(v_mean))
            pos = pos[valid]
            
            for p in np.unique(pos):
                v_mean[p] = max(v_mean[p], 20.0) # Artificial peak for visualization
                
        return v_mean

    v_dict = {
        'L23e': V_spiked(spikes_L23e, states_L23e),
        'L23i': V_spiked(spikes_L23i, states_L23i),
        'L4e': V_spiked(spikes_L4e, states_L4e),
        'L4i': V_spiked(spikes_L4i, states_L4i),
        'L5e': V_spiked(spikes_L5e, states_L5e),
        'L5i': V_spiked(spikes_L5i, states_L5i),
        'L6e': V_spiked(spikes_L6e, states_L6e),
        'L6i': V_spiked(spikes_L6i, states_L6i),
    }

    epsilon_dict = {
        'L23e': np.mean(states_L23e.epsilon, axis=0),
        'L23i': np.mean(states_L23i.epsilon, axis=0),
        'L4e': np.mean(states_L4e.epsilon, axis=0),
        'L4i': np.mean(states_L4i.epsilon, axis=0),
        'L5e': np.mean(states_L5e.epsilon, axis=0),
        'L5i': np.mean(states_L5i.epsilon, axis=0),
        'L6e': np.mean(states_L6e.epsilon, axis=0),
        'L6i': np.mean(states_L6i.epsilon, axis=0),
    }

    I_exc_dict = {
        'L23e': np.mean(states_L23e.I_exc / pA, axis=0),
        'L23i': np.mean(states_L23i.I_exc / pA, axis=0),
        'L4e': np.mean(states_L4e.I_exc / pA, axis=0),
        'L4i': np.mean(states_L4i.I_exc / pA, axis=0),
        'L5e': np.mean(states_L5e.I_exc / pA, axis=0),
        'L5i': np.mean(states_L5i.I_exc / pA, axis=0),
        'L6e': np.mean(states_L6e.I_exc / pA, axis=0),
        'L6i': np.mean(states_L6i.I_exc / pA, axis=0),
    }

    I_inh_dict = {
        'L23e': np.mean(states_L23e.I_inh / pA, axis=0),
        'L23i': np.mean(states_L23i.I_inh / pA, axis=0),
        'L4e': np.mean(states_L4e.I_inh / pA, axis=0),
        'L4i': np.mean(states_L4i.I_inh / pA, axis=0),
        'L5e': np.mean(states_L5e.I_inh / pA, axis=0),
        'L5i': np.mean(states_L5i.I_inh / pA, axis=0),
        'L6e': np.mean(states_L6e.I_inh / pA, axis=0),
        'L6i': np.mean(states_L6i.I_inh / pA, axis=0),
    }

    results_dict = {
        't': states_L23e.t / second,
        'spikes': spike_dict,
        'rates': rate_dict,
        'v': v_dict,
        'epsilon': epsilon_dict,
        'I_exc': I_exc_dict,
        'I_inh': I_inh_dict,
    }

    if True:
        with open('network_simulations/fitted_network_disruption.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
    