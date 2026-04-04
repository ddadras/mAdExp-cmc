"""
Network Perturbation Analysis
"""

import sys
import os
import pickle
from brian2 import *
from brian2 import devices
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# %% Load Parameters
def load_layer_params(layer_num):
    param_path = os.path.join(os.path.dirname(__file__), 'layer_parameters', f'layer_{layer_num}.pkl')
    with open(param_path, 'rb') as f:
        return pickle.load(f)


# %% Information-theoretic functions (adapted from single cell version)
def compute_metrics(spike_times_trials, bin_size=2*ms, word_length=3, duration=None):
    """
    Compute Shannon Entropy, Entropy Rate, and Mutual Information.
    
    Definitions used:
    - Shannon Entropy (H_total): Entropy of the distribution of spike words across all trials.
    - Entropy Rate: H_total / (word_length * bin_size) [bits/second]
    - Mutual Information (MI): H_total - H_noise [bits per word]
      where H_noise is the average entropy of words across trials (noise entropy).
    """
    if len(spike_times_trials) == 0:
        return 0.0, 0.0, 0.0

    if duration is None:
        all_spikes = np.concatenate([st for st in spike_times_trials if len(st) > 0])
        if len(all_spikes) == 0:
            return 0.0, 0.0, 0.0
        duration = np.max(all_spikes)
    
    n_bins = int(duration / bin_size)
    n_trials = len(spike_times_trials)
    
    # Discretize all trials
    all_bins = []
    for spike_times in spike_times_trials:
        bins = np.zeros(n_bins, dtype=int)
        if len(spike_times) > 0:
            bin_indices = ((spike_times / bin_size) % n_bins).astype(int)
            # Clip indices just in case
            bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_bins)]
            bins[bin_indices] = 1
        all_bins.append(bins)
    
    # H_total: entropy of pooled responses (marginal entropy)
    pooled_words = []
    n_words_per_trial = n_bins - word_length + 1
    
    if n_words_per_trial <= 0:
        return 0.0, 0.0, 0.0

    for bins in all_bins:
        for i in range(n_words_per_trial):
            word = tuple(bins[i:i+word_length])
            pooled_words.append(word)
    
    word_counts = {}
    for word in pooled_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    total = len(pooled_words)
    H_total = 0.0
    for count in word_counts.values():
        p = count / total
        if p > 0:
            H_total -= p * np.log2(p)
            
    # H_noise: average entropy computed at each time bin across trials
    # Measures variability given the stimulus (trial-to-trial variability)
    H_noise = 0.0
    if n_trials > 1:
        for t in range(n_words_per_trial):
            word_counts_t = {}
            for trial_bins in all_bins:
                word = tuple(trial_bins[t:t+word_length])
                word_counts_t[word] = word_counts_t.get(word, 0) + 1
            
            H_t = 0.0
            for count in word_counts_t.values():
                p = count / n_trials
                if p > 0:
                    H_t -= p * np.log2(p)
            H_noise += H_t
        H_noise /= n_words_per_trial
    
    MI = max(0, H_total - H_noise)
    
    # Entropy Rate in bits/second
    entropy_rate = H_total / (word_length * float(bin_size))
    
    return H_total, entropy_rate, MI


def calculate_kuramoto(spike_monitor, t_start=None, t_end=None):
    """Calculates Kuramoto Order Parameter (dimensionless)."""
    
    t_start_val = t_start if t_start is not None else 0*second
    t_end_val = t_end if t_end is not None else spike_monitor.t[-1]
    
    # Time vector for analysis
    dt_analysis = 1.0*ms 
    times = np.arange(t_start_val/ms, t_end_val/ms, dt_analysis/ms) * ms
    n_steps = len(times)
    
    if n_steps == 0:
        return 0.0
    
    # Complex phase sum
    Z = np.zeros(n_steps, dtype=complex)
    
    # Get all spikes
    all_spikes_t = spike_monitor.t
    all_spikes_i = spike_monitor.i
    
    if len(all_spikes_t) == 0:
        return 0.0
        
    # Filter by time window
    mask = (all_spikes_t >= t_start_val) & (all_spikes_t <= t_end_val)
    spikes_t = all_spikes_t[mask]
    spikes_i = all_spikes_i[mask]
    
    if len(spikes_t) == 0:
        return 0.0
        
    # Group by neuron index
    
    unique_units = np.unique(spikes_i)
    valid_neurons = 0
    
    trains = {}
    
    sort_idx = np.argsort(spikes_i)
    sorted_i = spikes_i[sort_idx]
    sorted_t = spikes_t[sort_idx]
    
    split_idx = np.where(np.diff(sorted_i) > 0)[0] + 1
    unit_spikes = np.split(sorted_t, split_idx)
    
    unit_ids = sorted_i[np.concatenate(([0], split_idx))] if len(sorted_i) > 0 else []
    
    for _, spikes in zip(unit_ids, unit_spikes):
        if len(spikes) < 2:
            continue
            
        valid_neurons += 1
        
        sp_ms = spikes/ms
        t_eval_ms = times/ms
        
        idx = np.searchsorted(sp_ms, t_eval_ms)
        
        valid_mask = (idx > 0) & (idx < len(sp_ms))
        if not np.any(valid_mask):
            continue
            
        k_idx = idx[valid_mask] - 1
        k1_idx = idx[valid_mask]
        
        t_k = sp_ms[k_idx]
        t_k1 = sp_ms[k1_idx]
        t_now = t_eval_ms[valid_mask]
        
        phi = 2 * np.pi * (t_now - t_k) / (t_k1 - t_k)
        
        Z[valid_mask] += np.exp(1j * phi)
        
    if valid_neurons == 0:
        return 0.0
        
    Z /= valid_neurons
    R_t = np.abs(Z)
    
    return np.mean(R_t)


def run_simulation():
    """
    Run network perturbation analysis
    """
    
    # Load layer parameters
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
    devices.device.seed(1)
    defaultclock.dt = 0.1 * ms

    V_E = 0. * mV
    V_I = -70. * mV

    # Noise parameters
    tau = 1 * ms ** -1
    sigma = 0.01 * mV

    # Connectivity parameters
    N_scale = 0.01
    connectivity_scale = 1.0

    # Neuron Counts (Potjans & Diesmann 2014)
    full_counts = {
        'L23e': 20683, 'L23i': 5834,
        'L4e': 21915, 'L4i': 5479,
        'L5e': 4850, 'L5i': 1065,
        'L6e': 14395, 'L6i': 2948
    }

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
        neurons.g_GABA = 1.25 * nS * I_scaling
        neurons.g_AMPA_rec = 0.104 * nS * E_scaling
        neurons.g_AMPA_ext = 2.08 * nS * Ext_scaling_E * 0.25

        start_idx = end_idx

    print("Neurons created and parameters set.")

    # Extract populations
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
        pop.g_GABA = 0.973 * nS * I_scaling
        pop.g_AMPA_rec = 0.081 * nS * E_scaling
        pop.g_AMPA_ext = 1.62 * nS * Ext_scaling_I * 0.25

    # Store original parameters for resetting
    original_params = {
        'energy_factor': {name: populations[name].energy_factor[0] for name in pop_names},
        'ATP_k': {name: populations[name].ATP_k[0] for name in pop_names},
        'g_NMDA': {name: populations[name].g_NMDA[0] for name in pop_names},
        'g_GABA': {name: populations[name].g_GABA[0] for name in pop_names},
    }

    print("Creating synapses...")

    # Synaptic equations
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

    # Create synapses (simplified for perturbation - only key connections)
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

    print("Synapses created.")

    # %% Poisson Input
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

    # %% Perturbation Configuration
    params_to_perturb = ['energy_factor', 'ATP_k', 'g_NMDA', 'g_GABA']
    multipliers = [0.01, 0.3, 0.5, 0.8, 1.00, 1.2, 1.5, 1.8, 2.0]  # Perturbation multipliers
    
    simulation_duration = 1.1 * second  # Shorter for network (computationally expensive)
    n_trials = 3  # Reduced number of trials for network
    
    results = {}
    
    # Build network
    net = Network(collect())
    net.store('initial_state')
    
    print(f"Starting perturbation analysis...")
    print(f"Parameters: {params_to_perturb}")
    print(f"Multipliers: {multipliers}")
    print(f"Trials per condition: {n_trials}")
    
    _pbar = tqdm(total=len(params_to_perturb)*len(multipliers)*n_trials)
    for param in params_to_perturb:
        
        print(f"\nPerturbing {param}...")
        
        res_param = {
            pop_name: {
                'rates': [], 'energy': [], 'cvs': [],
                'entropy': [], 'entropy_rate': [], 'mutual_info': [], 'kuramoto': []
            } for pop_name in pop_names
        }
        
        # Also store network-wide metrics
        res_param['network'] = {
            'total_rate': [], 'mean_energy': [], 'mean_cv': [],
            'mean_entropy': [], 'mean_entropy_rate': [], 'mean_mutual_info': [], 'mean_kuramoto': []
        }
        
        for mult in multipliers:
            print(f"  Multiplier: {mult:.2f}")
            
            # Storage for trials
            trial_data = {pop_name: {
                'spike_times': [], 'rates': [], 'cvs': [], 'energies': [], 'kuramoto': []
            } for pop_name in pop_names}
            
            # Run trials
            for trial in range(n_trials):
                _pbar.update(1)
                print(f"    Trial {trial + 1}/{n_trials}")
                
                net.restore('initial_state')
                
                # Reset all parameters to original
                for p_name in params_to_perturb:
                    for pop_name in pop_names:
                        setattr(populations[pop_name], p_name, original_params[p_name][pop_name])
                
                # Apply perturbation
                for pop_name in pop_names:
                    current_val = original_params[param][pop_name] * mult
                    setattr(populations[pop_name], param, current_val)
                
                # Set seed for reproducibility
                seed(1 + trial)
                
                # Create monitors
                spike_monitors = {name: SpikeMonitor(populations[name]) for name in pop_names}
                state_monitors = {name: StateMonitor(populations[name], ['epsilon'], record=True, dt=1*ms) 
                                for name in pop_names}
                
                # Add monitors to network
                for mon in spike_monitors.values():
                    net.add(mon)
                for mon in state_monitors.values():
                    net.add(mon)
                
                # Run simulation
                net.run(simulation_duration, report=None)
                
                # Analysis definitions
                analysis_start = 0.1 * second
                analysis_duration = simulation_duration - analysis_start
                
                # Collect metrics per population
                for pop_name in pop_names:
                    spikes = spike_monitors[pop_name]
                    states = state_monitors[pop_name]
                    
                    # Spike times
                    raw_st = spikes.t
                    mask = raw_st >= analysis_start
                    st = np.array((raw_st[mask] - analysis_start) / second)
                    trial_data[pop_name]['spike_times'].append(st)
                    
                    # Firing rate
                    rate = len(st) / analysis_duration / num_neurons[pop_name]
                    trial_data[pop_name]['rates'].append(rate)
                    
                    # CV
                    if len(st) > 2:
                        isi = np.diff(st)
                        cv = np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else 0.0
                    else:
                        cv = 0.0
                    trial_data[pop_name]['cvs'].append(cv)
                    
                    # Energy (mean epsilon) - filtering
                    if len(states.t) > 0:
                        state_mask = states.t >= analysis_start
                        avg_energy = np.mean(states.epsilon[:, state_mask])
                    else:
                        avg_energy = 0.0
                    trial_data[pop_name]['energies'].append(avg_energy)
                    
                    # Kuramoto (Synchrony)
                    k_score = calculate_kuramoto(spikes, t_start=analysis_start, t_end=simulation_duration)
                    trial_data[pop_name]['kuramoto'].append(k_score)
                
                # Remove monitors
                for mon in spike_monitors.values():
                    net.remove(mon)
                for mon in state_monitors.values():
                    net.remove(mon)
            
            # Aggregate across trials for each population
            for pop_name in pop_names:
                res_param[pop_name]['rates'].append(np.mean(trial_data[pop_name]['rates']))
                res_param[pop_name]['energy'].append(np.mean(trial_data[pop_name]['energies']))
                res_param[pop_name]['cvs'].append(np.mean(trial_data[pop_name]['cvs']))
                res_param[pop_name]['kuramoto'].append(np.mean(trial_data[pop_name]['kuramoto']))
                
                # Compute info-theoretic metrics
                h_total, h_rate, mi = compute_metrics(
                    trial_data[pop_name]['spike_times'],
                    bin_size=float(2*ms),
                    word_length=3,
                    duration=float(analysis_duration/second)
                )
                
                res_param[pop_name]['entropy'].append(h_total)
                res_param[pop_name]['entropy_rate'].append(h_rate)
                res_param[pop_name]['mutual_info'].append(mi)
            
            # Network-wide aggregates
            res_param['network']['total_rate'].append(
                np.sum([np.mean(trial_data[pop_name]['rates']) * num_neurons[pop_name] 
                       for pop_name in pop_names])
            )
            res_param['network']['mean_energy'].append(
                np.mean([np.mean(trial_data[pop_name]['energies']) for pop_name in pop_names])
            )
            res_param['network']['mean_cv'].append(
                np.mean([np.mean(trial_data[pop_name]['cvs']) for pop_name in pop_names])
            )
            res_param['network']['mean_entropy'].append(
                np.mean([res_param[pop_name]['entropy'][-1] for pop_name in pop_names])
            )
            res_param['network']['mean_entropy_rate'].append(
                np.mean([res_param[pop_name]['entropy_rate'][-1] for pop_name in pop_names])
            )
            res_param['network']['mean_mutual_info'].append(
                np.mean([res_param[pop_name]['mutual_info'][-1] for pop_name in pop_names])
            )
            res_param['network']['mean_kuramoto'].append(
                np.mean([res_param[pop_name]['kuramoto'][-1] for pop_name in pop_names])
            )
        
        results[param] = res_param
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'network_perturbation_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'multipliers': multipliers,
            'pop_names': pop_names,
            'params_perturbed': params_to_perturb
        }, f)
    
    print(f"\nResults saved to {output_path}")
    return results, multipliers


if __name__ == "__main__":
    run_simulation()
