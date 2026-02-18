from brian2 import *
from scipy.optimize import minimize
from scipy.optimize import brute
import scipy
import os
import sys

prefs.codegen.target = 'numpy'


def load_and_downsample(stim_file, resp_file, original_dt=0.02 * ms, downsample_factor=5):
    """
    Loads stimulus and response data, and downsamples it.
    
    Args:
        stim_file (str): Path to stimulus text file.
        resp_file (str): Path to response text file.
        original_dt (Quantity): Original sampling interval (default 0.02*ms => 50kHz).
        downsample_factor (int): Factor to downsample by (5 => 10kHz effective).
        
    Returns:
        dict: {
            't': time array (Quantity),
            'stim': stimulus array (Quantity, pA),
            'resp': response array (Quantity, mV),
            'dt': new dt (Quantity)
        }
    """
    print(f"Loading data from {stim_file} and {resp_file}...")
    stim_raw = np.loadtxt(stim_file)
    resp_raw = np.loadtxt(resp_file)

    # Downsample
    stim = stim_raw[::downsample_factor]
    resp = resp_raw[::downsample_factor]

    new_dt = original_dt * downsample_factor
    duration = len(stim) * new_dt
    t = np.arange(len(stim)) * new_dt

    return {
        't': t,
        'stim': stim * pA,
        'resp': resp * mV,
        'dt': new_dt
    }


def extract_constraints(data):
    """
    Extracts constrained parameters from the data.
    
    Returns:
        dict: {
            'E_0': Quantity (mV),
            'C_m': Quantity (pF),
            'g_L_plus_a': Quantity (nS),
            'constraints': dict of fixed params
        }
    """
    stim = data['stim']
    resp = data['resp']
    t = data['t']
    dt = data['dt']

    # 1. E_0: Median resting value (where I=0)
    mask_rest = (stim == 0 * pA)
    if not np.any(mask_rest):
        print("Warning: No resting state (I=0) found. Using min current.")
        mask_rest = (stim == np.min(stim))

    E_0 = np.median(resp[mask_rest])
    print(f"Estimated E_0: {E_0}")

    # 3. g_L + a: Steady state V_ss vs I
    # And Tau_m from dynamics

    from scipy.optimize import curve_fit

    def exponential_func(t, V0, Vss, tau):
        return Vss + (V0 - Vss) * np.exp(-t / tau)

    unique_I = np.unique(stim)
    change_indices = np.where(np.abs(np.diff(stim)) > 1 * pA)[0]
    # Add start and end
    blocks = []
    last_idx = 0

    tau_estimates = []
    R_estimates = []
    idxs = np.concatenate(([0], change_indices + 1, [len(stim)]))

    I_val_list = []
    V_ss_list = []

    for i in range(len(idxs) - 1):
        start = idxs[i]
        end = idxs[i + 1]

        # Duration
        dur = (end - start) * dt
        if dur < 100 * ms: continue

        I_seg = stim[start]

        if i > 0:
            I_prev = stim[idxs[i - 1]]
            if abs(I_seg - I_prev) < 5 * pA: continue  # Noise


            fit_dur = min(dur, 200 * ms)
            n_fit = int(fit_dur / dt)

            t_seg = np.arange(n_fit) * float(dt)  # sec
            v_seg = resp[start: start + n_fit]

            v0_guess = v_seg[0]
            vss_guess = np.mean(resp[start + int(n_fit / 2): start + n_fit])

            try:
                popt, pcov = curve_fit(exponential_func, t_seg, v_seg,
                                       p0=[v0_guess, vss_guess, 0.02],
                                       bounds=([-np.inf, -np.inf, 0.001], [np.inf, np.inf, 0.2]),
                                       maxfev=1000)

                v0_fit, vss_fit, tau_fit = popt

                tau_estimates.append(tau_fit)

                I_val_list.append(I_seg)
                V_ss_list.append(vss_fit * volt)

            except Exception as e:
                pass

    # Estimates
    if tau_estimates:
        tau_m = np.median(tau_estimates) * second
        print(f"Estimated tau_m: {tau_m}")
    else:
        print("Warning: Could not fit tau_m. Using default.")
        tau_m = 20 * ms

    # g_L + a
    if not V_ss_list and len(stim) > 100:
        print("Fallback to mean Vss extraction")
        pass

    if len(I_val_list) > 1:
        Y = np.array([(v - E_0) / mV for v in V_ss_list])
        X = np.array([i / pA for i in I_val_list])

        res = np.polyfit(X, Y, 1)
        slope_mV_pA = res[0]
        R_eff = slope_mV_pA * Gohm
        g_L_plus_a = 1.0 / R_eff
    else:
        g_L_plus_a = 10 * nS

    print(f"Estimated g_L + a: {g_L_plus_a}")

    C_m = tau_m * g_L_plus_a
    print(f"Estimated C_m: {C_m.in_unit(pF)}")

    return {
        'E_0': E_0,
        'C_m': C_m,
        'g_L_plus_a': g_L_plus_a,
    }

def run_my_mAdExp_numpy(params, stimulus_arr, t_arr, dt_val):
    """
    Numpy implementation of my mAdExp for fast fitting.
    Returns voltage trace (mV).
    """
    C_m = params['C_m']
    g_L = params['g_L']
    E_0 = params['E_0']
    V_th = params['V_th']
    Delta_T = params['Delta_T']
    a = params['a']
    tau_w = params['tau_w']
    b = params['b']
    V_reset = params['V_reset']

    # Energy params
    epsilon_0 = 1
    epsilon_c = 0.15
    delta = params.get('delta', 0.02)
    gamma = params.get('gamma', 200.)  # pA
    tau_e = params.get('tau_e', 500.)  # ms
    I_KATP = params.get('I_KATP', 1.)  # pA
    ATP_k = params.get('ATP_k', 3)
    pump_k = params.get('pump_k', 1 / (60 * mV))  # 1/mV
    energy_factor = params.get('energy_factor', 0.1)

    stim = stimulus_arr  # pA
    dt = float(dt_val)  # ms

    N = len(t_arr)
    V = np.zeros(N)
    w = np.zeros(N)
    epsilon = np.zeros(N)

    V[0] = float(E_0)
    w[0] = 0
    epsilon[0] = epsilon_0

    # Ensure params are floats
    g_L = float(g_L)
    Delta_T = float(Delta_T)
    epsilon_c = float(epsilon_c)
    epsilon_0 = float(epsilon_0)
    V_th = float(V_th)
    E_0 = float(E_0)
    ATP_k = float(ATP_k)
    pump_k = float(pump_k)
    energy_factor = float(energy_factor)
    delta = float(delta)
    b = float(b)
    V_reset = float(V_reset)
    a = float(a)
    tau_w = float(tau_w)
    gamma = float(gamma)
    tau_e = float(tau_e)
    I_KATP = float(I_KATP)
    C_m = float(C_m)

    spikes = []
    time_since_last_spike = 100

    for i in range(N - 1):
        # Current state
        v_curr = V[i]
        w_curr = w[i]
        eps_curr = epsilon[i]
        I_app = stim[i]

        # dynamic EL
        el_dyn = E_0 * (1 - energy_factor * ((epsilon_0 - eps_curr) / (epsilon_0 - epsilon_c)))

        # dV/dt
        try:
            exp_term = np.exp((v_curr - V_th) / Delta_T)
        except FloatingPointError:
            exp_term = 1e9  # Clamp

        I_exp = g_L * Delta_T * (eps_curr - epsilon_c) * exp_term / epsilon_0

        dV = (g_L * (el_dyn - v_curr) + I_exp + I_app - w_curr) / C_m

        # dw/dt
        term_katp = I_KATP * ((epsilon_0 - eps_curr) / (epsilon_0 - epsilon_c))
        dw = (a * (v_curr - el_dyn) - w_curr + term_katp) / tau_w

        # depsilon/dt
        ATP = ATP_k * (1 - eps_curr / epsilon_0)
        pump = pump_k * (v_curr - E_0) * (1 / (1 + exp(-(v_curr - E_0)))) * 1 / (
                1 + exp(-10 * (eps_curr - epsilon_c)))
        deps = (ATP - pump - w_curr / gamma) / tau_e

        # Update
        v_next = v_curr + dV * dt
        w_next = w_curr + dw * dt
        eps_next = eps_curr + deps * dt

        # Spike check
        # threshold='V_m > V_th and epsilon > epsilon_c'

        if v_next > V_th and eps_next > epsilon_c and time_since_last_spike > 2.0:
            v_next = V_reset
            w_next = w_curr + b
            eps_next = eps_curr - delta
            spikes.append(i + 1)
            time_since_last_spike = 0

        V[i + 1] = v_next
        w[i + 1] = w_next
        epsilon[i + 1] = eps_next
        time_since_last_spike += dt
    return V, spikes


def fit_my_model(stim_file, resp_file, x0):

    if type(x0) is dict:
        temp_array = []
        # Order parameters
        parameter_order = ['Delta_T', 'b', 'tau_w', 'V_reset', 'gamma', 'tau_e', 'ATP_k', 'pump_k', 'delta', 'I_KATP', 'energy_factor', 'g_L', 'C_m', 'a', 'V_th']
        for key in parameter_order:
            temp_array.append(x0[key])
        x0 = temp_array

    # Load
    data = load_and_downsample(stim_file, resp_file, downsample_factor=5)

    # Constraints
    con = extract_constraints(data)

    # Unpack constraints (convert to standard units for optimizer)

    unit_V = mV
    unit_T = ms
    unit_C = pF
    unit_G = nS
    unit_I = pA

    # Helper to strip units
    def to_float(val, unit):
        if hasattr(val, 'dim'):  # Quantity
            return float(val / unit)
        return float(val)

    E_0_val = to_float(con['E_0'], unit_V)
    C_m_val = to_float(con['C_m'], unit_C)
    Total_G_val = to_float(con['g_L_plus_a'], unit_G)

    min_pts = 0
    max_pts = 200000
    stim_np = data['stim'][min_pts:max_pts] / unit_I
    resp_np = data['resp'][min_pts:max_pts] / unit_V
    clipped_resp_np = data['resp'][min_pts:max_pts] / unit_V
    t_np = data['t'][min_pts:max_pts] / unit_T
    dt_np = float(data['dt'] / unit_T)

    # Bounds
    bounds = [
        (0.1, 10.0),  # Delta_T
        (0.0, 200.0),  # b
        (10.0, 1000.0),  # tau_w
        (x0[3] * 1.1, x0[3] * 0.90),  # V_reset
        (10.0, 10000.0),  # gamma
        (5.0, 1000.0),  # tau_e
        (0.1, 10),  # ATP_k
        (0.001, 1),  # pump_k
        (0.001, 1),  # delta
        (0.01, 10.0),  # I_KATP
        (0.01, 1),  # energy_factor
        (0.1 * Total_G_val, Total_G_val * 3),  # g_L
        (0.25 * C_m_val, C_m_val * 3),  # C_m
        (0.1 * Total_G_val, Total_G_val * 3),  # a
        (x0[14] * 1.1, x0[14] * 0.90)  # V_th
    ]

    def objective(x):

        p = {
            'C_m': x[12],
            'E_0': E_0_val,
            'V_th': x[14],
            'Delta_T': x[0],
            'b': x[1],
            'tau_w': x[2],
            'V_reset': x[3],
            'gamma': x[4],
            'tau_e': x[5],
            'ATP_k': x[6],
            'pump_k': x[7],
            'delta': x[8],
            'I_KATP': x[9],
            'energy_factor': x[10],
            'g_L': x[11],
            'a': x[13]
        }

        v_model, spikes = run_my_mAdExp_numpy(p, stim_np, t_np, dt_np)

        spikes_exp_idx, _ = scipy.signal.find_peaks(resp_np, height=-0.02, distance=50)
        n_spikes_exp = len(spikes_exp_idx)
        n_spikes = len(spikes)

        if n_spikes > 1000:  # sanity check
            print("Too many spikes")
            return 1e9 + n_spikes

        count_err = (n_spikes - n_spikes_exp) ** 2

        clipped_resp_np[clipped_resp_np > p["V_th"]] = p["V_th"]

        mse = np.mean((v_model - clipped_resp_np) ** 2) + count_err * 10
        print(f"Current loss: {mse}, Spikes: {n_spikes}/{n_spikes_exp}")
        return mse

    print("Starting optimization...")
    res = minimize(objective, x0, bounds=bounds, method='Nelder-Mead', options={'maxiter': 1000, 'disp': True})

    print("Optimization done.")
    print(res)

    # Best params
    x = res.x
    best_params = {
        'C_m': x[12] * pF,
        'g_L': x[11] * nS,
        'E_0': E_0_val * mV,
        'a': x[13] * nS,
        'V_th': x[14] * mV,
        'Delta_T': x[0] * mV,
        'b': x[1] * pA,
        'tau_w': x[2] * ms,
        'V_reset': x[3] * mV,
        'gamma': x[4] * pA,
        'tau_e': x[5] * ms,
        'ATP_k': x[6],
        'pump_k': x[7] / mV,
        'delta': x[8],
        'I_KATP': x[9] * pA,
        'energy_factor': x[10],
        'epsilon_0': 1,
        'epsilon_c': 0.15
    }

    return best_params


def create_brian_model(params, N=1):
    """
    Creates a Brian2 NeuronGroup with the given parameters
    """

    eqs = """
    E_L = E_0 * (1 - energy_factor * ((epsilon_0-epsilon)/(epsilon_0 - epsilon_c))) : volt
    dV/dt  = (g_L*(E_L-V) + g_L*Delta_T*(epsilon - epsilon_c)*exp((V-V_th)/Delta_T)/(epsilon_0) + I_stim(t) - w) / C_m : volt
    dw/dt   = (a*(V-E_L) - w + I_KATP*((epsilon_0-epsilon)/(epsilon_0 - epsilon_c))) / tau_w : amp
    depsilon/dt = (ATP - pump - w/gamma) / tau_e : 1
    ATP = ATP_k * (1 - epsilon / epsilon_0) : 1
    pump = pump_k * (V - E_L) * (1 / (1 + exp(-(V - E_L) / (1*mV)))) * 1 / (1 + exp(-10*(epsilon - epsilon_c) / 1)) : 1
    """

    neuron = NeuronGroup(1, model=eqs,
                         threshold='V > V_th and epsilon > epsilon_c',
                         reset="V = V_reset; w += b; epsilon -= delta",
                         method='heun',
                         namespace=params)

    # Set default values
    neuron.V_m = params['E_0']
    neuron.epsilon = params["epsilon_0"]
    neuron.w = 0 * pA

    return neuron


if __name__ == "__main__":
    # Paths
    stim_path = "voltage_fitting/data/stimulus_Layer 2.txt"
    resp_path = "voltage_fitting/data/response_Layer 2.txt"
    x0 = [5.0, 60.0, 10.0, -52.0, 200.0, 500, -35.0, -45.0, 0.02, 1.0, -50.0, 6, 1, 280, 3, -40]

    if os.path.exists(stim_path):
        final_params = fit_my_model(stim_path, resp_path, x0)
        print("Final Parameters:")
        for k, v in final_params.items():
            print(f"{k} = {v}")
    else:
        print("Data files not found in voltage_fitting/data/")
