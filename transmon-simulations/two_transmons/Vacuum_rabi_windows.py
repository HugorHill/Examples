from two_transmons.ZPulse import *

def vacuum_rabi_waveform (duration, phi_offset, Ts, params):
    signal = ZPulse(Ts,params)
    waveform1 = signal._normalized_pulse(200, 200 + duration)*phi_offset + params['phi_base_level']
    waveform2_const = ones_like(Ts)*(params['phi2z_base_level'])
    waveform3_const = ones_like(Ts)*0.6
    waveform4_const = ones_like(Ts)*0.7    
    return waveform1, waveform2_const, waveform3_const, waveform4_const   

def vacuum_rabi_population(chain, params, duration, phi_offset):
    wf1, wf2, wf3, wf4 = vacuum_rabi_waveform(duration, phi_offset, chain._Ts, params)
    H_full = chain.build_H_full([wf1, wf2, wf3, wf4], 
                                 params, [[33.45*5,0],[0,0],[0,0],[0,0]])
    result = mesolve(H_full, chain.rho0, chain._Ts, c_ops = [], e_ops = chain.e_ops, 
                     progress_bar = None,options=Options(nsteps = 20000, store_states = True, max_step = 1e-1))
    return result.expect[1][-1]
    
def vacuum_rabi_populations_one_phi(chain, params, durations, phi_offset):
    populations_one_phi = []
    size = len(durations)
    for dur in durations:
        populations_one_phi.append(vacuum_rabi_population(chain, params, dur, phi_offset))
    return populations_one_phi    
def vacuum_rabi_populations_onephi_2qubits(chain, params, durations, phi_offset):
    populations_one_phi = []
    size = len(durations)
    Ts=chain._Ts
    for dur in durations:
        signal = ZPulse(Ts,params)
        waveform1 = signal._normalized_pulse(200, 200 + dur)*phi_offset + params['phi_base_level']
        waveform2_const = ones_like(Ts)*(params['phi2z_base_level'])
        H_full = chain.build_H_full([waveform1, waveform2_const], 
                                 params, [[33.45*5,0],[0,0]])
        result = mesolve(H_full, chain.rho0, chain._Ts, c_ops = chain.build_c_ops(), e_ops = chain.e_ops, 
                     progress_bar = None,options=Options(nsteps = 20000, store_states = True, max_step = 1e-1))
        num_measure = (200 + dur)/params['finish']*len(Ts) 
        populations_one_phi.append(result.expect[1][int(num_measure)])
    return populations_one_phi 

def vacuum_rabi_populations_one_phi_windows(args):
    chain = args ['chain']
    params = args['params']
    durations = args['durations']
    phi_offset = args['phi_offset']    
    populations_one_phi = []
    size = len(durations)
    for dur in durations:
        populations_one_phi.append(vacuum_rabi_population(chain, params, dur, phi_offset))
    return populations_one_phi

