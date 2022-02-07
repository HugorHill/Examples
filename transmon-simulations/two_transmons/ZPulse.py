from scipy import *
from qutip import *
from matplotlib.pyplot import *
from numpy import pi, sign
import math
from scipy.signal import *
from two_transmons.ParallelCalculations import *

class ZPulse:
    
    def __init__(self, Ts, params):
        self._params = params
        self._Ts = Ts
        
    def _waveform_triangle (self, t_start, t_finish, delta = 0.1):
        tau = delta * (t_finish - t_start)
        wf_rise = (self._Ts - t_start)*(1 + sign(self._Ts-t_start))*(1+ sign(t_start + tau - self._Ts))/4/tau
        wf_down = (t_finish - self._Ts)*(1 + sign(self._Ts-t_finish+tau))*(1 + sign(t_finish - self._Ts))/4/tau
        wf_flat = (1+sign(self._Ts - t_start - tau))*(1+sign(t_finish -tau -self._Ts))/4
        waveform = wf_rise + wf_down + wf_flat
        return waveform
    
    def waveform_triangle_t(self, t, t_start, t_finish, delta = 0.1):
        tau = delta * (t_finish - t_start)
        wf_rise = (t - t_start)*(1 + sign(t-t_start))*(1+ sign(t_start + tau - t))/4/tau
        wf_down = (t_finish - t)*(1 + sign(t-t_finish+tau))*(1 + sign(t_finish - t))/4/tau
        wf_flat = (1+sign(t - t_start - tau))*(1+sign(t_finish -tau -t))/4
        waveform = wf_rise + wf_down + wf_flat
        return waveform 
    def phi_coeffs_t (self, t, t_start, t_finish):
        return self.waveform_triangle_t
        
        
    def _step_rising(self, t_start = None):
        params = self._params
        if t_start is None:
            t_start = params["start"]        
        return (tanh((self._Ts - 2 * params["tanh_sigma"] - t_start) / params["tanh_sigma"])+1)/2
    
    def _step_falling(self, t_finish = None):
        params = self._params
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]        
        return (tanh((-self._Ts - 2 * params["tanh_sigma"] + t_finish) / params["tanh_sigma"])+1)/2
    
    def _normalized_pulse(self, t_start = None, t_finish = None):
        params = self._params
        if t_start is None:
            t_start = params["start"]  
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]
        raw = self._step_rising(t_start)*self._step_falling(t_finish)
        normalized = raw/max(raw)
        return normalized
        #return self._waveform_triangle(t_start,t_finish,0.1)
    def waveform1(self, t_start = None, t_finish = None):
        params = self._params
        if t_start is None:
            t_start = params["start"]  
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]
        offset = self._params["phi_offset"]
        base = self._params["phi_base_level"]
        #return self._waveform_triangle(t_start, t_finish, 0.1)*offset + base
        return self._normalized_pulse(t_start, t_finish)*offset+base
    
    def waveform (self,t_start = None, t_finish = None):        
        params = self._params
        if t_start is None:
            t_start = params["start"]  
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]
        sos=butter(5,self._params['frequency'],btype='lowpass', output='sos')
        filtered=sosfilt(sos,self.waveform1(t_start, t_finish))
        #return filtered
        return self.waveform1(t_start, t_finish)
    
    
    def waveform_2iswap(self, number): # waveform for 4 sequential i-swap gates
        params = self._params
        if (number == 1):
            base = self._params["phi_base_level"]
        elif (number == 2):
            base = params["phi2z_base_level"]
            
        params = self._params
        t_starts = [params["start"], params["start"] + 90.1]
        t_finishes = t_starts.copy()
        for i in range (2):
            t_finishes[i] += params["duration"]
        #waveform = sum([self.waveform(t_starts[i],t_finishes[i]) for i in range(4)])
        waveform = self.waveform_iswap_zgate(number, t_starts[0], t_finishes[0]) +\
        self.waveform_iswap_zgate(number, t_starts[1], t_finishes[1]) - base
        #waveform = self.waveform_iswap_zgate(number, t_starts[1], t_finishes[1])
        return waveform
    
    def waveform_iswap_zgate(self, number, t_start = None, t_finish = None, t_zgate = None): #number - number of qubit
        params = self._params
        if t_start is None:
            t_start = params["start"]  
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]
        if t_zgate is None:
            if (number == 1):
                t_zgate = params["t_zgate"]
            elif number == 2 :
                 t_zgate = params["t_zgate2"]
        if (number == 1):
            offset = self._params['phi_offset']
            offsetz = self._params["phiz_offset"]
            base = self._params["phi_base_level"]
            waveform1 = self._normalized_pulse(t_start, t_finish)*offset + base
            waveform_z = self._normalized_pulse(20 + t_finish, 20 + t_finish + t_zgate)*offsetz + base
            waveform = waveform1  + waveform_z - base
        elif (number == 2):
            offsetz = self._params["phi2z_offset"]
            base = params["phi2z_base_level"]
            waveform = self._normalized_pulse(20 + t_finish, 20 + t_finish + t_zgate)*offsetz + base
            #waveform = ones_like(self._Ts)*0.5
        return waveform
    
    
    def waveform_cphase (self, t_start = None, t_finish = None, offset = None):
        params = self._params
        if t_start is None:
            t_start = params["start_cphase"]  
        if t_finish is None:
            t_finish =  params["start_cphase"] + params["duration_cphase"]
        if offset is None:
            offset = self._params["phi_offset_cphase"]
        base = self._params["phi_base_level"]
        return self._normalized_pulse(t_start, t_finish)*offset+base     
            
        
        
    
    '''
    def waveform_iswap_zgate_second (self, t_start = None, t_finish = None, t_zgate = None):
        params = self._params
        if t_start is None:
            t_start = params["start"]  
        if t_finish is None:
            t_finish =  params["start"] + params["duration"]
        if t_zgate is None:
            t_zgate = params["t_zgate2"]
        params = self._params
        offsetz = self._params["phi2z_offset"]
        base = params["phi2z_base_level"]
        #waveform_z = self._normalized_pulse(t_start-t_zgate, t_start)*offsetz + base
        waveform_z = self._normalized_pulse(5 + t_finish, 5 + t_finish + t_zgate)*offsetz + base
        return waveform_z
        '''
        
    def plot_iswap_zgate(self, number):
        plot(self._Ts, self.waveform_iswap_zgate(number))
    
    def plot_iswap_zgate_second(self):
        plot(self._Ts, self.waveform_iswap_zgate_second())
        
    def plot_4iswap(self):
        plot(self._Ts, self.waveform_4iswap())   
    
    def plot_2iswap(self, number):
        plot(self._Ts, self.waveform_2iswap(number))   
        
    def plot(self):
        plot(self._Ts, self.waveform())
    def plot_ideal(self):
        plot(self._Ts,self.waveform1())
       
