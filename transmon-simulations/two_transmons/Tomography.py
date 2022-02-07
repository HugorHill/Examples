from scipy import *
from qutip import *
from numpy import sqrt, ndarray, dot
from matplotlib.pyplot import *
from itertools import product
from scipy.linalg import cholesky,sqrtm
from tqdm import tqdm_notebook
from tqdm import tqdm
import p_tqdm
from p_tqdm import p_map
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm
from scipy.optimize import *
import multiprocessing
from multiprocessing import Pool
from functools import partial, partialmethod
from numpy.linalg import norm, inv
import numpy as np
#import numba
#from numba import jit, njit
from scipy.interpolate import * 

from two_transmons.ZPulse import ZPulse

class Tomography:
    
    def __init__(self, dts, Ts, params, readout_resonator):
        self._dts = dts
        self._dts.e_ops = [dts.two_qubit_operator(qubit1_operator = dts._tr1.sx()),dts.two_qubit_operator(qubit1_operator = dts._tr1.sy()),
        dts.two_qubit_operator(qubit1_operator = dts._tr1.sz()), dts.two_qubit_operator(qubit2_operator = dts._tr2.sx()),
        dts.two_qubit_operator(qubit2_operator = dts._tr2.sy()),dts.two_qubit_operator(qubit2_operator = dts._tr2.sz())]
        self._Ts = Ts
        self._params = params
        self._r = readout_resonator
        self._options = Options(nsteps=20000, store_states=True, max_step = 1e-2)
        self._c_ops = [] #self._dts.c_ops(0, 1 / 2)
        #self._pi_duration = 33.45 # don't touch
        #self._drive_amplitude = 0.01*2*pi/2*3 # don't touch
        self._pi_duration = 33.45*5
        self._drive_amplitude = 0.01*2*pi/2*3/5 
        E0 = dts.gg_state(params['phi_base_level'], 1/2, True)[1]
        E10 = dts.e_state(params['phi_base_level'], 1/2, 1, True)[1]
        E01 = dts.e_state(params['phi_base_level'], 1/2, 2, True)[1]
        self._freqs = ((E10-E0)/2/pi,(E01-E0)/2/pi) #  resonance frequencies of qubits        
        self._q1_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
            
        self._q2_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
        
        self._2q_rotations = list(product(self._q1_rotations, self._q2_rotations)) #time and angle representation
        #self._H0 = dts.H_td_diag_approx(*self.build_waveforms()) # part of Hamiltonian without drive        
        self._measurement_operators = []
        
        #self._meas_op = self._dts.two_qubit_operator(qubit1_operator = 60*Qobj([[-1,0],[0,1]])) +\
        #self._dts.two_qubit_operator(qubit2_operator = 50*Qobj([[-1,0],[0,1]])) +\
        #40*self._dts.two_qubit_operator(qubit1_operator = Qobj([[-1,0],[0,1]]),qubit2_operator = Qobj([[-1,0],[0,1]]))
       
        
        
        self._meas_op = self.build_2qubit_operator(q1_ax = 'Z')*60 + self.build_2qubit_operator(q2_ax = 'Z')*50 +\
        self.build_2qubit_operator(q1_ax = 'Z', q2_ax = 'Z')*40
       
        rotations_1 = [Qobj([[1,0,0],[0,1,0],[0,0,1]]),  #identity
            Qobj(array([[0,-1j,0],[-1j,0,0],[0,0,1]])),  #X +pi
            Qobj(1/sqrt(2)*array([[1,-1j,0],[-1j,1,0],[0,0,sqrt(2)]])), #X +pi/2
            Qobj(1/sqrt(2)*array([[1,1j,0],[1j,1,0],[0,0,sqrt(2)]])), #X -pi/2          
            Qobj(1/sqrt(2)*array([[1,1,0],[-1,1,0],[0,0,sqrt(2)]])),  # Y +pi/2
            Qobj(1/sqrt(2)*array([[1,-1,0],[1,1,0],[0,0,sqrt(2)]]))] # Y -pi/2
        rotations_2 = rotations_1
        self._rotations_matrix = [] #matrix representation of rotations
        for state1 in rotations_1:
            for state2 in rotations_2:
                self._rotations_matrix.append(self._dts.two_qubit_operator(state1,state2))
        
        for rotation in self._rotations_matrix:
            self._measurement_operators.append(rotation.dag()*self._meas_op*rotation)
            #self._measurement_operators.append(rotation*self._meas_op*rotation.dag())
        
        '''
        pauli_mat_1q = [basis(3,0)*basis(3,0).dag()+basis(3,1)*basis(3,1).dag() + basis(3,2)*basis(3,2).dag(),
                        basis(3, 0) * basis(3, 1).dag() + basis(3, 1) * basis(3, 0).dag() + basis(3,2)*basis(3,2).dag(),
                        -(-1j * basis(3, 0) * basis(3, 1).dag() + 1j * basis(3, 1) * basis(3, 0).dag()) + basis(3,2)*basis(3,2).dag(),
                        -(ket2dm(basis(3, 0)) - ket2dm(basis(3, 1))) + basis(3,2)*basis(3,2).dag()]
        '''
        pauli_mat_1q = [basis(2,0)*basis(2,0).dag()+basis(2,1)*basis(2,1).dag(),
                        basis(2, 0) * basis(2, 1).dag() + basis(2, 1) * basis(2, 0).dag(),
                        -(-1j * basis(2, 0) * basis(2, 1).dag() + 1j * basis(2, 1) * basis(2, 0).dag()),
                        -(ket2dm(basis(2, 0)) - ket2dm(basis(2, 1)))]
        self._pauli_mat_2qubits = []               #basis of 2-qubit operators for process tomography
        for p1 in pauli_mat_1q:
            for p2 in pauli_mat_1q:
                self._pauli_mat_2qubits.append(tensor(p1,p2).full())
#        self._rho0 = dts.ee_state(0,1/2)
#        self._rho0 = 1/2*(dts.gg_state(0,1/2)+dts.e_state(0, 1/2,1)+
#                           dts.e_state(0, 1/2,2)+
#                          dts.ee_state(0, 1/2))
#        self._rho0 = dts.e_state(0, 1/2, 1)
#        self._rho0 = 1/sqrt(2)*(dts.e_state(0, 1/2, 1)*1j+dts.e_state(0, 1/2, 2))
        self._rho0 = Qobj(ket2dm(basis(9,0)), dims = [[3,3],[3,3]])
#        self._rho0 = self._rho0*self._rho0.dag()
        self._rho_prepared_states = []
        for rotation in self._rotations_matrix:
            self._rho_prepared_states.append(rotation*self._rho0*rotation.dag())
            #self._rho_prepared_states.append(rotation.dag()*self._rho0*rotation) #base states for maximum likelihood estimation
        d = 4
        self._PP = np.empty((d**2,36,d**2,d,d),dtype = complex) # pre-calculation of matrices in operator decomposition
        self._PP_identity = np.empty((d**2, d**2,d,d),dtype = complex)
        
        self._rho_prepared_full = array ([rho.full() for rho in self._rho_prepared_states]) # 3*3 matrix
        self._rho_prepared_full_trunc = array ([Tomography.rho3dim_to_rho(rho).full() for rho in self._rho_prepared_states]) # 2*2 matrix
        for j in range (d**2):
            for i in range (len(self._rho_prepared_states)):
                for k in range (d**2):
                    self._PP_identity[j,k] = dot(self._pauli_mat_2qubits[j], self._pauli_mat_2qubits[k])                 
                    self._PP[j,i,k] = dot(dot(self._pauli_mat_2qubits[j],self._rho_prepared_full_trunc[i]),self._pauli_mat_2qubits[k])
                    
                    
                    
    def set_pi_duration (self, pi_duration):
        self._pi_duration = pi_duration
        self._q1_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
            
        self._q2_rotations = [(0, 0),
                              (self._pi_duration, 0),
                              (self._pi_duration/2, 0),
                              (self._pi_duration/2, pi),
                              (self._pi_duration/2, pi/2),                              
                              (self._pi_duration/2, -pi/2)]
        self._2q_rotations = list(product(self._q1_rotations, self._q2_rotations))
    def set_freqs (self, freq1, freq2):
        self._freqs = (freq1, freq2,)
        
    def build_waveforms(self):
        signal = ZPulse(self._Ts, self._params)
        waveform1 = signal.waveform_iswap_zgate(1)
        waveform2 = signal.waveform_iswap_zgate(2)
        #waveform1 = ones_like(self._Ts)*0
        #waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
    def my_iswap_test(self, rotations, full_output = False):
        signal = ZPulse(self._Ts,self._params)
        waveform1 = signal.waveform_iswap_zgate(1)
        waveform2 = ones_like(waveform1)*1/2
        H0 = self._dts.H_td_diag_approx(waveform1, waveform2)
        Hdr = self._dts.Hdr([self._drive_amplitude]*2,
              (0,self._pi_duration/2),
               (10,10), 
              (0,0), 
             self._freqs, 
            (self._params['phi_base_level'],1/2))
        H_const = H0[0][0]*H0[0][1][0] + H0[1][0]*H0[1][1][0] + H0[2][0]*H0[2][1][0]
        H = H0 + Hdr
        result = mesolve(H, self._rho0, self._Ts, c_ops = [],e_ops = [],progress_bar=True,options=Options(nsteps = 20000, store_states = True, max_step = 1e-3))
        density_matrix = result.states
        dm_interaction = empty_like(self._Ts, dtype = Qobj)
        for ind, t in enumerate (self._Ts):
            H_current = H_const*t*1j
            matrix_exp = H_current.expm()
            dm_interaction[ind] = matrix_exp*density_matrix[ind]*matrix_exp.dag()
        return dm_interaction
    
   # def build_waveforms_cphase(self)
    def build_waveforms_cphase (self):
        signal = ZPulse(self._Ts, self._params)
        waveform1 = signal.waveform_cphase() 
        waveform2 = self._params['phi2z_base_level']*ones_like(waveform1)
        return waveform1, waveform2
        
    def build_waveforms_4iswap(self):
        waveform1 = ZPulse(self._Ts, self._params).waveform_4iswap()
        #waveform1 = ones_like(self._Ts)*0
        waveform2 = ones_like(self._Ts)*1/2
        return waveform1, waveform2
    
    def build_waveforms_2iswap(self):
        signal = ZPulse(self._Ts, self._params)
        waveform1_1 = signal.waveform_iswap_zgate(1)
        waveform1_2 = signal.waveform_iswap_zgate(1, self._params['start'] + 300, self._params['start']+250 + self._params['duration'], 
                                                  self._params['t_zgate_2iswap'] )
        #waveform1 = ones_like(self._Ts)*0
        waveform2_1 = signal.waveform_iswap_zgate(2)
        waveform2_2 = signal.waveform_iswap_zgate(2, self._params['start'] + 300, self._params['start']+300 + self._params['duration'], 
                                                  self._params['t_zgate2_2iswap'] )
        return waveform1_1 + waveform1_2 - self._params["phi_base_level"], waveform2_1 + waveform2_2 - self._params["phi2z_base_level"]
    
    def build_const_waveforms(self):
        waveform1 = ones_like(self._Ts)*self._params['phi_base_level']
        waveform2 = ones_like(self._Ts)*self._params['phi2z_base_level']
        return waveform1, waveform2
       
        
    def run(self):
       # self._options = Options(nsteps=20000, store_states=True)
        
       # self._c_ops = [] #self._dts.c_ops(0, 1/2)
        
        try: 
            self._results
        except AttributeError:
            self._results = []
        
       
        
        self._results = parallel_map(self._tomo_step_parallel, self._2q_rotations)
            
        return self._results

    def run_rabi (self, freqs, n_qubit):
        #self._options = Options(nsteps=20000, store_states=True)
        
        #self._c_ops = [] #self._dts.c_ops(0, 1/2)
        
       
        self._results_rabi = []
        
        E0 = self._dts.gg_state(self._params['phi_base_level'], 1/2, True)[1]
        E10 = self._dts.e_state(self._params['phi_base_level'], 1/2, 1, True)[1]
        E01 = self._dts.e_state(self._params['phi_base_level'], 1/2, 2, True)[1]
        #self._freqs = ((E10 - E0)/2/pi, (E01 - E0)/2/pi)
        if (n_qubit == 1):
            self._results_rabi = self._tomo_step ([(self._pi_duration, 0),(0,0)], freqs)
        elif ( n_qubit == 2):
            self._results_rabi = self._tomo_step ([(0,0),(self._pi_duration, 0)], freqs)
        
        return self._results_rabi

       
       
    def run_iswap_test (self, cphase = False,iswap2 = False, num_cpus = 24):
        self._results_iswap_test = []  
        self._results_iswap_test = p_map(self.run_iswap_test_step_parallel, self._2q_rotations, [cphase]*len(self._2q_rotations),\
                                         [iswap2]*len(self._2q_rotations),[False]*len(self._2q_rotations),\
                                         [False]*len(self._2q_rotations), num_cpus = num_cpus)
    
    
    def run_iswap_test_step_parallel (self, rotations, cphase = False, iswap2 = False, constant_flows = False, full_output = False): #возвращает матрицу плотности после гейта в                                                                                                    представлении вз-я
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        if iswap2 :
            waveform1, waveform2 = self.build_waveforms_2iswap()
        elif cphase:
            waveform1, waveform2 = self.build_waveforms_cphase()
        elif constant_flows:
            waveform1, waveform2 = self.build_const_waveforms()
        else:
            waveform1, waveform2 = self.build_waveforms()      
        H_new = self._dts.H_iswap_RF_RWA_td(waveform1, waveform2, self._params, rotations = rotations)
        result = mesolve(H_new, self._rho0, self._Ts,e_ops = self._dts.e_ops,\
        c_ops = self._c_ops,progress_bar=None,options=self._options)    
        
        
        
        if (full_output):
            return result
        else:
            return result.states[-1]

    def run_iswap (self, num_cpus  = 24, store_states = False):
        
        self._results_iswap = []  
        self._results_iswap = p_map(self.run_iswap_step_parallel, self._2q_rotations, [False]*len(self._2q_rotations), 
                                    num_cpus = num_cpus)
    
    
    
    def run_iswap_step_parallel (self, rotations, full_output = False, iswap2 = False):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        if (iswap2):
            H0 = self._dts.H_td_diag_approx(self.build_waveforms_2iswap()) #H0 - is the Hamiltonian of state preparing and our gate, after                                                                               #that we need tomography
        else:
             H0 = self._dts.H_td_diag_approx(self.build_waveforms())
        H0 += self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1, dur2),
                           (self._params['start']- dur1-10,self._params['start']-dur2-10),  # time of starting state preparation
                           #(phi1 + 430.12830040232643, phi2 - 0.9735746086225845),
                           (phi1, phi2), 
                           self._freqs,
                            (self._params['phi_base_level'],0.5))
        density_matrix_simulated = []
        density_matrix_final_states = [] 
        for rotation in tqdm_notebook(self._2q_rotations):
            dur1_tomo, dur2_tomo = rotation[0][0], rotation[1][0]
            phi1_tomo, phi2_tomo = rotation[0][1], rotation[1][1]
            H = H0 + self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1_tomo, dur2_tomo),
                           (self._params["duration"] + self._params['start'] + self._params["t_zgate"] + 20, 
                            self._params["duration"] + self._params['start'] + self._params["t_zgate"] + 20),
                           (phi1_tomo, phi2_tomo),
                           self._freqs,
                            (self._params['phi_base_level'],0.5))
            density_matrix = mesolve(H, self._rho0, self._Ts, c_ops = self._c_ops,progress_bar=None,options=self._options).states
            if (full_output):
                density_matrix_simulated.append(density_matrix)
            density_matrix_final_states.append(density_matrix[-1])
            del H
        
        if full_output:
            return (density_matrix_simulated, density_matrix_final_states)
        return density_matrix_final_states
    
    '''
    def run_4iswap (self, num_cpus = 12, store_states = False):
        
        self._results_4iswap = []  
        self._results_4iswap = p_map(self.run_4iswap_step_parallel, self._2q_rotations, [False]*len(self._2q_rotations), num_cpus = 24)       
    def run_4iswap_step_parallel (self, rotations, full_output = False):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        H0 = self._dts.H_td_diag_approx(*self.build_waveforms_4iswap()) #H0 - is the Hamiltonian of state preparing and our gate, after that
        H0 += self._dts.Hdr([self._drive_amplitude] * 2,                     # we need tomography
                           (dur1, dur2),
                           (self._params['start']- dur1,self._params['start']-dur2),  # time of starting state preparation
                           #(phi1 + 430.12830040232643, phi2 - 0.9735746086225845),
                           (phi1, phi2), 
                           self._freqs)
        density_matrix_simulated = []
        density_matrix_final_states = [] 
        for rotation in tqdm_notebook(self._2q_rotations):
            dur1_tomo, dur2_tomo = rotation[0][0], rotation[1][0]
            phi1_tomo, phi2_tomo = rotation[0][1], rotation[1][1]
            H = H0 + self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1_tomo, dur2_tomo),
                           (self._params["duration"]+self._params['start'] + 3*(self._params["duration"] +10),
                            self._params["duration"] + self._params['start'] +3*(self._params["duration"] +10)), 
                           (phi1_tomo + 0*self._phase1_offset, phi2_tomo + 0*self._phase2_offset),
                           self._freqs)
            density_matrix = mesolve(H, self._rho0, self._Ts, c_ops = self._c_ops,progress_bar=None,options=self._options).states
            if (full_output):
                density_matrix_simulated.append(density_matrix)
            #num_point = self._params['t_points']*
            density_matrix_final_states.append(density_matrix[-1])
            del H
        
        if full_output:
            return (density_matrix_simulated, density_matrix_final_states)
        return density_matrix_final_states
        '''
    def run_2iswap (self, num_cpus = 24, store_states = False):
        
        self._results_2iswap = []  
        self._results_2iswap = p_map(self.run_2iswap_step_parallel, self._2q_rotations, [False]*len(self._2q_rotations), 
                                     num_cpus = num_cpus)       
    def run_2iswap_step_parallel (self, rotations, full_output = False):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]
        H0 = self._dts.H_td_diag_approx(*self.build_waveforms_2iswap()) #H0 - is the Hamiltonian of state preparing and our gate, after that
        H0 += self._dts.Hdr([self._drive_amplitude] * 2,                     # we need tomography
                           (dur1, dur2),
                           (self._params['start']- dur1,self._params['start']-dur2),  # time of starting state preparation
                           #(phi1 + 430.12830040232643, phi2 - 0.9735746086225845),
                           (phi1, phi2), 
                           self._freqs,
                           (0,0.5))
        density_matrix_simulated = []
        density_matrix_final_states = [] 
        for rotation in tqdm_notebook(self._2q_rotations):
            dur1_tomo, dur2_tomo = rotation[0][0], rotation[1][0]
            phi1_tomo, phi2_tomo = rotation[0][1], rotation[1][1]
            H = H0 + self._dts.Hdr([self._drive_amplitude] * 2,
                           (dur1_tomo, dur2_tomo),
                           (self._params["duration"] + self._params['start'] + self._params["t_zgate"] + 
                            3*(self._params["duration"] + self._params["t_zgate"] +15)+20,
                            self._params["duration"]+self._params['start']+ self._params["t_zgate"]
                            +3*(self._params["duration"] + self._params["t_zgate"] +15)+20), 
                           (phi1_tomo, phi2_tomo),
                           self._freqs,
                           (0,0.5))
            density_matrix = mesolve(H, self._rho0, self._Ts, c_ops = self._c_ops,progress_bar=None,options=self._options).states
            if (full_output):
                density_matrix_simulated.append(density_matrix)
            #num_point = self._params['t_points']*
            density_matrix_final_states.append(density_matrix[-1])
            del H
        
        if full_output:
            return (density_matrix_simulated, density_matrix_final_states)
        return density_matrix_final_states
        
    
    def _find_rho_iswap(self, averages, rho_final_simulated): #rho_final_simulated is an array of 
                                                               #density matrixes at the end of simulation, for tomography
        x0 = random.rand(16)
        for n in range(averages):
            new = minimize (self._likelihood_rho_iswap, x0, args = (rho_final_simulated), method='BFGS', tol= 1e-09,\
                            options = {'gtol': 1e-09,'maxiter': 5000})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            #x0 =best.x + 1*(random.rand(16)*2 - 1)
            x0 =random.rand(16)*2 - 1
            print (best.fun, new.fun)            
        return self.x_to_rho(best.x) 
    
    
    def find_iswap_rotation_error (self, number):
        rho_initial = Tomography.rho3dim_to_rho(self._rho_prepared_full[number])
        iswap_matrix = array([[1,0,0,0],[0,0,-1j,0],[0,-1j,0,0],[0,0,0,1]])
        rho_ideal =dot(iswap_matrix,dot(rho_initial, iswap_matrix.transpose().conj())) 
        dm_final = self.run_iswap_step_parallel(self._2q_rotations[number], full_output = False)
        rho_predicted = self._find_rho_iswap(3,dm_final).full()
        delta = (rho_predicted - rho_ideal)
        return norm(delta, ord = 'fro')
    
    
    def find_iswap_rotation_matrix (self,number):
        rho_initial = Tomography.rho3dim_to_rho(self._rho_prepared_full[number])
        iswap_matrix = array([[1,0,0,0],[0,0,-1j,0],[0,-1j,0,0],[0,0,0,1]])
        rho_ideal =dot(iswap_matrix,dot(rho_initial, iswap_matrix.transpose().conj())) 
        dm_final = self.run_iswap_test_step_parallel(self._2q_rotations[number], iswap2 =False, full_output = False)
        #rho_predicted = self._find_rho_iswap(5,dm_final).full()
        return (Tomography.rho3dim_to_rho(dm_final), rho_ideal)
    
    def find_cphase_rotation_matrix (self, number, phase):
        rho_initial = Tomography.rho3dim_to_rho(self._rho_prepared_full[number])
        cphase_matrix = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,exp(1j*phase)]])
        rho_ideal =dot(cphase_matrix,dot(rho_initial, cphase_matrix.transpose().conj())) 
        dm_final = self.run_iswap_test_step_parallel(self._2q_rotations[number], cphase = True, full_output = False)
        #rho_predicted = self._find_rho_iswap(5,dm_final).full()
        return (Tomography.rho3dim_to_rho(dm_final), rho_ideal)        
        
    
    def find_iswap2_rotation_matrix (self,number):
        rho_initial = Tomography.rho3dim_to_rho(self._rho_prepared_full[number])
        iswap2_matrix = array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        rho_ideal =dot(iswap2_matrix,dot(rho_initial, iswap2_matrix.transpose().conj())) 
        dm_final = self.run_iswap_test_step_parallel(self._2q_rotations[number], iswap2 =True, full_output = False)
        #rho_predicted = self._find_rho_iswap(5,dm_final).full()
        return (Tomography.rho3dim_to_rho(dm_final), rho_ideal)
    
    
    '''
    def find_iswap_error (self, phi1 = None, phi2 = None):
        self._options = Options(nsteps=20000, store_states=False)
        self._c_ops = self._dts.c_ops(0, 1 / 2)
       
        
        find_iswap_partial = partial (Tomography.find_iswap_rotation_matrix, self)
        errors_array = p_map (find_iswap_partial, range(len(self._2q_rotations)))
        return np.sum(errors_array**2)
    '''
   
    def _likelihood_rho_iswap (self, x, rho_final_simulated): # calculation likelihood giving results of the system evolution
        rho = self.x_to_rho(x)     #with process parametrization x  
        rho_3dim = Tomography.rho_to_rho3dim(rho)
        lh = 0
        for ind, rho_true in enumerate(rho_final_simulated):
            meas_op = self._measurement_operators[ind]
            lh+=(expect(self._meas_op, rho_true) - expect(meas_op,rho_3dim))**2
            #lh+=(expect(self._meas_op, rho_true) - expect(meas_op,rho))**2
        return abs(lh)
    
    def _operator_representation (self, chi): #gives part of sum which includes chi
        d = 4
        q_dim = 2
        #operator = Qobj(numpy.zeros((4,4)),dims=[[q_dim, q_dim],[q_dim, q_dim]])
        #pauli = self._pauli_mat_2qubits
        #rho_initial = self._rho_prepared_states[number]
        
        #for j in range (d**2):
         #   for k in range (d**2):
                #operator+= chi[j,k]*pauli[j]*rho_initial*pauli[k]
          #      operator+=chi[j,k]*self._PP[j,number,k]
        operator = np.tensordot(chi, self._PP, ((0,1),(0,2)))
        return operator
                
   
     
    def _likelihood_iswap (self, x, density_matrix_after_gate): #density matrix array - is an array of resulte_dm (lambda) after tomography
        chi = Tomography.x_to_chi(x)                             # use numpu ndarray, not qutip matrix for good productivity
        lh = 0
        #likelihood_par = partial(Tomography._likelihood_1_iter,self,x)
        #lh_array = p_map(likelihood_par, density_matrix_after_gate, range(len( density_matrix_after_gate)))
        operator = np.tensordot(chi, self._PP, ((0,1),(0,2)))
        lh += norm( np.tensordot (chi, self._PP_identity, ((0,1),(0,1)) )  - np.identity(4), ord = 'fro')
        for ind, rho_final in enumerate (density_matrix_after_gate):
            delta_rho = rho_final - operator[ind]
            lh += norm(delta_rho, ord = 'fro')
        return lh
        #return sum(lh_array)
        
        
    def _likelihood_1_iter (self, x, rho_final, ind):
        chi = Tomography.x_to_chi(x)
        delta_rho = rho_final - self._operator_representation(chi,ind)
        return (norm(delta_rho, ord = 'fro'))
    
    def process_tomo(self, averages, maxiter, iswap4 = False, iswap2 = False, test = False, iswap = False):    #start only after run_iswap
        density_matrix_after_gate = []
        #for ind, rho_initial in enumerate(self._rho_prepared_states):
           #density_matrix_after_gate.append(self._find_rho_iswap(3,self._results_iswap[ind])) # можно распараллелить
        find_rho_partial = partial(Tomography._find_rho_iswap,self,averages)
        if (iswap4):
            density_matrix_after_gate = p_map(find_rho_partial, self._results_4iswap ,num_cpus=24)
        elif (iswap2):
            density_matrix_after_gate = p_map(find_rho_partial, self._results_2iswap ,num_cpus=24)  
        elif (test):
            density_matrix_after_gate_3dim = self._results_iswap_test
            for dm in density_matrix_after_gate_3dim:
                density_matrix_after_gate.append(Tomography.rho3dim_to_rho(dm))
            
        elif (iswap):
            density_matrix_after_gate = p_map(find_rho_partial, self._results_iswap ,num_cpus=24)
        density_matrix_after_gate_full = []
        for dm in density_matrix_after_gate:
            density_matrix_after_gate_full.append(dm.full())    
        x0 = random.rand(256)
        for n in tqdm_notebook(range(averages), desc='Tomography iswap_operator: Likelihood minimization', ncols=700):
            new = minimize (self._likelihood_iswap, x0, args = (density_matrix_after_gate_full), method='BFGS', tol= 1e-09,\
                            options = {'gtol': 1e-05,'maxiter': maxiter})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            x0 =best.x + 0.1*(random.rand(256)*2 - 1)
            print ('error is', best.fun)            
        return self.x_to_chi(best.x) 
            

    def _tomo_step(self, rotations, freqs):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]

        H = self._H0
        H += self._dts.Hdr([self._drive_amplitude]*2, 
                                (dur1, dur2), 
                                #(self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                                #(10 + self._params["duration"]+self._params['start'], 10+ self._params["duration"] +self._params['start']),         
                                (10,10),
                                (phi1, phi2),
                                freqs)         #part of drive for Tomography
        #self._H += self._dts.Hdr_cont([self._drive_amplitude]*2)
        dm = mesolve(H, self._rho0, self._Ts, c_ops = self._c_ops, progress_bar=None, options=self._options).states
        del H

        return dm
    
    
    def _tomo_step_parallel(self, rotations):
        dur1, dur2 = rotations[0][0], rotations[1][0]
        phi1, phi2 = rotations[0][1], rotations[1][1]

        H = self._dts.H_td_diag_approx(*self.build_waveforms())
        H += self._dts.Hdr([self._drive_amplitude]*2,
                                (dur1, dur2), 
                                #(self._params["duration"]+20, self._params["duration"]+20+self._pi_duration+10),
                                (10 + self._params["duration"]+self._params['start'], 10 +self._params["duration"] + self._params['start']),                                 (phi1, phi2),
                                self._freqs,
                                  (self._params['phi_base_level'],0.5))
        #self._H += self._dts.Hdr_cont([self._drive_amplitude]*2)

        return mesolve(H, self._rho0,
                       self._Ts, c_ops = self._c_ops, 
                       progress_bar=None, 
                       options=self._options)
    
                       
    def _joint_expect(self, state, f, chi1, chi2):
        return expect(state, self._r.measurement_operator(f, chi1, chi2))
    
    def visualize_joint_readout(self, f, chi1, chi2):
        
        fig, axes = subplots(1, 2, figsize=(15,5))
        for result in self._results:
            data = array(serial_map(self._joint_expect, result.states, task_args = (f, chi1, chi2)))
            axes[0].plot(self._Ts, data.real)
            axes[1].plot(self._Ts, data.imag)
            
    def build_2qubit_operator (self, q1_ax = 'id', q2_ax= 'id'):
        q_ax =[q1_ax,q2_ax]
        q_ans = [Qobj(identity(3)),Qobj(identity(3))]
        
        for i in range (2):
            if q_ax[i]=='X':
                q_ans[i] = basis(3, 0) * basis(3, 1).dag() + basis(3, 1) * basis(3, 0).dag()
            if q_ax[i]=='Y':
                q_ans[i] = -(-1j * basis(3, 0) * basis(3, 1).dag() + 1j * basis(3, 1) * basis(3, 0).dag())
            if q_ax[i]=='Z': 
                q_ans[i] = -(ket2dm(basis(3, 0)) - ket2dm(basis(3, 1)))
        
        
        return tensor(*q_ans)
    
    
    def plot_qubits_dynamic (T, n, dm = None, interaction = False):   #start only after run function
        Ts = T._Ts
        if dm is not None:
            density_matrix = dm
        else:
            density_matrix = T._results[n].states
        if (interaction):
            dm_interaction = density_matrix
        else:        
            H0 = T._H0[0][0]*T._H0[0][1][0] +T._H0[1][0]*T._H0[1][1][0] + T._H0[2][0]*T._H0[2][1][0] # constant part of Hamiltonian
           # H0 = T._H0[0][0]*T._H0[0][1][0]+T._H0[1][0]*T._H0[1][1][0] 
            H0_list_expm = empty_like(Ts,dtype = Qobj)
            H0_list_expm_conj = empty_like(Ts, dtype = Qobj)
            dm_interaction = empty_like(Ts, dtype = Qobj)
            for ind, t in enumerate (Ts):
                H_current = H0*t*1j
                matrix_exp = H_current.expm()
                H0_list_expm [ind] = matrix_exp
                H0_list_expm_conj[ind] = matrix_exp.dag()
                dm_interaction[ind] = matrix_exp*density_matrix[ind]*matrix_exp.dag()
        
        qubit1_z_interaction = expect(T.build_2qubit_operator('Z','id'), dm_interaction)
        qubit1_y_interaction = expect(T.build_2qubit_operator('Y','id'), dm_interaction)
        qubit1_x_interaction = expect(T.build_2qubit_operator('X','id'), dm_interaction)
        qubit2_z_interaction = expect(T.build_2qubit_operator('id','Z'), dm_interaction)
        qubit2_y_interaction = expect(T.build_2qubit_operator('id','Y'), dm_interaction)
        qubit2_x_interaction = expect(T.build_2qubit_operator('id','X'), dm_interaction)
        
        fig = figure(figsize=(14,7))
        axes = fig.add_axes([0.05, 0.2, 0.4, 0.8], projection="3d")
        axes1 = fig.add_axes([0.55, 0.2, 0.4, 0.8], projection="3d")


        sph = Bloch(fig=fig, axes=axes)
        sph.clear()
        sph.sphere_alpha = 0
        sph.zlabel = [r'$\left|e\rightangle\right.$', r"$\left|g\rightangle\right.$"]
        sph.xlpos = [1.3, -1.3]
        sph.xlabel = [r'$\left.|+\right\rangle$', r"$\left.|-\right\rangle$"]
        sph.ylpos = [1.2, -1.3]
        sph.ylabel = [r'$\left.|-_i\right\rangle$', r"$\left.|+_i\right\rangle$"]
        sph.xlpos = [1.3, -1.3]
        nrm=matplotlib.colors.Normalize(0,Ts[-1])
        colors=cm.RdBu_r(nrm(Ts))
        sph.point_size=[40]
        sph.point_color = list(colors)
        sph.point_marker=['.']
        sph.add_points([qubit1_x_interaction, qubit1_y_interaction, qubit1_z_interaction], meth='m')
        #sph.add_points ([[1/sqrt(2),1,0],[1/sqrt(2),0,0],[0,0,1]])  
        sph.render(fig,axes)

        sph1 = Bloch(fig=fig, axes=axes1)
        sph1.clear()
        sph1.sphere_alpha = 0
        sph1.zlabel = [r'$\left|e\rightangle\right.$', r"$\left|g\rightangle\right.$"]
        sph1.xlpos = [1.3, -1.3]
        sph1.xlabel = [r'$\left.|+\right\rangle$', r"$\left.|-\right\rangle$"]
        sph1.ylpos = [1.2, -1.3]
        sph1.ylabel = [r'$\left.|-_i\right\rangle$', r"$\left.|+_i\right\rangle$"]
        sph1.xlpos = [1.3, -1.3]
        sph1.point_size=[40]
        sph1.point_color = list(colors)
        sph1.point_marker=['.']
        sph1.add_points([qubit2_x_interaction, qubit2_y_interaction, qubit2_z_interaction], meth='m')
        #sph.add_points ([[1/sqrt(2),1,0],[1/sqrt(2),0,0],[0,0,1]])  
        sph1.render(fig,axes)


        m = cm.ScalarMappable(cmap=cm.RdBu_r, norm=nrm)
        m.set_array(Ts)
        m.set_clim(0, Ts[-1])
        position=fig.add_axes([0.05,0.15,0.4,0.03])
        cb = fig.colorbar(m, orientation='horizontal', cax=position)
        cb.set_label("Time, ns")
        cb.set_ticks(np.linspace(0,round(Ts[-1]),6))
        sph.make_sphere()

        m1 = cm.ScalarMappable(cmap=cm.RdBu_r, norm=nrm)
        m1.set_array(Ts)
        m1.set_clim(0, Ts[-1])
        position1=fig.add_axes([0.55,0.15,0.4,0.03])
        cb1 = fig.colorbar(m1, orientation='horizontal', cax=position1)
        cb1.set_label("Time, ns")
        cb1.set_ticks(np.linspace(0,round(Ts[-1]),6))
        sph1.make_sphere()
        axes.text2D(0.3,1, "Qubit 1 state evolution",transform=axes.transAxes, fontsize = 20)
        axes1.text2D(1,1, "Qubit 2 state evolution",transform=axes.transAxes, fontsize = 20)  
        return fig
    
    
    def show_density_matrix (density_matrix):
        density_matrix_re = real (density_matrix)
        density_matrix_im = imag (density_matrix)
        fig = figure (figsize = (12,9) )
        ax1=fig.add_subplot(121, projection = '3d')
        ax2=fig.add_subplot(122, projection = '3d')
        xlabels = ['00','01','10','11']
        ylabels = ['00','01','10','11']
        matrix_histogram (density_matrix_re, xlabels, ylabels, (r'$Re\ \rho$'), limits = [-1,1], fig = fig, ax = ax1)
        matrix_histogram (density_matrix_im, xlabels, ylabels, (r'$Im\ \rho$'), limits = [-1,1], fig = fig, ax = ax2)
        ax1.view_init(azim = -30, elev = 40)
        ax2.view_init(azim = -30, elev = 40)
        return 
        
    
 
    def likelihood (self, x, n):                    # Вычисление Likelihood по загруженным измерениям \
        rho = self.x_to_rho(x)                      # для матрицы плотности заданной через x
        rho_3dim = Tomography.rho_to_rho3dim(rho)
        lh = 0
        rho_true = self._results[n].states[-1]      #replace to number of state
        for meas_op_result in self._measurement_operators:
            lh+=(expect(meas_op_result, rho_true) - expect(meas_op_result,rho_3dim))**2
        return abs(lh)
    
    def find_rho (self, averages, number): # number - number of rotation
        
        x0 = random.rand(16)
        for n in tqdm_notebook(range(averages), desc='Tomography: Likelihood minimization', ncols=700):
            new = minimize (self.likelihood, x0, args = (number), method='BFGS', tol= 1e-09, options = {'gtol': 1e-05,\
                                                                                                    'maxiter': 1000})
            if (n==0):
                best = new
            elif (new.fun < best.fun):
                best = new           
            x0 =best.x + 0.1*(random.rand(16)*2 - 1)
            print (best.fun, new.fun)
            
        return self.x_to_rho(best.x)
    
    def find_rho_all_parallel (self, averages, x_start = random.rand(16)):
        #with Pool(4) as p:
         #   self.rho_tomo = []
        find_rho_avg = partial(Tomography.find_rho, self, averages)
           # for item in tqdm(p.imap(find_rho_avg, range(len(self._2q_rotations)))):
            #    self.rho_tomo.append(item)
        self.rho_tomo = p_map (find_rho_avg, range(len(self._2q_rotations)), num_cpus = 20)
        
    def find_rho_all (self, averages, x_start = random.rand(16)):
        self.rho_tomo = []
        for number in tqdm(range(len(self._2q_rotations))):
            self.rho_tomo.append(self.find_rho(averages, number))
   
    @staticmethod
    def x_to_rho(x):                                # Density matrix parametrization via Choletsky decomposition
        dim = int(sqrt(len(x)))
        t = np.identity(dim, complex)
        for i in range(dim):
            t[i, i] = abs(x[i])
        k = dim
        for i in range(dim):
            for l in range(i):
                t[i, l] = x[k] + 1j * x[k + 1]
                k += 2
        q_dim = 2
        L = Qobj(t, dims=[[q_dim, q_dim],[q_dim, q_dim]])
        L1 = L.full()
        L2 = L.dag().full()
        L12=dot(L1,L2)
        rho = Qobj(L12,dims=[[q_dim, q_dim],[q_dim, q_dim]])
        rho = rho / rho.tr()
        return rho
    @staticmethod
    def rho_to_rho3dim(rho):
        rho_3dim = np.zeros((9,9), dtype = complex)
        rho_tr = rho
        rho_3dim[0,0]=rho_tr[0,0]
        rho_3dim[0,1]=rho_tr[0,1]
        rho_3dim[1,0]=rho_tr[1,0]
        rho_3dim[1,1]=rho_tr[1,1]
        rho_3dim[3,0]=rho_tr[2,0]
        rho_3dim[3,1]=rho_tr[2,1]
        rho_3dim[4,0]=rho_tr[3,0]
        rho_3dim[4,1]=rho_tr[3,1]
        rho_3dim[0,3]=rho_tr[0,2]
        rho_3dim[0,4]=rho_tr[0,3]
        rho_3dim[1,3]=rho_tr[1,2]
        rho_3dim[1,4]=rho_tr[1,3]
        rho_3dim[3,3]=rho_tr[2,2]
        rho_3dim[4,3]=rho_tr[3,2]
        rho_3dim[3,4]=rho_tr[2,3]
        rho_3dim[4,4]=rho_tr[3,3]
        return Qobj(rho_3dim, dims = [[3,3],[3,3]])
    @staticmethod
    def rho3dim_to_rho(rho3dim):
        rho_tr = np.zeros((4,4), dtype = complex)
        rho_3dim = rho3dim
        rho_tr[0,0] = rho_3dim[0,0]
        rho_tr[0,1] = rho_3dim[0,1]
        rho_tr[1,0] = rho_3dim[1,0]
        rho_tr[1,1] = rho_3dim[1,1]
        rho_tr[2,0] = rho_3dim[3,0] 
        rho_tr[2,1] = rho_3dim[3,1]
        rho_tr[3,0] = rho_3dim[4,0]
        rho_tr[3,1] = rho_3dim[4,1]
        rho_tr[0,2] = rho_3dim[0,3]
        rho_tr[0,3] = rho_3dim[0,4]
        rho_tr[1,2] = rho_3dim[1,3]
        rho_tr[1,3] = rho_3dim[1,4]
        rho_tr[2,2] = rho_3dim[3,3]
        rho_tr[3,2] = rho_3dim[4,3]
        rho_tr[2,3] = rho_3dim[3,4]
        rho_tr[3,3] = rho_3dim[4,4]
        return Qobj(rho_tr, dims = [[2,2],[2,2]])
            
    
        
    @staticmethod
    def rho_to_x(rho):
        if rho is None:
            return None
        rho = rho*rho.dag()
        ps_rho = rho + 1e-12                                            # wtf KOSTYL'
        L_ar = (Qobj(cholesky(ps_rho.full() ))).dag().full()             # dont touch
        L_dim = L_ar.shape[0]
        x = np.zeros(L_dim ** 2)
        k = L_dim
        for i in range(L_dim):
            x[i] = L_ar[i, i]
            for j in range(i):
                x[k] = real(L_ar[i, j])
                x[k + 1] = imag(L_ar[i, j])
                k += 2
        return x         
    @staticmethod
    def x_to_chi(x):
        dim = int(sqrt(len(x)))
        t = np.identity(dim, complex)
        for i in range(dim):
            t[i, i] = abs(x[i])
        k = dim
        for i in range(dim):
            for l in range(i):
                t[i, l] = x[k] + 1j * x[k + 1]
                k += 2
        q_dim = 4
        L = Qobj(t, dims=[[q_dim, q_dim],[q_dim, q_dim]])
        L1 = L.full()
        L2 = L.dag().full()
        L12=dot(L1,L2)
        #chi = Qobj(L12)
        chi = L12                        
        return chi 
    
    @staticmethod
    def state_fidelity (rho_ideal, rho_predicted):
        return np.trace(sqrtm(dot( sqrtm(rho_ideal),dot( rho_predicted, sqrtm(rho_ideal) ) ) ) )**2
      
    def chi_to_R (self, chi):
        R = np.zeros_like(chi)
        d = 4
        pauli = self._pauli_mat_2qubits
        for i in range(d**2):
            for j in range (d**2):
                for k in range (d**2):
                    for l in range (d**2):
                        R[i,j]+=chi[k,l]* np.trace(dot(pauli[i],dot(pauli[k],dot(pauli[j],pauli[l]))))
                R[i][j]/=d  
        return R
      
      
    def get_R_iswap_ideal (self):
    	pauli = self._pauli_mat_2qubits
    	iswap_ideal = np.array ([[1,0,0,0],[0,0,-1j,0],[0,-1j,0,0],[0,0,0,1]])
    	R_ideal = np.empty((16,16))
    	for i in range (16):
    		for j in range (16):
        		image = dot(iswap_ideal,dot(pauli[j],iswap_ideal.transpose().conj()))
        		R_ideal[i][j]=1/4*np.trace(dot(pauli[i], image))
    	return R_ideal
    
    
    def get_R_4iswap_ideal (self):
    	pauli = self._pauli_mat_2qubits
    	iswap_ideal = np.array ([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    	R_ideal = np.empty((16,16))
    	for i in range (16):
    		for j in range (16):
        		image = dot(iswap_ideal,dot(pauli[j],iswap_ideal.transpose().conj()))
        		R_ideal[i][j]=1/4*np.trace(dot(pauli[i], image))
    	return R_ideal
    
    def get_R_2iswap_ideal (self):
    	pauli = self._pauli_mat_2qubits
    	iswap_ideal = np.array ([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    	R_ideal = np.empty((16,16))
    	for i in range (16):
    		for j in range (16):
        		image = dot(iswap_ideal,dot(pauli[j],iswap_ideal.transpose().conj()))
        		R_ideal[i][j]=1/4*np.trace(dot(pauli[i], image))
    	return R_ideal
    def get_R_cphase_ideal (self, phi):
    	pauli = self._pauli_mat_2qubits
    	cphase_ideal = np.array ([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,exp(1j*phi)]])
    	R_ideal = np.empty((16,16))
    	for i in range (16):
    		for j in range (16):
        		image = dot(cphase_ideal,dot(pauli[j],cphase_ideal.transpose().conj()))
        		R_ideal[i][j]=1/4*np.trace(dot(pauli[i], image))
    	return R_ideal
    
    def get_R_2iswap_phi (self, phi):
    	pauli = self._pauli_mat_2qubits
    	iswap2_phi = np.array ([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,exp(-2j*phi)]])
    	R_ideal = np.empty((16,16))
    	for i in range (16):
    		for j in range (16):
        		image = dot(iswap2_phi,dot(pauli[j],iswap2_phi.transpose().conj()))
        		R_ideal[i][j]=1/4*np.trace(dot(pauli[i], image))
    	return R_ideal
    
    def R_fsim_like(self, theta, phi, delta_off, delta1, delta2):
        pauli = self._pauli_mat_2qubits
        Gate_matrix = np.array([[1,0,0,0],[0, cos(theta)*exp(1j*(delta1+delta2)), -1j*sin(theta)*exp(1j*(delta1-delta_off)),0],
                                [0,-1j*sin(theta)*exp(1j*(delta1+delta_off)), cos(theta)*exp(1j*(delta1-delta2)),0],
                                [0,0,0,exp(1j*(2*delta1+phi))]])
        
        images = []
        for j in range (16):
            image = dot(Gate_matrix,dot(pauli[j], Gate_matrix.transpose().conj()))
            images.append(image)
        R_gate=np.empty((16,16))
        for i in range (16):
            for j in range (16):
                R_gate[i][j]=1/4*np.trace(dot(pauli[i], images[j]))
        return R_gate
    
    
    def likelihood_R (self,x,R):
        R_gate = self.R_fsim_like(x[0],x[1],x[2],x[3], x[4])
        diff = R - R_gate
        return norm (diff, ord = 'fro')
        
        
    
    def Fidelity (self, R, R_ideal):
        fidelity = 1.0/20*(np.trace(dot(inv(R_ideal),R)) + 4)
        return fidelity
    
    
    def fidelity_function_zgate (self, params, number, iswap2, cphase): # error of i-swap for pi/2 Y 2nd qubit rotation 
                                                   #its necessary to find additional phase of first qubit
        self._params = params
        if iswap2:
            rho_predicted, rho_ideal = self.find_iswap2_rotation_matrix(number)
        elif cphase:
            rho_predicted, rho_ideal = self.find_cphase_rotation_matrix(number)            
        else:    
            rho_predicted, rho_ideal = self.find_iswap_rotation_matrix(number)
        return Tomography.state_fidelity(rho_ideal, rho_predicted)
    
    def phase_zgate(self, t_zgate_1, t_zgate_2, t_start, number):
        self._params["t_zgate"] = t_zgate_1
        self._params["t_zgate2"] = t_zgate_2
        self._params["start"] = t_start
        dm, dm_final = self.run_iswap_test_step_parallel(self._2q_rotations[number],False,True)
        qubit1_z_interaction = expect(self.build_2qubit_operator('Z','id'), dm)
        qubit1_y_interaction = expect(self.build_2qubit_operator('Y','id'), dm)
        qubit1_x_interaction = expect(self.build_2qubit_operator('X','id'), dm)
        qubit2_z_interaction = expect(self.build_2qubit_operator('id','Z'), dm)
        qubit2_y_interaction = expect(self.build_2qubit_operator('id','Y'), dm)
        qubit2_x_interaction = expect(self.build_2qubit_operator('id','X'), dm)
        phi1s = np.arctan2(np.real(qubit1_y_interaction), np.real(qubit1_x_interaction))
        phi2s = np.arctan2(np.real(qubit2_y_interaction), np.real(qubit2_x_interaction)) 
        return unwrap(phi1s, period = pi), unwrap(phi2s, period = pi)
        
        

    def error_function (self, n):      #gives an error of iswap with n_th rotation
        #self._phase1_offset = phi1
        #self._phase2_offset = phi2
        rho_predicted, rho_ideal = self.find_iswap_rotation_matrix(n)
        delta = (rho_predicted - rho_ideal)
        return norm(delta, ord = 'fro')
    
    
    def find_errors_array (self):
        phi1s = linspace (0, 2*pi, 40)
        phi2s = linspace (0, 2*pi, 40)
        errors_1_array =  p_map(Tomography.error_function_1, [self]*len(phi1s), phi1s, [0]*len(phi1s), num_cpus = 20)
        errors_2_array =  p_map(Tomography.error_function_2, [self]*len(phi2s), [0]*len(phi2s),  phi2s, num_cpus = 20)
        return errors_1_array, errors_2_array
    
    
    def find_appropriate_phase(self):
        phi1s = linspace (0, 2*pi, 40)
        phi2s = linspace (0, 2*pi, 40)
        errors_1_array =  p_map(Tomography.error_function_1, [self]*len(phi1s), phi1s, [0]*len(phi1s), num_cpus = 20)
        errors_2_array =  p_map(Tomography.error_function_2, [self]*len(phi2s), [0]*len(phi2s),  phi2s, num_cpus = 20)
        X = errors_1_array
        Y = errors_2_array
        Z = np.empty((len(Y), len(X)))
        for i in range (len(Y)):
            for j in range (len(X)):
                Z[i,j] = Y[i]**2 + X[j]**2
                
        f = interp2d(phi1s,phi2s,Z)
        
        def g (phi_array):
            return f(phi_array[0], phi_array[1])[0]
        result = minimize (g, array([1,1]))
        phi1 = result.x[0]
        phi2 = result.x[1]
        return phi1, phi2
           
                
                
                
                
