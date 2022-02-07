from single_transmon.Transmon import *
from qutip import *
from itertools import product
from numpy import zeros, array, where, ones_like
from matplotlib.pyplot import *

class TransmonChain:

    def __init__(self, length, transmon_truncation=2,
                 transmon_diagonalization_dimension=15,
                 bose_hubbard_approximation = False):
        self._length = length
        self._transmon_truncation = transmon_truncation
        self._transmon_diagonalization_dimension = transmon_diagonalization_dimension
        self._bose_hubbard_approximation = bose_hubbard_approximation
        self._Ec = []
        self._Ej = []
        self._J = []
        self._d = []
        self._Omega = []
        self._omega = None
        self._gamma_phi = []
        self._gamma_rel = []
        self._phi = []
        self._c_ops = None
        self._transmons = [None] * self._length
        self._zero_op = tensor(*[qzero(self._transmon_truncation) for i in range(0, self._length)])
        self._identity_array = [qeye(self._transmon_truncation) for i in range(0, self._length)]

        self._RWA_driving_cache = {}
        self._RWA_subtrahend_cache = {}
        self._transmon_H_cache = [{} for _ in range(self._length)]
        self._interaction_cache = {}

        self._low_energy_states = []
        self._low_energy_state_indices = []
        
    def set_e_ops(self):
        sigma_z_chain = []
        sigma_x_chain = []
        sigma_y_chain = []
        for i in range (self._length): #TO DO - вынести в функцию класса
            sigma_z = self._transmons[i].sz()
            sigma_x = self._transmons[i].sx()
            sigma_y = self._transmons[i].sy()
            oper_z = self._identity_array.copy()
            oper_z[i]=sigma_z
            oper_x = self._identity_array.copy()
            oper_x[i]=sigma_x
            oper_y = self._identity_array.copy()
            oper_y[i]=sigma_y
            sigma_z_chain.append(self.truncate_to_low_population_subspace(tensor(*oper_z)))
            sigma_x_chain.append(self.truncate_to_low_population_subspace(tensor(*oper_x)))
            sigma_y_chain.append(self.truncate_to_low_population_subspace(tensor(*oper_y)))
        self.e_ops = sigma_z_chain + sigma_x_chain + sigma_y_chain
        

        

    def clear_caches(self):
        self._RWA_driving_cache = {}
        self._RWA_subtrahend_cache = {}
        self._transmon_H_cache = [{} for _ in range(self._length)]
        self._interaction_cache = {}
        self._c_ops = None

    def build_low_energy_kets(self, total_population, max_simultaneously_above_first_excited):
        self._low_energy_states = []
        self._low_energy_state_indices = []
        transmon_states = [list(range(self._transmon_truncation)) for i in range(self._length)]
        transmon_states = product(*transmon_states)
        
        idx = 0
        for idx, state_combination in enumerate(transmon_states):
            state_combination = array(state_combination)
            if sum(state_combination) <= total_population:
               # if len(where(state_combination >= 2)[0]) <= max_simultaneously_above_first_excited:
                self._low_energy_states.append(state_combination)
                self._low_energy_state_indices.append(idx)
           
        self._low_energy_states_mask = zeros([idx+1]*2, dtype=bool)
        for idx1 in self._low_energy_state_indices:
            for idx2 in self._low_energy_state_indices:
                self._low_energy_states_mask[idx1, idx2] = True
        
        print("Total %d kets included" % len(self._low_energy_states))

    def build_H_RWA(self):
        return self.build_H_full() - self.build_RF_subtrahend()

    def build_H_full(self, waveforms, params, rotations):
        H_chain = []

        for i in range(0, self._length): #single transmon part
            H_chain += self._build_transmon_H_at_index(i, waveforms[i])
        for i in range(0, self._length - 1): #transmons interaction
            evals1, evecs1 = self._transmons[i].H_diag_trunc_approx(self._phi[i]).eigenstates()
            evals2, evecs2 = self._transmons[i+1].H_diag_trunc_approx(self._phi[i+1]).eigenstates()
            omega1 = (evals1[1]- evals1[0])
            omega2 = (evals2[1]- evals2[0])
            H_chain += self.Hint_RF_RWA(i, i + 1, omega2 - omega1)
        H_chain += self.build_RWA_driving(params,rotations)
        return H_chain

    def build_c_ops(self):  # building at 0 flux since no real dependence on anything
        if self._c_ops is not None:
            return self._c_ops

        self._c_ops = []

        for i in range(self._length):
            chain_operator_rel = self._identity_array.copy()
            chain_operator_rel[i] = self._transmons[i].c_ops(0)[0]
            chain_operator_deph = self._identity_array.copy()
            
            chain_operator_deph[i] = self._transmons[i].c_ops(0)[1]
            self._c_ops.append(self.truncate_to_low_population_subspace(tensor(*chain_operator_rel)))
            self._c_ops.append(self.truncate_to_low_population_subspace(tensor(*chain_operator_deph)))
        return self._c_ops

    def truncate_to_low_population_subspace(self, operator):
        return Qobj(operator.data[self._low_energy_states_mask].reshape((len(self._low_energy_state_indices), -1)))

    def build_RWA_driving(self, params, rotations):
        driving_operator = self._zero_op.copy()
        H_drive_full = []
        for i in range(0, self._length):
            dur, phi = rotations[i]
            Hdr, Hdr_coeff = self._transmons[i].Hdr_RF_RWA(params['drive_amplitude'], params['start'] - dur - 10, dur, phi)
            chain_operator = self._identity_array.copy()
            chain_operator[i] = Hdr
            chain_operator = tensor(*chain_operator)
            Hdr_trunc = self.truncate_to_low_population_subspace(chain_operator)
            H_drive_full +=[[Hdr_trunc,Hdr_coeff]] 
        return H_drive_full

    '''
    def build_RF_subtrahend(self):
        try:
            return self._RWA_subtrahend_cache[self._omega]
        except:
            subtrahend = self._zero_op.copy()
            for i in range(0, self._length):
                chain_operator = self._identity_array.copy()
                chain_operator[i] = self._omega * ket2dm(self._transmons[i].e_state()) + \
                                    self._omega * 2 * ket2dm(
                    self._transmons[
                        i].f_state()) if self._transmon_truncation is 3 else self._omega * ket2dm(
                    self._transmons[i].e_state())
                subtrahend += tensor(*chain_operator)
            self._RWA_subtrahend_cache[self._omega] = subtrahend
            return subtrahend
    '''
    def _build_transmon_H_at_index(self, i, waveform):
        
        chain_operator = self._identity_array.copy()
        H1 = self._identity_array.copy()
        H2 = self._identity_array.copy()
        Huu = self._identity_array.copy()
        [H1[i],H1_coeff], [H2[i], H2_coeff] = self._transmons[i].H_td_diag_trunc_approx(waveform)
        #self._transmon_H_cache[i][self._phi[i]] = tensor(*chain_operator)
        H1 = tensor(*H1)
        H2 = tensor(*H2)
        evals1, evecs1 = self._transmons[i].H_diag_trunc_approx(self._phi[i]).eigenstates()
        freq1 = (evals1[1]- evals1[0])/2/pi
        Huu[i] = self._transmons[i].H_uu(freq1)*(-1)
        Huu = tensor(*Huu)
        Huu_trunc = self.truncate_to_low_population_subspace(Huu)
        H1_trunc = self.truncate_to_low_population_subspace(H1)
        H2_trunc = self.truncate_to_low_population_subspace(H2)
        
        return [[H1_trunc,H1_coeff]] + [[H2_trunc, H2_coeff]] + [[Huu_trunc, ones_like(H1_coeff)]]
        
    def plot_chain_dynamic(self, result):
        Ts = self._Ts
        fig, axes = subplots (3,1, figsize = (7,6))
        labels = ['z', 'x', 'y']
        for ind in range (3):            
            for i in range (self._length):
                axes[ind].plot(Ts, result.expect[i + ind*self._length], label = labels[ind] + str(i))       
        for ax in axes:
            ax.set_ylim((-1.05,1.05))
            ax.legend()
            ax.grid()
        return fig, axes
        
        
    
    def Hint_RF_RWA(self, i, j, offset):
        chain_operator1 = self._identity_array.copy()      
        chain_operator2 = self._identity_array.copy()
        op1 = self._transmons[i].n(self._phi[i]).data.copy()
        op2 = self._transmons[j].n(self._phi[j]).data.copy()
        # zeroing triangles
        #print (op1)
        #print (op2)
        for k in range(op1.shape[0]):
            for l in range(op1.shape[1]):
                if k < l:
                    op1[k, l] = 0
                if k > l:
                    op2[k, l] = 0
        chain_operator1[i] = Qobj(op1, dims = [[3],[3]])
        chain_operator1[j] = Qobj(op2, dims = [[3],[3]])
        half = tensor(*chain_operator1) 
        #print (half)
        half_trunc = self.truncate_to_low_population_subspace(half)
        return [[self._J[i] * half_trunc, "exp(-1j*(%f)*t)"%offset],[self._J[i] * half_trunc.dag(), "exp(1j*(%f)*t)"%offset]]

    '''
    def _build_interaction_RWA(self, i, j):
        try:
            return self._interaction_cache[(i, j)][(self._phi[i], self._phi[j])]
        except:
            chain_operator1 = self._identity_array.copy()
            
            chain_operator1[i] = self._transmons[i].raising(self._phi[i])
            chain_operator1[j] = self._transmons[j].lowering(self._phi[j])

            chain_operator2 = self._identity_array.copy()
            chain_operator2[i] = self._transmons[i].lowering(self._phi[i])
            chain_operator2[j] = self._transmons[j].raising(self._phi[j])

            if (i,j) not in self._interaction_cache:
                self._interaction_cache[(i,j)] = {}
            self._interaction_cache[(i,j)][(self._phi[i], self._phi[j])] = \
                self._J[i] * (tensor(*chain_operator1) + tensor(*chain_operator2))
            return self._interaction_cache[(i,j)][(self._phi[i], self._phi[j])]
    '''    
        

    def build_transmons(self):
        for i in range(self._length):
            self._transmons[i] = Transmon(self._Ec[i], self._Ej[i], self._d[i],
                                          self._gamma_rel[i], self._gamma_phi[i],
                                          self._transmon_diagonalization_dimension,
                                          self._transmon_truncation, i,
                                          self._bose_hubbard_approximation)

    def get_length(self):
        return self._length

    def set_Ec(self, Ec):
        try:
            assert len(Ec) == self._length
            self._Ec = Ec
        except TypeError:
            self._Ec = [Ec] * self._length
            print("Setting all Ec to be equal")
        except AssertionError:
            raise ValueError("Length of Ec is not equal to the number of transmons")

    def set_Ej(self, Ej):
        try:
            assert len(Ej) == self._length
            self._Ej = Ej
        except TypeError:
            self._Ej = [Ej] * self._length
            print("Setting all Ej to be equal")
        except AssertionError:
            raise ValueError("Length of Ej is not equal to the number of transmons")

    def set_J(self, J):
        try:
            assert len(J) == self._length
            self._J = J
        except TypeError:
            self._J = [J] * self._length
            print("Setting all J to be equal")
        except AssertionError:
            raise ValueError("Length of J is not equal to the number of transmons")

    def set_Omega(self, Omega):
        try:
            assert len(Omega) == self._length
            self._Omega = Omega
        except TypeError:
            self._Omega = [Omega] * self._length
            print("Setting all Omega to be equal")
        except AssertionError:
            raise ValueError("Length of Omega is not equal to the number of transmons")

    def set_omega(self, omega):
        self._omega = omega

    def set_asymmetry(self, d):
        try:
            assert len(d) == self._length
            self._d = d
        except TypeError:
            self._d = [d] * self._length
            print("Setting all d to be equal")
        except AssertionError:
            raise ValueError("Length of d is not equal to the number of transmons")

    def set_gamma_rel(self, gamma_rel):
        try:
            assert len(gamma_rel) == self._length
            self._gamma_rel = gamma_rel
        except TypeError:
            self._gamma_rel = [gamma_rel] * self._length
            print("Setting all gamma_rel to be equal")
        except AssertionError:
            raise ValueError("Length of gamma_rel is not equal to the number of transmons")

    def set_gamma_phi(self, gamma_phi):
        try:
            assert len(gamma_phi) == self._length
            self._gamma_phi = gamma_phi
        except TypeError:
            self._gamma_phi = [gamma_phi] * self._length
            print("Setting all gamma_phi to be equal")
        except AssertionError:
            raise ValueError("Length of gamma_phi is not equal to the number of transmons")

    def set_phi(self, phi):
        try:
            assert len(phi) == self._length
            self._phi = phi
        except TypeError:
            self._phi = [phi] * self._length
            print("Setting all fluxes to be equal")
        except AssertionError:
            raise ValueError(
                "Length of fluxes is not equal to the number of transmons: " + str(phi))