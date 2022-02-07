import two_transmons.DoubleTransmonSystem
from two_transmons.DoubleTransmonSystem import *
import two_transmons.VacuumRabiSimulation
from two_transmons.VacuumRabiSimulation import *
import single_transmon.Transmon
from single_transmon.Transmon import *
from ReadoutResonator import *
from qutip import *


def calculate (phi):
    Nc = 7
    Ec1 = 0.25 * 2 * np.pi
    Ec2 = 1.03 * Ec1
    Ecc = 0.01 / 1.2 ** 2 * 2 * np.pi  # 1.2 is my estimate for n_{ge} matrix element
    Ej1 = Ec1 * 100
    Ej2 = Ec2 * 100
    d = 0.2

    T1_1, T2_1 = 10e3, 5e3
    T1_2, T2_2 = 5e2, 1e2

    tr1 = Transmon(Ec1, Ej1, d, 1 / T1_1, 1 / T2_1, Nc, 3, 1)
    tr2 = Transmon(Ec2, Ej2, d, 1 / T1_2, 1 / T2_2, Nc, 3, 2)
    dts = DoubleTransmonSystem(tr1, tr2, Ecc)
    r = ReadoutResonator(6, 1118, 1964, phi=0.29)
    r.set_qubit_parameters(.05, 0.05, 5, 5.3, .2, .2)
    Ts = linspace(0, params["finish"], params['t_points']) 
    params1=params
    params1['phi_offset']=phi
    VRS = VacuumRabiSimulation(dts, Ts, params1, r)
    result = VRS.run()

    state1 = VRS._rho0
    state2 = VRS._dts.e_state(1 / 2, 1 / 2, 2)
    state2 = state2 * state2.dag()
    ar1 = []
    ar2 = []

    for state in result.states:
        if len(VRS._c_ops) == 0:
            state = state * state.dag()
        ar1.append(expect(state, state1))
        ar2.append(expect(state, state2))
    ret = {'projections1': ar1, 'projections2': ar2}
    return ret
def f1 (x):
    ret= {'r1' : x*x,'r2': x}
    return ret



