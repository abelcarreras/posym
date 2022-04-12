from wfnsympy import WfnSympy
from posym import SymmetryBase
import pandas as pd

basis = {'name': 'STO-3G',
         'primitive_type': 'gaussian',
         'atoms': [{'symbol': 'O',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [130.70932, 23.808861, 6.4436083],
                                'con_coefficients': [0.154328969, 0.535328136, 0.444634536],
                                'p_con_coefficients': [0.0, 0.0, 0.0]},
                               {'shell_type': 'sp',
                                'p_exponents': [5.0331513, 1.1695961, 0.380389],
                                'con_coefficients': [-0.0999672287, 0.399512825, 0.700115461],
                                'p_con_coefficients': [0.155916268, 0.607683714, 0.391957386]}]},
                   {'symbol': 'H',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [3.42525091, 0.62391373, 0.1688554],
                                'con_coefficients': [0.154328971, 0.535328142, 0.444634542],
                                'p_con_coefficients': [0.0, 0.0, 0.0]}]},
                   {'symbol': 'H',
                    'shells': [{'shell_type': 's',
                                'p_exponents': [3.42525091, 0.62391373, 0.1688554],
                                'con_coefficients': [0.154328971, 0.535328142, 0.444634542],
                                'p_con_coefficients': [0.0, 0.0, 0.0]}]}]}

mo_coefficients = [[ 0.994216442, 0.025846814, 0.000000000, 0.000000000,-0.004164076,-0.005583712,-0.005583712],
                   [ 0.233766661,-0.844456594, 0.000000000, 0.000000000, 0.122829781,-0.155593214,-0.155593214],
                   [ 0.000000000, 0.000000000, 0.612692349, 0.000000000, 0.000000000,-0.449221684, 0.449221684],
                   [-0.104033343, 0.538153649, 0.000000000, 0.000000000, 0.755880259,-0.295107107,-0.295107107],
                   [ 0.000000000, 0.000000000, 0.000000000,-1.000000000, 0.000000000, 0.000000000, 0.000000000],
                   [-0.125818566, 0.820120983, 0.000000000, 0.000000000,-0.763538862,-0.769155124,-0.769155124],
                   [ 0.000000000, 0.000000000, 0.959800163, 0.000000000, 0.000000000, 0.814629717,-0.814629717]]

wf_results = WfnSympy(coordinates=[[ 0.0000000000, 0.0000000000, -0.0428008531],
                                   [-0.7581074140, 0.0000000000, -0.6785995734],
                                   [ 0.7581074140, 0.000000000, -0.6785995734]],
                      symbols=['O', 'H', 'H'],
                      basis=basis,
                      alpha_mo_coeff=mo_coefficients[:5],
                      alpha_occupancy=[0, 1, 1, 0, 0],
                      beta_occupancy=[1, 1, 1, 1, 1],
                      group='c2v')

wf_results.print_alpha_mo_IRD()
wf_results.print_overlap_mo_alpha()
wf_results.print_overlap_wf()
wf_results.print_wf_mo_IRD()


def get_orbital_state(orbital_soev):
    state_orb = SymmetryBase(group='c2v',
                             rep=pd.Series(orbital_soev, index=["E", "C2", "sv_xz", "sd_yz"])
                             )
    return state_orb


o1 = get_orbital_state(wf_results.mo_SOEVs_a[0])
o2 = get_orbital_state(wf_results.mo_SOEVs_a[1])
o3 = get_orbital_state(wf_results.mo_SOEVs_a[2])
o4 = get_orbital_state(wf_results.mo_SOEVs_a[3])
o5 = get_orbital_state(wf_results.mo_SOEVs_a[4])

print('\n')
for i, orbital in enumerate([o1, o2, o3, o4, o5]):
    print('Orbital {}: '.format(i+1), orbital)


print('\ntotal alpha: ', o2*o3)
print('total beta: ', o1*o2*o3*o4*o5)
print('Total WF: ', o2*o3*o1*o2*o3*o4*o5)