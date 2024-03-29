"""
This script returns active plasma lens parameters for a given target electron energy and focal size
"""

import sys
sys.path.append("/home/bruemt/code/APLTS/")
import APLTS.ActivePlasmaLens as APL
import numpy as np

'''
Input on calling this function
'''
gammae=float(sys.argv[1]) # electron (central) energy
sigmar_f=float(sys.argv[2]) # required electron focal waist (m)

'''
Output:
I_0   # required APL current (A)
z_F   # resulting focal plane (m)
'''


'''
Some hard coded APL data
'''
L_APL=0.1        # APL length (m)
z_0=5.0e-2       # APL position, i.e. beginning of APL field
r_APL=2.0e-3     # APL radius
eps_n=1.0e-6     # electron bunch normalised emittance (m x rad) 
sigmar_i=1.0e-6  # initialelectron bunch waist at z=0 (in focus after plasma acceleration)
I0_min=100.0     # minimum of the APL current range
I0_max=1500.0    # maximum of the APL current range


'''
We want to do six runs to find the optimum.
Each time we close in more on the opt values
'''

for opt_run in range(0,7):
    I0_arr = np.linspace(I0_min,I0_max,10)
    delta_I0=I0_arr[1]-I0_arr[0]
    instances=APL.APL_setup(L=L_APL,I_0=I0_arr,z_0=z_0,r_0=r_APL,gammae=gammae,eps_n=eps_n,sigmar_i=sigmar_i)
    match_arg = np.nanargmin(abs(instances.focalWaist()-sigmar_f))
    I0=I0_arr[match_arg]
    zF=instances.focalPlane()[match_arg]
    check_Waist=instances.focalWaist()[match_arg]
    I0_min=I0-2*delta_I0
    I0_max=I0+2*delta_I0
print("---------------------------")
print("---Simulation Parameters---")
print("I0 = " + str("%.2f" %I0)+" A")
print("zF = " + str("%.3f" %zF)+" m")
print("check sigmar_f = "+str("%.3f" %(check_Waist*1e6))+" um")
print("---------------------------")


