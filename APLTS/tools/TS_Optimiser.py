"""
Thomson Yield Optimiser within given BW and opening angle

Prints out results. 

Example printout:

a0 = 0.9
gamma = 87.10306859267975
Ekin = 43.99866805085935 eV
minimum tau was fixed to 1e-12 s
tau=1.0000e-12 s
w0=3.384e-06 m
sigmar=2.106e-06 m
"""
import sys
sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")
from APLTS.utilities.physical_constants import *
import APLTS.ActivePlasmaLens as APL
from APLTS import APLTS,Laser,ThomsonScattering
from APLTS import ThomsonScatteringTools as TST

import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
import mpmath as mp
import math
#
printall="False"
#
"""
Laser and Electron Bunch Parameters
"""
a0=0.9
Ep=0.2
llambda=1030.e-9
taumin_fix=0#1e-12 
taumin=1.0e-12#if laser duration is fixed, set taumin=taumax. Optimisation will then exclude tau
taumax=10.e-12
#
eps_n=1.0e-6
sigmar_i=1.0e-6
sigma_el=1.5e-6
sigmamin=1.e-6
sigmamax=20.e-6
Q=10.e-12 # just some ballpark number required to calculate number of photons

'''
TS and X-ray Parameters
'''
Etarget=26e3
alpha_coll = np.pi
BWlim=0.15 
theta_c = 4.0e-3

########################################
EL=h*c_light/llambda/e_charge
Nl=Ep/(h*c_light/llambda)
gamma_e=sqrt(Etarget/4/EL*(1+a0**2/2))
########################################

besttau,bestw0,bestsigmar,bestNgamma=TST.Optimise(sigmamin,sigmamax,taumin,taumax,llambda,Ep,a0,gamma_e,Q,eps_n,sigma_el,theta_c,printall,taumin_fix,BWlim,N_run=3)

print("---------------------------")
print("Target photon energy = "+str(Etarget/1e3) + " keV")
print("within " + str(BWlim*100) + " % FWHM")
print("and collimation angle = " + str(theta_c*1e3) + " mrad") 
print("---------------------------")
print("For a0 = "+str(a0))
print("gamma = "+str("%.3f" %gamma_e))
if taumin_fix>0:
    print("minimum tau was fixed to "+str(taumin_fix)+" s")
    print("opt. tau = "+str("%.4e" %(besttau))+" s")
elif taumin==taumax:
    print("tau was fixed to "+str(taumin)+" s")
else:
    print("opt. tau = "+str("%.4e" %(besttau))+" s")
print("w0="+str("%.3e" %(bestw0))+ " m")
print("sigmar="+str("%.3e" %(bestsigmar))+" m")
print("---------------------------")


