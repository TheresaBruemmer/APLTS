"""
APL+Thomson Yield Optimiser within given BW and opening angle

Prints out results. 

Example printout:

number of cores=64
opt run0
tau array:1e-12 to 1.5e-11
opt run1
tau array:1e-12 to 2.5555555555555553e-12
opt run2
tau array:1e-12 to 1.1728395061728396e-12
opt run3
tau array:1e-12 to 1.0192043895747599e-12
No laser bandwidth defined, Fourier-limited bandwidth calculated
---------------------------
Simulation Parameters
a0 = 0.9
gamma = 87.10306859267975
Ekin = 43.99866805085935 eV
minimum tau was fixed to 1e-12 s
tau=1.0000e-12 s
w0=3.384e-06 m
sigmar=2.106e-06 m
I0 = 558.81 A
zF = 0.321 m
check sigmar_f = 2.106 um
Zstart=2.1177063102143947e-05
L_H = 2.1460454808952732e-08
eff enspread = 0.190 %
simulate sig_Ekin = 0.09746637968754784 eV

"""
import sys
sys.path.append("/home/bruemt/code/APLTS/")
from APLTS.utilities.physical_constants import *
import APLTS.utilities.Astra_tools
import APLTS.ActivePlasmaLens as APL
from APLTS import APLTS,Laser,ThomsonScattering
from APLTS import ThomsonScatteringTools as TST

import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
import mpmath as mp
import math
#parallel calculation
from joblib import Parallel, delayed
import multiprocessing
#
printall="False"
num_cores = multiprocessing.cpu_count()
print("number of cores="+str(num_cores))

"""
Laser and Electron Bunch Parameters
"""
a0=0.9
Ep=0.2
llambda=1030.e-9
taumin_fix=1e-18#should be >0, better: realistic value, tau=0 will throw runtime erro 
taumin=1.0e-12
taumax=1.0e-12
#
eps_n=1.0e-6
sigmar_i=1.0e-6
sigma_el=1.5e-6
sigmamin=1.e-6
sigmamax=20.e-6
Q=10.e-12 # just some ballpark number

'''
TS and X-ray Parameters
'''
Etarget=26e3
alpha_coll = np.pi
BWlim=0.15
theta_c = 4.0e-3

'''
Some hard coded APL data
'''
L_APL=0.1
z_0=5.0e-2
r_APL=2.0e-3
I0_min=100.0
I0_max=1500.0

#number of opt runs for APL input
Nrun=10
#TS sampling steps per wavelength
NUM=80


EL=h*c_light/llambda/e_charge
Nl=Ep/(h*c_light/llambda)
gamma_e=sqrt(Etarget/4/EL*(1+a0**2/2))


besttau,bestw0,bestsigmar,bestNgamma=TST.Optimise(sigmamin,sigmamax,taumin,taumax,llambda,Ep,a0,gamma_e,Q,eps_n,sigma_el,theta_c,printall,taumin_fix,BWlim,N_run=3)
I0,zF,check_Waist = APL.FindAPLConfig_fixed_focalwaist(gamma_e,sigmar_i,eps_n,bestsigmar,L_APL,r_APL,z_0,I0_min,I0_max,Nrun)
zstart=Astra_tools.Laser_zstart(gamma_e,zF)
APL_instance=APL.APL_setup(L=L_APL,I_0=I0,z_0=z_0,r_0=r_APL,gammae=gamma_e,eps_n=eps_n,sigmar_i=sigmar_i)
Laser_instance=Laser.Laser(_wavelength=llambda,_Ep=Ep,_a0=a0,_tau_FWHM=besttau)
eff_enspread=APLTS.effective_energyspread_fixedsetup(Laser_instance,APL_instance)[0]
Ekin,sig_Ekin, delta_gammae= Astra_tools.generatorEkin(gamma_e,eff_enspread)


print("---------------------------")
print("Simulation Parameters")
print("For a0 = "+str(a0))
print("gamma = "+str("%.3f" %gamma_e))
print("Ekin = "+str("%.3e" %Ekin) + " eV")
if taumin_fix>0:
    print("minimum tau was fixed to "+str(taumin_fix)+" s")
print("tau="+str("%.4e" %(besttau))+" s")
print("w0="+str("%.3e" %(bestw0))+ " m")
print("sigmar="+str("%.3e" %(bestsigmar))+" m")
print("I0 = " + str("%.2f" %I0)+" A")
print("zF = " + str("%.3f" %zF)+" m")
print("check sigmar_f = "+str("%.3f" %(check_Waist*1e6))+" um")
print("Zstart="+ str("%.3e" %zstart))
print("L_H = " + str("%.3e" %(Astra_tools.L_H(llambda,alpha_coll,gamma_e,NUM))))
print("eff enspread = " + str("%.3f" %(100*eff_enspread)) + " %")
print("simulate sig_Ekin = " + str("%.3e" %sig_Ekin) + " eV")
print("---------------------------")


