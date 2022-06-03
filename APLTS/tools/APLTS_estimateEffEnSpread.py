"""
Calculates the effective energy spread in an APLTS setup. Based on Brümmer et al. "Compact all-optical tunable narrowband Compton hard X-ray source", to be published
"""

import sys
sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")
from APLTS import APLTS,Laser
import APLTS.ActivePlasmaLens as APL



'''
Electron Bunch data
'''
gammae=65.0 #bunch central electron energy
eps_n=1.0e-6 # normalised bunch emittance
sigmar_i=1.0e-6 #initial bunch waist at plasma acceleration stage (PAS) exit
'''
Active Plasma Lens (APL) data
'''
L_APL=0.1 # APL length (m)
z_0=5.0e-2 # distance PAS exit to APL (m)
r_APL=2.0e-3 # APL radius (m)
I0=500 # APL current (A)
APL_instance=APL.APL_setup(L=L_APL,I_0=I0,z_0=z_0,r_0=r_APL,gammae=gammae,eps_n=eps_n,sigmar_i=sigmar_i) #initiate instance of APL setup 
'''
Laser data
'''
Ep=0.2 # laser pulse energy (J)
tau=1.0e-12 # laser pulse FWHM duration (s)
a0=0.1 # laser strength parameter
llambda=1030.0e-9 # laser wavelength (m)
Laser_instance=Laser.Laser(_wavelength=llambda,_Ep=Ep,_a0=a0,_tau_FWHM=tau) #initiate instance of Gaussian laser


eff_enspread=APLTS.effective_energyspread_fixedsetup(Laser_instance,APL_instance)[0]

print("Effective energy spread "+str("%.3f" %(eff_enspread*100)) + " %")
