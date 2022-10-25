import sys
sys.path.append('/p/home/jusers/bruemmer1/juwels/my_project/software/APLTS')
from APLTS import ElectronBunch,Laser,ThomsonScattering,ActivePlasmaLens


Bunch = ElectronBunch.ElectronBunch(_gammae=120.0,_emittance=0.350e-6,_sigma_r=3.2e-6,_Q=10e-12,_sigma_l=1.5e-6,_sigma_div=None,_Delta_gammae=0.2)

ThomsonLaser= Laser.Laser(_wavelength=800.0e-9,_Ep=0.1,_a0=None,_w0=1.22e-6,_tau_FWHM=9.64e-13,_Delta_lambda=None)

APL=ActivePlasmaLens.APL_setup(L=0.04,I_0=778,z_0=7.4e-2,r_0=1e-3,gammae=Bunch._gammae,eps_n=Bunch._emittance,sigmar_i=Bunch._sigma_r)
ThomsonBunch=APL.return_focusedBunch()
ThomsonBunch._Q=0.1e-12*0.2*0.511*ThomsonBunch._gammae #charge per MeV * BWrel * electronenergy

print("simulated charge is "+str("%.2f" %(ThomsonBunch._Q*1e12))+" pC")

ICS = ThomsonScattering.ThomsonScattering(ThomsonLaser,ThomsonBunch)

print("Full photon number "+str("%.2e" %ICS.Photons_tot()))
print("PN in cone of pm0.5 mrad " +str("%.2e" %ICS.Photons_cone(0.5e-3)))
print("PN in cone of pm0.5 mrad and 15% BW " +str("%.2e" %ICS.Photons_cone(0.5e-3,BWlim=0.15)))
