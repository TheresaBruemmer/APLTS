"""
----------------------------------------
Physical constants
----------------------------------------
"""
from numpy import pi,sqrt,log

alpha_f=1./137. # fine-structure constant
e_charge=1.6022e-19 # electron charge 
c_light=299792458 # speed of light
h=6.62607e-34 # Planck's constant
hbar_Js = 1.054572e-34
hbar_eVs = 6.58212e-16
eps_0=8.854e-12 # electric constant
m_e=9.109e-31 # electron mass
r_e=e_charge**2/(m_e*c_light**2)/(4.*pi*eps_0) # electron radius
sigma_T=8.*pi/3*r_e**2 # Thomson cross section 6.65e-29

E0 = 0.510998928e6; # 
mu0 = 4*pi*1e-7;
u0 = 0.067; 
mI = 0.106;

RMStoFWHM = 2.*sqrt(2.*log(2.))
FWHMtoRMS = 1.0/RMStoFWHM
