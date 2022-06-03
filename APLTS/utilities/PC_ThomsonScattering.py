"""
This module is not meant for the cluster, but for local calculations,hence the pc path definition
It contains the Rykovanov equations for a Thomson interaction

----------------------------------------
Path adjustment for different computers
----------------------------------------
"""
from sys import platform
#print(platform)
macpath = "/Users/theresa/"
linuxpath = "/home/bruemt/"
if platform=='linux':
    pcpath = linuxpath
else:
    pcpath = macpath
import sys
sys.path.append(pcpath+"ownCloud/Simulations/scripts/")
sys.path.append(pcpath+"ownCloud/Simulations/scripts/Modules/")

import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
from mpmath import mp
import math
import scipy
from LaserClass import constants
from LaserClass import Laser
#from scipy.integrate import tplquad
#from scipy.integrate import quad
#import scipy.integrate as integrate
mp.dps = 50

def BW_thetac(self,thetac):
    temp = self.gamma**2*thetac**2
    kappa = temp/(1+temp)
    return kappa

    
#number of photons emitted into the synchrotron angle
def Photons_tot(llambda,Ep,tau,w0,Q,sig_el,sig_et,eps_n,gammae):
    Nl=Ep/(constants.h*constants.c_light/llambda)
    Ne=Q/constants.e_charge
    sig_ll=(tau/(2.*sqrt(2.*log(2.))))*constants.c_light
    sig_lt=w0/2.
    sigma_l=np.sqrt(sig_el**2+sig_ll**2) # 535 um, mainly infl. by laser length
    beta_e=sig_et**2/(eps_n/gammae)
    beta_l=sig_lt**2/(llambda/(4.*np.pi))
    x=np.sqrt(2.)/sigma_l*np.sqrt((sig_et**2+sig_lt**2)/(sig_et**2/(beta_e**2)+sig_lt**2/(beta_l**2)))
    if type(x) is np.ndarray:
        F=np.array([mp.exp(xi**2)*(mp.erfc(xi)) for xi in x])   
    else:
        F=mp.exp(x**2)*(mp.erfc(x))
    N_gamma=constants.sigma_T*Ne*Nl*F/(np.sqrt(2*np.pi)*sigma_l*np.sqrt(sig_et**2+sig_lt**2)*np.sqrt(sig_et**2/beta_e**2+sig_lt**2/beta_l**2))
    return N_gamma

def Photons_cone(gammae,eps_n,sigma_e,div_e,Q,llambda,w0,theta_max,BW_max,tau,Ep,theta_steps):
    sig_el = 1.5e-6
    sig_lt=w0/2
    tt=np.linspace(0,theta_max,theta_steps)
    deltatheta=(tt[-1]-tt[0])/(len(tt)-1)
    yy=np.linspace(1-BW_max/2,1,1000)
    deltay=(yy[-1]-yy[0])/(len(yy)-1)
    sigma0=div_e#*FWHMtoRMS
    if type(sigma_e) is np.ndarray:
        spectrum=np.zeros((len(sigma_e),len(yy)))
        N=np.zeros(len(sigma_e))
        for ss,s in enumerate(div_e):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gammae**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gammae**2*y))/s**2)*theta*deltatheta
                N[ss]+=Photons_tot(llambda,Ep,tau,w0,Q,sig_el,sigma_e[ss],eps_n,gammae)*spectrum[ss][i]*deltay
               
    elif type(div_e) is np.ndarray:
        spectrum=np.zeros((len(div_e),len(yy)))
        N=np.zeros(len(div_e))
        for ss,s in enumerate(div_e):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gammae**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gammae**2*y))/s**2)*theta*deltatheta
                N[ss]+=Photons_tot(llambda,Ep,tau,w0,Q,sig_el,sigma_e,eps_n,gammae)*spectrum[ss][i]*deltay
    else:
        spectrum=np.zeros(len(yy))
        N=0
        s=div_e
        for i,y in enumerate(yy):
            for theta in tt:
                spectrum[i]+=3/(2*sigma0**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gammae**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gammae**2*y))/sigma0**2)*theta*deltatheta
            N+=Photons_tot(llambda,Ep,tau,w0,Q,sig_el,sigma_e,eps_n,gammae)*spectrum[i]*deltay
    return spectrum,N,deltay


def rel_Photons_cone(gammae,div_e,theta_max,BW_max,theta_steps):
    tt=np.linspace(0,theta_max,theta_steps)
    deltatheta=(tt[-1]-tt[0])/(len(tt)-1)
    yy=np.linspace(1-BW_max/2,1,1000)
    deltay=(yy[-1]-yy[0])/(len(yy)-1)
    sigma0=div_e#*FWHMtoRMS
    if type(div_e) is np.ndarray:
        spectrum=np.zeros((len(div_e),len(yy)))
        N=np.zeros(len(div_e))
        for ss,s in enumerate(div_e):
            if type(gammae) is np.ndarray:
                gam=gammae[ss]
            else: 
                gam=gammae
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gam**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gam**2*y))/s**2)*theta*deltatheta
                N[ss]+=spectrum[ss][i]*deltay
    elif type(gammae) is np.ndarray and type(div_e) is not np.ndarray:
        spectrum=np.zeros((len(gammae),len(yy)))
        N=np.zeros(len(gammae))
        for ig,gam in enumterate(gammae):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*sigma0**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gam**2*y)))/(2*sigma0**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gam**2*y))/sigma0**2)*theta*deltatheta
                N[ig]+=spectrum[ss][i]*deltay
    else:
        spectrum=np.zeros(len(yy))
        N=0
        s=div_e
        for i,y in enumerate(yy):
            for theta in tt:
                spectrum[i]+=3/(2*s**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gammae**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gammae**2*y))/s**2)*theta*deltatheta
            N+=spectrum[i]*deltay
    return N

def kappa(gammae,thetac):
    temp=gammae**2*thetac**2
    return temp/(1+temp)
def sigma(gammae,thetac):
    k=kappa(gammae,thetac)
    return k*(k**2-3./2.*k+3./2.)
#
def BW_cone(gammae,thetac):
    temp = gammae**2*thetac**2
    return temp/(1+temp)
def BW_div_FWHM(gammae,div_fwhm):
    return (gammae*div_fwhm/2)**2
def BW_dgamma(dgamma):
    #print(type(dgamma))
    return 2*dgamma
def BW_dlambdaL(dlambdaL):
    return dlambdaL



def BW_thetac(gammae,thetac):
    temp = gammae**2*thetac**2
    kappa = temp/(1+temp)
    return kappa
def BW_div_RMS(gammae,divergence_rms):
    Delta_theta = 2*np.sqrt(2*np.log(2))*divergence_rms
    return (gammae*Delta_theta)**2/4
def BW_a0(a0):
    return a0**2/2
def BW_electronEnergyspread(Delta_gamma):
    return 2*Delta_gamma
def BW_laserBandwidth(DeltaLambda):
    return DeltaLambda

#To do for later
#class ThomsonScattering(constants,laser,electrons)
