"""
---------------------------------------
Active-Plasma Lens + Thomson Scattering (APLTS)
----------------------------------------
This module contains the functions for Thomson Scattering of a Gaussian laser with an APL-focused, thus chromatically focused, electron bunch 
Reference: Brümmer et al. "Compact all-optical tunable narrowband Compton hard X-ray source", to be published
"""
#import sys
#sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")


import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#import scipy
#from scipy import special
from scipy.optimize import curve_fit
import math
#
from APLTS import Laser,ThomsonScattering
import APLTS.utilities.physical_constants as constants
import APLTS.ActivePlasmaLens as APL
import APLTS.utilities.GeneralDataAnalysis as GDA
from APLTS.utilities.FittingFunctions import Lorentzian_amp as Lorentzian


def N_collimated(APL_instance,TS_instance,collimation_angle):
    """
    Calculates the number of emitted photons for given APL, Laser and TS instances into a collimation angle
    This function gives the photon number for the interaction in the focus of electrons with energy focused by an APL with I0 both defined in the APL_instance. It does not calculate the number of photons emited by electrons of different gammae at the given current and target-gammae focus. To calculate theor contribution, use function N_collimated_gammae_var
    --------------------------
    input:
    APL_instance
    Laser_instance
    TS_instance
    collimation_angle: half-opening angle for detection cone
    """
    waist = APL_instance.focalWaist()
    div = APL_instance.focalDivergence()
    w0=TS_instance.laser._w0
    tau=TS_instance.laser._tau_FWHM
    Ep=TS_instance.laser._Ep
    N_cone_low = float(TS_instance.Photons_cone(collimation_angle))
    return N_cone

def N_collimated_gammae_var(APL_instance,TS_instance,collimation_angle,gammae):
    """
    This function calculates the number of emitted photons for given APL, Laser and TS instances into a collimation angle by electrons of arbitrary energy gammae
    Here, an APL_instance is given based on the target gammae parameters. The gammae under investigation is defined in the fct arguments (see below).
    --------------------
    input: 
    APL_instance,Laser_instance,TS_instance,collimation_angle,gammae
    output: 
    waist,div,N_cone of the given gammae at the focus of the APL_instance
    """
    f0=APL_instance.focalLength() # focal length of target gammae
    #create a new instance with different gammae
    APL_instance2 = APL_instance
    APL_instance2.gammae=gammae
    APL_=APL_instance2.propagate()[3]
    FD_=APL.propagateTwiss(APL_,APL.M_freedrift(f0-APL_instance.L-APL_instance.z_0)) # propagate to the target gamma focus
    TS_instance.bunch._sigma_div = np.sqrt(FD_[2]*APL_instance2.eps_n/APL_instance2.gammae) # find the divergence of electrons with gammae=gam at target gammae focus f0
    TS_instance.bunch._sigma_r = np.sqrt(FD_[0]*APL_instance2.eps_n/APL_instance2.gammae) # find the waist of electrons wth gammae=gam at target gammae focus f0
    TS_instance.gamma=gammae # overwrite gamma in the TS_instance
    w0=TS_instance.laser._w0
    tau=TS_instance.laser._tau_FWHM
    Ep=TS_instance.laser._Ep
    N_cone = float(TS_instance.Photons_cone(collimation_angle))
    return TS_instance.bunch._sigma_r,TS_instance.bunch._sigma_div,N_cone

def effective_energyspread_fromdata(gammas,Ngammas,save_plot=None):
    """
    returns effective energy spread from single-trace data
    fits Lorentzian function to data
    -------------
    input parameters:
    gammas: array of energy gamma of simulated macroparticles
    Ngammas: array of radiated number of photons by simulated macroparticles (calculated by clara2)
    -------------
    output: 
    effective energy spread: FWHM width of Lorentzian fit to data (Ngamma vs gamma)
    gammae: electron energy with most contribution to the spectrum, i.e. peak position of fitted Lorentzian
    popt: fit parameters of the Lorentzian fit to the data
    """
    target_gammae = GDA.weighted_avg_and_std(gammas,Ngammas,True)
    gamma_arr = np.linspace(np.nanmin(gammas),np.nanmax(gammas),1000)
    Ngamma_arr = np.zeros_like(gamma_arr)
    try:
        popt,pcov=curve_fit(Lorentzian,gammas,Ngammas,p0=[target_gammae,1,1])
        Ngamma_arr = Lorentzian(gamma_arr,*popt)
        effective_energy_spread_FWHM = 2*abs(popt[1])/popt[0]
    except:
        print("Lorentzian fit not possible, try find_FWHM")
        effective_energy_spread_FWHM = np.NAN
	popt=np.nan
    if save_plot is not None:
        plt.plot(gammas,Ngammas,".")
        plt.plot(gamma_arr,Ngamma_arr)
        plt.savefig(save_plot)
    try:
        peak,realwidth,width=GDA.find_FWHM(gamma_arr,Ngamma_arr)
        gammae = peak
        if effective_energy_spread_FWHM==np.NAN:
            effective_energy_spread_FWHM=width
        if abs(gammae-target_gammae)/target_gammae>0.05:
            print("fit peak ("+str("%.2f" %gammae)+") differs from weighted mean ("+str("%.2f" %target_gammae)+")")
    except:
        print("could not determine FWHM")
    return effective_energy_spread_FWHM,gamma_arr,popt
def N_eff_simple(sigma_e,w0):
    """
    returns photon production factor given solely by the waist relation in the Thomson interaction of a bunch with a Gaussian laser
    for this scenario, N_eff_simple follows a Lorentzian distribution
    ------------
    input parameters:
    sigma_e: RMS bunch waist (m), float
    w0: Gaussian laser waist (m), float
    output: 
    Neff_simple: overlap-dependent photon production factor
    """
    return 1/(4*sigma_e**2/w0**2+1)
def N_eff(sigma_e,w0,a0,amp):
    """
    returns effective photon yield N_eff for an electron bunch with defined waist interacting with a Gaussian laser 
    for this scenario, N_eff has a Lorentzian distribution
    ------------
    input parameters:
    sigma_e: RMS bunch waist (m), float
    w0: Gaussian laser waist (m), float
    a0: Gaussian laser norm. amplitude, i.e. laser-strength parameter (dimensionless)
    amp: scaling amplitude (dimensionless), required for fit
    output:
    N_eff: effective photon yield, float
    """
    return N_eff_simple(sigma_e,w0)*a0**2*amp
def effective_energyspread(zF,Laser_instance,APL_instance):
    """
    returns effective energy spread for a fixed electron energy gammae and focal plane zF
    requires instance of Laser and Active Plasma Lens classes
    performs Lorentzian fit, since Neff is assumed to follow a Lorentzian distribution
    ------------------
    input parameters:
    zF: focal plane (m), i.e. distance from acceleration stage to focus z0+L+f
    APL_instance: plasma lens parameters and gammae are fixed, current is array
    Laser_instance: instance of Gaussian Laser class
    output: 
    effective_energy_spread_FWHM: effective  energy spread (FWHM)
    I0_zF: plasma lens current to focus target gamma at target focal plane zF
    popt: Lorentzian fit parameters
    """
    gammae = APL_instance.gammae
    #Find lens current which focuses the given gammae at the given zF 
    I0_arr = APL_instance.I_0
    f=zF-APL_instance.L-APL_instance.z_0
    I0_zF=I0_arr[GDA._nanargmin(abs(APL_instance.focalLength()-(f)))]
    gammae_instance_arr=gammae*np.linspace(0.8,1.2,1000)
    APL_instance_zF = APL_instance
    APL_instance_zF.gammae=gammae_instance_arr
    APL_instance_zF.I_0=I0_zF
    Twiss_APL_zF=APL_instance_zF.propagate()[:][3]
    beta_at_target_foc_new_instance=APL.propagateTwiss(Twiss_APL_zF,APL.M_freedrift(f))[0]
    sigma_at_target_foc_new_instance=sqrt(beta_at_target_foc_new_instance*APL_instance_zF.eps_n/gammae)
    Neff= N_eff(sigma_at_target_foc_new_instance,Laser_instance._w0,Laser_instance._a0,1)
    #guess
    peak,realwidth,width=GDA.find_FWHM(gammae_instance_arr,Neff,None,False)
    #fit
    popt,pcov = curve_fit(Lorentzian,gammae_instance_arr,Neff,p0=[peak,width,np.nanmax(Neff)])
    effective_energy_spread_FWHM = abs(2*popt[1]/popt[0]) # attention: not %
    return effective_energy_spread_FWHM,I0_zF,popt
def effective_energyspread_fixedsetup(Laser_instance,APL_instance):
    """
    Analogous to function effective_energyspread, but for APL_instance with fixe plasma current:
    Effective energy spread for a fixed electron energy gammae and focal plane zF
    requires instance of Laser and Active Plasma Lens classes
    performs Lorentzian fit, since Neff is assumed to follow a Lorentzian distribution
    -------------------------
    input:
    Laser_instance: Instance of Gaussian laser class
    APL_instance: plasma lens parameters including current and gammae are fixed
    output:
    effective_energy_spread_FWHM: effective  energy spread (FWHM)
    popt: Lorentzian fit parameters
    """
    gammae = APL_instance.gammae    
    f=APL_instance.focalLength()
    I0_zF=APL_instance.I_0
    gammae_instance_arr=gammae*np.linspace(0.8,1.2,1000)
    APL_instance_zF = APL_instance
    APL_instance_zF.gammae=gammae_instance_arr
    Twiss_APL_zF=APL_instance_zF.propagate()[:][3]
    beta_at_target_foc_new_instance=APL.propagateTwiss(Twiss_APL_zF,APL.M_freedrift(f))[0]
    sigma_at_target_foc_new_instance=sqrt(beta_at_target_foc_new_instance*APL_instance_zF.eps_n/gammae)
    Neff= N_eff(sigma_at_target_foc_new_instance,Laser_instance._w0,Laser_instance._a0,1)
    #guess
    peak,realwidth,width=GDA.find_FWHM(gammae_instance_arr,Neff,None,False)
    #fit
    popt,pcov = curve_fit(Lorentzian,gammae_instance_arr,Neff,p0=[peak,width,np.nanmax(Neff)])
    effective_energy_spread_FWHM = abs(2*popt[1]/popt[0])
    return effective_energy_spread_FWHM,popt
def effective_divergence_fromdata(thetas,Ngammas):
    """
    returns the weighted propagation angles of the simulated macroparticles from single-trace data
    Weight is emitted photon number
    No fit because fit funtion yet unknown, data is binned and weighted
    ----------------------------------------
    input:
    thetas: propagation angle of macroparticle with respect to mean propagation axis (rad), array of floats
    Ngammas: array of radiated number of photons by simulated macroparticles (calculated by clara2)
    output:
    mean: angle with most contribution to spectrum (rad)
    FWHM: width of distribution (rad, FWHM)
    """
    x_arr,y_arr=GDA.binData(thetas,Ngammas)
    mean,FWHM=GDA.find_FWHM_savgol_0center(x_arr,y_arr)
    return mean,FWHM
def Kraemer_eff_div_BW(gamma_e,sigma_theta_eff):
    """
    Krämer, J. M., Jochmann, A., Budde, M., Bussmann, M., Couperus, J. P., Cowan, T. E., … Irman, A. (2018). Making spectral shape measurements in inverse Compton scattering a tool for advanced diagnostic applications. Scientific Reports, 8(1), 1398. https://doi.org/10.1038/s41598-018-19546-0
    #
    X-ray BW as function of eff. RMS bunch divergence, according to Krämer et al. 
    """
    num=1.05*(gamma_e*sigma_theta_eff)**2
    denom=1+(gamma_e*sigma_theta_eff)**2
    return num/denom


