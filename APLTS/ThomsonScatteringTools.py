"""
This module contains Thomson Scattering Theory

Villa, F., Luccio, A., & Zolotorev, M. (1996). A source of kilovolt X-ray. Microsystem Technologies, 2(1), 79–82. https://doi.org/10.1007/BF02739535

Rykovanov, S. G., Geddes, C. G. R., Vay, J.-L., Schroeder, C. B., Esarey, E., & Leemans, W. P. (2014). Quasi-monoenergetic femtosecond photon sources from Thomson Scattering using laser plasma accelerators and plasma channels. Journal of Physics B: Atomic, Molecular and Optical Physics, 47(23), 234013. https://doi.org/10.1088/0953-4075/47/23/234013

"""
import sys
sys.path.append("/beegfs/desy/group/fla/ICS/tools/ModulesAndClasses/")
from APLTS import Laser
import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
import mpmath as mp
import math
import scipy
import physical_constants as constants
from joblib import Parallel, delayed
import multiprocessing
#
num_cores = multiprocessing.cpu_count()
#print("number of cores="+str(num_cores))
mp.dps = 50

"""
Photon Yield Formulas
"""    
def Photons_tot(laser_wavelength,laser_pulse_energy,laser_FWHM_duration,laser_waist,ebunch_gamma_e,ebunch_charge,ebunch_RMS_waist,ebunch_RMS_length,ebunch_norm_emittance):
    """
    returns total emitted photon number from Thomson interaction of Gaussian laser with electron bunch
    -----------------
    input parameters:
    laser_wavelength:      wavelength of the Thomson laser (m), float
    laser_pulse_energy:    energy of the laser pulse (J), float
    laser_FWHM_duration:   laser pulse duration (s, FWHM), float
    laser_waist:           laser focal waist (m), float
    ebunch_gamma_e:        electron bunch energy (dimensionless), float
    ebunch_charge:         electron bunch charge (C), float
    ebunch_RMS_waist:      electron bunch waist (m), float
    ebunch_RMS_length:     electron bunch length (m), float
    ebunch_norm_emittance: electron bunch normalised emittance
    """
    llambda=laser_wavelength
    Ep=laser_pulse_energy
    tau=laser_FWHM_duration
    w0=laser_waist
    gamma_e = ebunch_gamma_e
    Q_bunch=ebunch_charge
    if Q_bunch==None:
        Q_bunch=10e-12
        print("Q is none, set to " +str(Q_bunch)+" C")
    sig_et=ebunch_RMS_waist
    sig_el=ebunch_RMS_length
    eps_n=ebunch_norm_emittance
    #
    Nl=Ep/(constants.h*constants.c_light/llambda)
    Ne=Q_bunch/constants.e_charge
    sig_ll=(tau/(2.*sqrt(2.*log(2.))))*constants.c_light
    sig_lt=w0/2.
    sigma_l=np.sqrt(sig_el**2+sig_ll**2) # 535 um, mainly infl. by laser length
    beta_e=sig_et**2/(eps_n/gamma_e)
    beta_l=sig_lt**2/(llambda/(4.*np.pi))
    x=np.sqrt(2.)/sigma_l*np.sqrt((sig_et**2+sig_lt**2)/(sig_et**2/(beta_e**2)+sig_lt**2/(beta_l**2)))
    if type(x) is np.ndarray:
        F=np.array([np.float(mp.exp(xi**2)*(mp.erfc(xi))) for xi in x])   
    else:
        F=np.float(mp.exp(x**2)*(mp.erfc(x)))
    N_gamma=constants.sigma_T*Ne*Nl*F/(np.sqrt(2*np.pi)*sigma_l*np.sqrt(sig_et**2+sig_lt**2)*np.sqrt(sig_et**2/beta_e**2+sig_lt**2/beta_l**2))
    return N_gamma


def Photons_cone(laser_wavelength,laser_pulse_energy,laser_FWHM_duration,laser_waist,ebunch_gamma_e,ebunch_charge,ebunch_RMS_waist,ebunch_RMS_divergence,ebunch_RMS_length,ebunch_norm_emittance,collimation_angle,BWlim=0,angle_steps=50):
    """
    returns collimated emitted photon number from Thomson interaction of Gaussian laser with electron bunch into a confined observation angle
    --------------------
    input parameters:
    laser_wavelength:      wavelength of the Thomson laser (m), float
    laser_pulse_energy:    energy of the laser pulse (J), float
    laser_FWHM_duration:   laser pulse duration (s, FWHM), float
    laser_waist:           laser focal waist (m), float 
    ebunch_gamma_e:        electron bunch energy (dimensionless), float
    ebunch_charge:         electron bunch charge (C), float
    ebunch_RMS_waist:      electron bunch waist (m), float or array of floats
    ebunch_RMS_divergence: electron bunch divergence (rad), float or array of floats
    ebunch_RMS_length:     electron bunch length (m), float
    ebunch_norm_emittance: electron bunch normalised emittance
    collimation_angle:     half-opening angle of the collimation (rad), float
    BWlim:                 FWHM bandwidth limit for TS photon spec, float
    angle_steps:           sampling steps for te collimation angle array (dimensionless), int
    -------------------
    output parameters:
    N: number of photons emitted into collimation angle +-theta_max (dimensionless), float or array of floats
    spectrum: photon number energy spectrum (dN/dE) emitted into collimation angle +-theta  
    deltay: binning of the energy axis dE for spectrum
    """
    sigma_e=ebunch_RMS_waist
    sigma_div=ebunch_RMS_divergence
    llambda=laser_wavelength
    Ep=laser_pulse_energy
    tau=laser_FWHM_duration
    w0=laser_waist
    theta_max = collimation_angle
    gamma_e = ebunch_gamma_e
    Q_bunch=ebunch_charge
    sig_el=ebunch_RMS_length
    eps_n=ebunch_norm_emittance
    theta_steps=angle_steps
    #######################
    sig_lt=w0/2 # RMS laser waist
    tt=np.linspace(0,theta_max,theta_steps) #collimation angle theta array
    deltatheta=(tt[-1]-tt[0])/(len(tt)-1) # dtheta
    if BWlim==0:
        yy=np.linspace(0.1,1,1000) # normalised emitted photon energy yy=E_gamma/E_gamma_max
    else:
        yy=np.linspace(1.0-BWlim/2,1.0,1000)
    deltay=(yy[-1]-yy[0])/(len(yy)-1)
    if type(sigma_e) is np.ndarray:
        spectrum=np.zeros((len(sigma_e),len(yy)))
        N=np.zeros(len(sigma_e))
        for ss,s in enumerate(sigma_div):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3./(2.*s**2)*(1.-2.*y*(1.-y))*mp.exp(-(theta**2+((1.-y)/(gamma_e**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gamma_e**2*y))/s**2)*theta*deltatheta
                N[ss]+=Photons_tot(llambda, Ep, tau, w0, gamma_e,Q_bunch,sigma_e[ss],sig_el,eps_n)*spectrum[ss][i]*deltay
    elif type(sigma_div) is np.ndarray:
        spectrum=np.zeros((len(sigma_div),len(yy)))
        N=np.zeros(len(sigma_div))
        for ss,s in enumerate(sigma_div):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gamma_e**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gamma_e**2*y))/s**2)*theta*deltatheta
                N[ss]+=Photons_tot(llambda, Ep, tau, w0, gamma_e,Q_bunch,sigma_e,sig_el,eps_n)*spectrum[ss][i]*deltay
    else:
        spectrum=np.zeros(len(yy))
        N=0
        s=sigma_div
        for i,y in enumerate(yy):
            for theta in tt:
                term = (1-y)/(gamma_e**2*y)
                if term<0:
                    spectrum[i]+=0
                else:
                    spectrum[i]+=np.float(3/(2*sigma_div**2)*(1-2*y*(1-y))*mp.exp(-(theta**2+((1-y)/(gamma_e**2*y)))/(2*s**2))*mp.besseli(0, theta*np.sqrt((1-y)/(gamma_e**2*y))/sigma_div**2)*theta*deltatheta)
            N+=Photons_tot(llambda, Ep, tau, w0, gamma_e,Q_bunch,sigma_e,sig_el,eps_n)*spectrum[i]*deltay
        #print(type(N))
    if np.sum(N)==0:
        if np.sum(spectrum==0):
            print("spectrum is all zeros")
        print(deltay)
    #print(type(N))
    return N,spectrum,deltay


"""
Bandwidth Formulas
"""
def BW_collimation(ebunch_energy,collimation_angle):
    gamma_e=ebunch_energy
    theta=collimation_angle
    temp = gamma_e**2*theta**2
    kappa = temp/(1+temp)
    return kappa
def BW_divergence(ebunch_energy,ebunch_RMS_divergence):
    ebunch_energy=gamma_e
    ebunch_RMS_divergence=div
    Delta_theta = 2*np.sqrt(2*np.log(2))*div
    return (gamma_e*Delta_theta)**2/4
def BW_a0(laser_a0):
    return laser_a0**2/2
def BW_ebunchEnergySpread(Delta_gamma):
    return 2.0*Delta_gamma
def BW_laserBandwidth(DeltaLambda):
    return DeltaLambda





def Optimise(sigmamin,sigmamax,taumin,taumax,llambda,Ep,a0,gamma_e,Q,eps_n,sigma_el,theta_c,printall,taumin_fix=0,BWlim=0,N_run=3):
    #First part: Thomson Optimisation. Returns laser and bunch params at target gamma which give highest photon number in cone
    #We want to do three runs to find the optimum
    #Each time we close in more on the opt values
    for opt_run in range(0,N_run):
        print("opt run "+str(opt_run))
        sigma_arr_theo=np.linspace(sigmamin,sigmamax,10)
        delta_sigma=sigma_arr_theo[1]-sigma_arr_theo[0]
        if taumin==taumax:
            w0 = Laser.w0_calc(llambda,Ep,a0,taumin)
            result = Parallel(n_jobs = num_cores)(delayed(Photons_cone)(llambda,Ep,taumin,w0,gamma_e,Q,sigma_arr_theo[i],eps_n/gamma_e/sigma_arr_theo[i],sigma_el,eps_n,theta_c,BWlim) for i,s in enumerate(sigma_arr_theo))
            #print(type(result))
            #print(type(result[0]))#
            #print(result[0])
            #N_theo = np.array(result)[:,0]
            N_theo = np.array([first for (first,middle,last) in result])
            for j in range(0,len(N_theo)):
                N_theo[j]=np.float(N_theo[j])
                if np.isnan(N_theo[j])==True:
                    N_theo[j] = 0
                elif np.isfinite(N_theo[j])==False:
                    N_theo[i][j]=0
            bestNgamma = np.nanmax(N_theo)
            bestsigmar = sigma_arr_theo[np.nanargmax(N_theo)]
            besttau = taumin
            bestw0 = w0
            Nmax = bestNgamma
            sigma_max = bestsigmar
        else:
            tau_arr = np.linspace(taumin,taumax,10)
            delta_tau = tau_arr[1]-tau_arr[0]
            w0_arr = Laser.w0_calc(llambda,Ep,a0,tau_arr)
            print("tau array:"+str(taumin)+" to "+str(taumax))
            result = Parallel(n_jobs = num_cores)(delayed(Photons_cone)(llambda,Ep,tau_arr[i],w0_arr[i],gamma_e,Q,sigma_arr_theo,eps_n/gamma_e/sigma_arr_theo,sigma_el,eps_n,theta_c,BWlim) for i,t in enumerate(tau_arr))
            #print(type(result))
            #print(type(result[0]))
            #print(result[0])
            #print(np.array(result))
            N_theo = np.array(result)[:,0]
            for i,t in enumerate(tau_arr):
                for j in range(0,len(N_theo[i])):
                    if np.isnan(N_theo[i][j])==True:
                        N_theo[i][j]=0
                    elif np.isfinite(N_theo[i][j])==False:
                        N_theo[i][j]=0
            N_max = np.zeros(len(tau_arr))
            sigma_max = np.zeros(len(tau_arr))
            for j in range(0,len(tau_arr)):
                N_max[j]=np.nanmax(N_theo[j])
                sigma_max[j]=sigma_arr_theo[np.nanargmax(N_theo[j])]
            besttau=tau_arr[np.nanargmax(N_max)]
            bestw0=w0_arr[np.nanargmax(N_max)]
            bestsigmar=sigma_max[np.nanargmax(N_max)]
            bestNgamma=np.nanmax(N_max)
        if printall=="True":
            print("Maximum yield at")
            print("tau="+str(besttau))
            print("w0="+str(bestw0))
            print("sigmar="+str(bestsigmar))
            print("Ngamma="+str(bestNgamma))
            print("")
        if taumin==taumax:
            temp=sigma_max-delta_sigma
            if temp<0:
                sigmamin=0
            else:
                sigmamin=temp
            sigmamax=sigma_max+delta_sigma
        else:
            temp=sigma_max[np.nanargmax(N_max)]-delta_sigma
            if temp<0:
                sigmamin=0
            else:
                sigmamin=sigma_max[np.nanargmax(N_max)]-delta_sigma
            sigmamax=sigma_max[np.nanargmax(N_max)]+delta_sigma
            temp=tau_arr[np.nanargmax(N_max)]-delta_tau
            if temp<taumin_fix:
                taumin=taumin_fix
            else:
                taumin=tau_arr[np.nanargmax(N_max)]-delta_tau
            taumax=tau_arr[np.nanargmax(N_max)]+delta_tau
    return besttau,bestw0,bestsigmar,bestNgamma
