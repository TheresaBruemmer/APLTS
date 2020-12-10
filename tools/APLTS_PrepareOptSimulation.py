import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.optimize import curve_fit
from numpy import sqrt,cos,sin,tan,exp, pi, log
import scipy
from scipy import special
import mpmath as mp
import math
from joblib import Parallel, delayed
import multiprocessing
import sys
sys.path.append("/beegfs/desy/group/fla/ICS/tools/ModulesAndClasses/")
import APLTS.ActivePlasmaLens as APL
from APLTS import APLTS,Laser
printall="False"
num_cores = multiprocessing.cpu_count()
print("number of cores="+str(num_cores))

"""
Laser and Electron Bunch Parameters
"""
a0=0.9
Ep=0.2
llambda=1030.e-9
taumin_fix=1e-12
taumin=1.0e-12
taumax=15.e-12
#
eps_n=1.0e-6
sigmar_i=1.0e-6
sig_el=1.5e-6
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

'''
Constants and calculations
'''
alpha_f=1./137.
e_charge=1.6022e-19
c_light=299792458
h=6.62607e-34
eps_0=8.854e-12
m_e=9.109e-31
r_e=e_charge**2/(m_e*c_light**2)/(4.*np.pi*eps_0)
sigma_T=8.*np.pi/3*r_e**2 # Thomson cross section 6.65e-29


EL=h*c_light/llambda/e_charge
Nl=Ep/(h*c_light/llambda)
gamma=sqrt(Etarget/4/EL*(1+a0**2/2))



"""
Functions
"""
#These could be imported from ICS/tools
def a0_calc(w0, tau_fwhm):
    tau=tau_fwhm/(sqrt(2.*np.log(2)))
    return llambda*e_charge/(2*m_e*c_light**2*pi)*sqrt(Ep/(sqrt(pi/2)*pi/4*eps_0*c_light*w0**2*tau))
def tau_calc(a0,w0):
    return (llambda*e_charge/(2*m_e*c_light**2*pi*a0))**2*(Ep/(sqrt(pi/2)*pi/4*eps_0*c_light*w0**2))*np.sqrt(2*np.log(2))
def w0_calc(a0,tau_fwhm):
    return e_charge/a0*llambda/(2.*pi*m_e*c_light**2)*sqrt(Ep/(sqrt(pi/2.)*pi/4.*eps_0*c_light*(tau_fwhm/(sqrt(2.*log(2.))))))





def yield_(tau, w0,eps_n,sig_et,Ep):
    Nl=Ep/(h*c_light/llambda)
    Ne=Q/e_charge
    sig_ll=(tau/(2.*sqrt(2.*log(2.))))*c_light
    sig_lt=w0/2.
    sigma_l=np.sqrt(sig_el**2+sig_ll**2) # 535 um, mainly infl. by lxaser length
    beta_e=sig_et**2/(eps_n/gamma)
    beta_l=sig_lt**2/(llambda/(4.*np.pi))
    x=np.sqrt(2.)/sigma_l*np.sqrt((sig_et**2+sig_lt**2)/(sig_et**2/(beta_e**2)+sig_lt**2/(beta_l**2)))
    F=mp.exp(x**2)*(mp.erfc(x))
    N_gamma=sigma_T*Ne*Nl*F/(np.sqrt(2*np.pi)*sigma_l*np.sqrt(sig_et**2+sig_lt**2)*np.sqrt(sig_et**2/beta_e**2+sig_lt**2/beta_l**2))
    return float(N_gamma)


def yield_cone_div_theta(sigma_e,eps_n,w0,theta_max, gamma,tau,Ep,theta_steps):
    sig_lt=w0/2
    tt=np.linspace(0,theta_max,theta_steps)
    deltatheta=(tt[-1]-tt[0])/(len(tt)-1)
    yy=np.linspace(1.0-BWlim/2,1.0,1000)
    deltay=(yy[-1]-yy[0])/(len(yy)-1)
    div_e=eps_n/(gamma*sigma_e)
    sigma0=div_e#*FWHMtoRMS
    if type(sigma_e) is np.ndarray:
        spectrum=np.zeros((len(sigma_e),len(yy)))
        N=np.zeros(len(sigma_e))
        for ss,s in enumerate(div_e):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*float(mp.exp(-(theta**2+((1-y)/(gamma**2*y)))/(2*s**2)))*float(mp.besseli(0, theta*np.sqrt((1-y)/(gamma**2*y))/s**2))*theta*deltatheta
                N[ss]+=yield_(tau, w0,eps_n,sigma_e[ss],Ep)*spectrum[ss][i]*deltay
    elif type(eps_n) is np.ndarray:
        spectrum=np.zeros((len(eps_n),len(yy)))
        N=np.zeros(len(eps_n))
        for ss,s in enumerate(div_e):
            for i,y in enumerate(yy):
                for jj,theta in enumerate(tt):
                    spectrum[ss][i]+=3/(2*s**2)*(1-2*y*(1-y))*float(mp.exp(-(theta**2+((1-y)/(gamma**2*y)))/(2*s**2)))*float(mp.besseli(0, theta*np.sqrt((1-y)/(gamma**2*y))/s**2))*theta*deltatheta
                N[ss]+=yield_(tau, w0,eps_n[ss],sigma_e,Ep)*spectrum[ss][i]*deltay
    else:
        spectrum=np.zeros(len(yy))
        N=0
        s=div_e
        for i,y in enumerate(yy):
            for theta in tt:
                spectrum[i]+=3/(2*sigma0**2)*(1-2*y*(1-y))*float(mp.exp(-(theta**2+((1-y)/(gamma**2*y)))/(2*s**2)))*float(mp.besseli(0, theta*np.sqrt((1-y)/(gamma**2*y))/sigma0**2))*theta*deltatheta
            N+=yield_(tau, w0,eps_n,sigma_e,Ep)*spectrum[i]*deltay
    return N

#We want to do three runs to find the optimum
#Each time we close in more on the opt values
for opt_run in range(0,4):
    sigma_arr_theo=np.linspace(sigmamin,sigmamax,10)
    delta_sigma=sigma_arr_theo[1]-sigma_arr_theo[0]
    tau_arr=np.linspace(taumin,taumax,10)
    delta_tau=tau_arr[1]-tau_arr[0]
    w0_arr=w0_calc(a0,tau_arr)
    print("opt run"+str(opt_run))
    print("tau array:"+str(taumin)+" to "+str(taumax))
    #N_theo=np.zeros((len(tau_arr),len(sigma_arr_theo)))
    #print(type(yield_cone_div_theta(sigma_arr_theo,eps_n,w0_arr[0],theta_c,gamma,tau_arr[0],Ep,50)))
    #yield_cone_div_theta(sigma_e,eps_n,w0,theta_max, gamma,tau,Ep,theta_steps)
    #result=Parallel(n_jobs = num_cores)(delayed(yield_cone_div_theta)(1,1,1,i,1,1,1,10) for i in range(10))
    result=Parallel(n_jobs = num_cores)(delayed(yield_cone_div_theta)(sigma_arr_theo,eps_n,w0_arr[i],theta_c,gamma,tau_arr[i],Ep,50) for i,t in enumerate(tau_arr))
    N_theo=np.array(result)
    for i,t in enumerate(tau_arr):
        #N_theo[i]=yield_cone_div_theta(sigma_arr_theo,eps_n,w0_arr[i],theta_c,gamma,tau_arr[i],Ep,50)[1]
        for j in range(0,len(N_theo[i])):
            if np.isnan(N_theo[i][j])==True:
                N_theo[i][j]=0
            elif np.isfinite(N_theo[i][j])==False:
                N_theo[i][j]=0
        N_max=np.zeros(len(tau_arr))
        sigma_max=np.zeros(len(tau_arr))
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


def FindAPLSimData(gammae,sigmar_f,Nrun,I0_min,I0_max):
    '''
    We want to do Nrun runs to find the optimum.
    Each time we close in more on the opt values
    Output: I_0
    z_F
    '''
    for opt_run in range(0,Nrun+1):
        I0_arr = np.linspace(I0_min,I0_max,10)
        delta_I0=I0_arr[1]-I0_arr[0]
        instances=APL.APL_setup(L=L_APL,I_0=I0_arr,z_0=z_0,r_0=r_APL,gammae=gammae,eps_n=eps_n,sigmar_i=sigmar_i)
        match_arg = np.nanargmin(abs(instances.focalWaist()-sigmar_f))
        I0=I0_arr[match_arg]
        zF=instances.focalPlane()[match_arg]
        check_Waist=instances.focalWaist()[match_arg]
        I0_min=I0-2*delta_I0
        I0_max=I0+2*delta_I0
    return I0,zF,check_Waist



def L_H(llambda, alpha, gamma, num):
    L=llambda/(num*(1-(1-1/gamma**2)**(0.5)*math.cos(alpha))*((1-1/gamma**2))**(0.5)*3e8)*1e9
    return L



def AstraInputCalc(GAMMA,zF,llambda):
    beta=np.sqrt(1.0-1./GAMMA**2)
    zstart=zF*(1./beta-1.)
    return zstart
def Ekin(gamma):
    return (gamma-1)*0.511




def generatorEkin(gammae,enspread):
    E_kin=Ekin(gammae)
    gammamin = gammae*(1-2*enspread)
    gammamax = gammae*(1+2*enspread)
    deltagamma=gammamax-gammamin
    #sig_Ekin=(E_kin-Ekin(gammamin))/np.sqrt(3)
    sig_Ekin=(Ekin(gammamax)-E_kin)/np.sqrt(3)
    return E_kin,sig_Ekin, deltagamma

I0,zF,check_Waist = FindAPLSimData(gamma,bestsigmar,Nrun,I0_min,I0_max)
zstart=AstraInputCalc(gamma,zF,llambda)
APL_instance=APL.APL_setup(L=L_APL,I_0=I0,z_0=z_0,r_0=r_APL,gammae=gamma,eps_n=eps_n,sigmar_i=sigmar_i)
Laser_instance=LaserClass.Laser(_wavelength=llambda,_Ep=Ep,_a0=a0,_tau_FWHM=besttau)
eff_enspread=APLTS.effective_energyspread_fixedsetup(Laser_instance,APL_instance)[0]
Ekin,sig_Ekin, delta_gammae= generatorEkin(gamma,eff_enspread)


print("---------------------------")
print("Simulation Parameters")
print("a0 = "+str(a0))
print("gamma = "+str(gamma))
print("Ekin = "+str(Ekin) + " eV")
if taumin_fix>0:
    print("minimum tau was fixed to "+str(taumin_fix)+" s")
print("tau="+str("%.4e" %(besttau))+" s")
print("w0="+str("%.3e" %(bestw0))+ " m")
print("sigmar="+str("%.3e" %(bestsigmar))+" m")
print("I0 = " + str("%.2f" %I0)+" A")
print("zF = " + str("%.3f" %zF)+" m")
print("check sigmar_f = "+str("%.3f" %(check_Waist*1e6))+" um")
print("Zstart="+ str(zstart))
print("L_H = " + str(L_H(llambda,alpha_coll,gamma,NUM)))
print("eff enspread = " + str("%.3f" %(100*eff_enspread)) + " %")
print("simulate sig_Ekin = " + str(sig_Ekin) + " eV")
print("---------------------------")


