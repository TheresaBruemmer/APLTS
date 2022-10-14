"""
------------------------------------------------------------------------------------------------------------
--- Photon number (dN) emission for different electron energies (dgamma) (focused by APL) ---
Rykovanov, S. G., Geddes, C. G. R., Vay, J.-L., Schroeder, C. B., Esarey, E., & Leemans, W. P. (2014). Quasi-monoenergetic femtosecond photon sources from Thomson Scattering using laser plasma accelerators and plasma channels. Journal of Physics B: Atomic, Molecular and Optical Physics, 47(23), 234013. https://doi.org/10.1088/0953-4075/47/23/234013
------------------------------------------------------------------------------------------------------------
This script calculates (and plots) the number of emitted photons from the Thomson interaction of an APL-focused bunch with a head-on propagating Gaussian laser.
Since the APLTS package does not include energy spread bunch propagation/interaction, the influece of this is approached by determining the propagation and TS photon emission of electrons of different energies gamma_e in the APLTS setup:
The APL focuses the target gamma_e (required for a specified Thomson photon energy), the interaction with the TS laser is matched to this focus position. Then, gamma_e is varied and the TS photons calculated via Rykovanov et al. JPB 2014. 
------------------------------------------------------------------------------------------------------------
Requires APLTS package and associated python packages

This script's result can be compared to the results dN/dgamma_e from the single-trace clara2 evaluation. 


To do: Make this script an operation in APLTS.py e.g. Calc_dNdgammae or similar
"""
#My modules
import sys
sys.path.append("/home/bruemt/code/APLTS/")
from APLTS import Laser,ActivePlasmaLens,ThomsonScattering,APLTS
import APLTS.utilities.physical_constants as constants
from APLTS.utilities.GeneralDataAnalysis import _nanargmin #in GDA, mpl and plt are already loaded
#
import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
import h5py
#Plotting packages
import matplotlib as mpl
#if mpl.get_backend()=='Qt5Agg':
#    mpl.use("pdf")
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#Parallel processing
from joblib import Parallel, delayed
import multiprocessing as mpro
num_cores = mpro.cpu_count()


"""
Calculation Parameters
"""
plot_data=True
length = 100 # length of the gamma arrays around target gamma, integer
f0=1.0 # focal plane (m), float
gamma_arr = np.array([100,300,500,700,900]) # array of different target gamma_e
eps_n = 1.e-6
#initiate laser instance
llambda=800.e-9
w0 = 16.7e-6
tau=1.0e-12
Ep=0.1
Laser_instance=Laser.Laser(_wavelength=llambda,_Ep=Ep,_tau_FWHM=tau,_w0=w0)
theta_max = 0.5e-3
#
I0_arr = np.linspace(100,1100,5000) #range of APL currents
I0_1m_arr = np.zeros(len(gamma_arr)) #initiate current array which focuses target gamma_e at focal plane 1m
gamma_foc = np.zeros(len(gamma_arr))
APL_1m = dict()
for ig, gg in enumerate(gamma_arr):
    APL_instance=ActivePlasmaLens.APL_setup(gammae=gg,I_0=I0_arr)
    #array of current that focuses gamma_e=gg at a 1m focal plane (later: rename for different possible f0≠1.0)
    I0_1m_arr[ig]=I0_arr[_nanargmin(abs(APL_instance.focalLength()-(f0-APL_instance.L-APL_instance.z_0)))]
    #APL instance for each gamma_e with current set to focus at 1m focal plane
    APL_1m[gg]=ActivePlasmaLens.APL_setup(gammae=gg,I_0=I0_1m_arr[ig])
#
N_cone_D=dict()
waist_D=dict()
div_D=dict()
gamma_instance_D=dict()
#
"""
Calculate photon number arrays for each setup with varying gamma_e
"""
for index,gg in enumerate(gamma_arr):
    I0=I0_1m_arr[index]
    APL_instance=APL_1m[gg]
    #create Thomson Scattering instance
    TS=ThomsonScattering.ThomsonScattering(Laser_instance,APL_instance.return_focusedBunch())
    gamma_instance_low_arr=np.linspace(gg*0.95,gg,length)
    gamma_instance_high_arr=np.linspace(gg,gg*1.05,length)
    #calculate the number of photons emitted for all gammas<gg
    result_low=Parallel(n_jobs = num_cores)(delayed(APLTS.N_collimated_gammae_var)(APL_instance,TS,theta_max,gam) for jg,gam in enumerate(gamma_instance_low_arr))
    waist_low = np.array(result_low)[:,0]
    div_low = np.array(result_low)[:,1]
    N_cone_low = np.array(result_low)[:,2]
    #calculate the number of photons emitted for all gammas>gg
    result_high=Parallel(n_jobs = num_cores)(delayed(APLTS.N_collimated_gammae_var)(APL_instance,TS,theta_max,gam) for jg,gam in enumerate(gamma_instance_high_arr))
    waist_high=np.array(result_high)[:,0]
    div_high=np.array(result_high)[:,1]
    N_cone_high=np.array(result_high)[:,2]
    #Concatenate arrays
    waist_D[index]=np.concatenate((waist_low,waist_high))
    div_D[index]=np.concatenate((div_low,div_high))
    N_cone_D[index]=np.concatenate((N_cone_low,N_cone_high))
    gamma_instance_D[index]=np.concatenate((gamma_instance_low_arr,gamma_instance_high_arr))
    if plot_data==True:
        print("Plotting Data "+str(index) +" ...")
        plt.plot(gamma_instance_D[index],waist_D[index])
        plt.xlabel("gamma")
        plt.ylabel("waist")
        plt.savefig("TheoreticalEffEnSpread_eWaist_"+str(gg)+".pdf",bbox_inches="tight")
        plt.clf()
        plt.plot(gamma_instance_D[index],N_cone_D[index])
        plt.xlabel("gamma")
        plt.ylabel("Ngamma")
        plt.savefig("TheoreticalEffEnSpread_"+str(gg)+".pdf",bb0x_inches="tight")
        plt.clf()


print("Save data to hdf5 file")    
with h5py.File("TheoreticalEffEnSpread.h5","w") as hf:
    hf.create_dataset("gammae_target",data=gamma_arr)   
    hf.create_dataset("I0_1m_target",data=I0_1m_arr)    
    waist_G = hf.create_group("Waist")
    div_G = hf.create_group("Divergence")
    N_cone_G = hf.create_group("N_cone")
    gamma_instance_G = hf.create_group("gamma_e")
    for index in waist_D:
        #print(index)
        waist_G.create_dataset(str(index),data=waist_D[index])
        div_G.create_dataset(str(index),data=div_D[index])
        N_cone_G.create_dataset(str(index),data=N_cone_D[index])
        gamma_instance_G.create_dataset(str(index),data=gamma_instance_D[index])
