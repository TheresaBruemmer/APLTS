"""
This module contains python modules for clara2 postprocessinf
"""
#import sys
#sys.path.append("/home/bruemt/ICS/tools/ModulesAndClasses/")
import APLTS.utilities.GeneralDataAnalysis as GDA
from APLTS.utilities.physical_constants import *
import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def sumspectra(radData,thetamin,thetamax,Energymin,Energymax, theta_extent_arr,pixel_dOmega,Energy_arr):
    id_y_min=np.nanargmin(np.abs(theta_extent_arr - thetamin*1e3))
    id_y_max0=np.nanargmin(np.abs(theta_extent_arr - thetamax*1e3))
    id_y_max=id_y_max0+1
    id_x_min=np.nanargmin(np.abs(Energy_arr - Energymin))
    id_x_max0=np.nanargmin(np.abs(Energy_arr - Energymax))
    id_x_max=id_x_max0+1
    spec=(abs(np.pi*np.sin(theta_extent_arr*1.0e-3))[id_y_min:id_y_max,None]*(radData[0][id_y_min:id_y_max, id_x_min:id_x_max]+radData[1][id_y_min:id_y_max, id_x_min:id_x_max])/2/np.sqrt(pixel_dOmega))
    dNdE = np.sum(spec,axis=0)
    E_arr = Energy_arr[id_x_min:id_x_max]
    if len(dNdE)!=len(E_arr):
        print("error")
    return spec, np.sum(spec), dNdE, E_arr

def PlotSpectrum(dNdE_data,Energy_data):
    delta_E=Energy_data[1]-Energy_data[0]
    plt.plot(Energy_data,dNdE_data)
    plt.xlabel("photon energy (keV)")
    plt.ylabel("photons per "+str("%.2f" %delta_E)+" keV")
    plt.savefig("Spec.pdf")
    plt.clf()
def plotBanana(data,extent):
    plt.imshow(data,aspect="auto",extent=extent,norm=LogNorm())
    plt.colorbar
    plt.xlabel("photon energy (keV)")
    plt.ylabel("observation angle (mrad)")
    plt.savefig("Banana.pdf")
    plt.clf()



def read_SettingsFile(srcdir):
    data=dict()
    srcF=open(srcdir+"settings.hpp",'r')
    inpt=srcF.readlines()
    srcF.close()
    for line in inpt:
        if "const double omega_max" in line:
            omegaL=line.split()
            omega_max=np.float(eval(omegaL[4][:-1]))
            E_max=hbar_eVs*omega_max/1e3
            data["E_max"]=E_max
        elif "const double theta_max" in line:
            thetaL=line.split()
            theta_max=eval(thetaL[4][:-3])*np.pi/180
            data["theta_max"]=theta_max
        elif "const double phi_max" in line:
            phiL=line.split()
            phi_max=eval(phiL[4][:-3])*np.pi/180
            data["phi_max"]=phi_max
        elif "const unsigned int N_theta" in line:
            theta_stepsL=line.split()
            theta_steps=np.float(theta_stepsL[5][:-1])
            data["theta_steps"]=theta_steps
        elif "const unsigned int N_phi" in line:
            phi_stepsL=line.split()
            phi_steps=np.float(phi_stepsL[5][:-1])
            data["phi_steps"]=phi_steps
        elif "const unsigned int N_spectrum" in line:
            E_stepsL=line.split()
            E_steps=np.float(E_stepsL[5][:-1])
            data["E_steps"]=E_steps
        elif "const unsigned int N_trace" in line:
            N_simL = line.split()
            N_sim = np.int(N_simL[5][:-1])
            print("N_sim="+str(N_sim))
            data["N_sim"]=N_sim
    return data

def evaluate_allData(Q_bunch,theta_cone,BWlim,grid=False):
    """    
    """
    print(os.getcwd())
    #Make dictionary of all run directories
    runDict=dict()
    lst=sorted([elems for elems in os.listdir(".") if elems.startswith("run")])
    i=0

    for elem in lst:
        runDict[i]="run"+elem[3:]
        i+=1

    for dirs in runDict:
        print("------------------------")
        print(runDict[dirs])
        os.chdir(runDict[dirs])
        #Check if clara2 process data is there and load it
        if os.path.exists("my_spectrum_all_000.dat") and os.path.exists("my_spectrum_all_001.dat"):
            print("Load data ...")
            data_x=np.loadtxt("my_spectrum_all_000.dat")
            data_y=np.loadtxt("my_spectrum_all_001.dat")
        else:
            os.chdir("..")
            continue
        if grid==False:
            srcdir="src/"
        else:
            srcdir="src_grid/"
        #read data from settings header file in clara2
        settings=read_SettingsFile(srcdir)
        thetamax=settings["thetamax"]
        thetasteps=settings["thetasteps"]
        Emax=settings["Emax"]
        Esteps=settings["Esteps"]
        N_sim=settings["N_sim"]
        pixel_dOmega = (2*thetamax)**2/(thetasteps)**2 # sterradiant
        pixel_dE = Emax/(Esteps) # eV
        omega = np.linspace(0.0, Emax/hbar_eVs, Esteps)
        weighting = Q_bunch/e_charge / N_sim
        factor=np.zeros(len(omega))
        for i, om in enumerate(omega):
            if i==0:
                factor[i]=0
            else:
                factor[i]= weighting/(om*hbar_Js) * pixel_dOmega * (pixel_dE/hbar_eVs)
        radData = [data_x, data_y] * factor
        extent=(0, Emax, -1e3*thetamax, thetamax*1e3)

        for ii in range(0,2):
            Energy_arr = np.linspace(extent[0], extent[1], np.shape(radData[0])[1])
            theta_extent = np.linspace(extent[-2], extent[-1], np.shape(radData[0])[0])

        x_min=0
        x_max=Emax
        y_min=-1*theta_cone
        y_max=theta_cone
        #spectrum in cone in full energy range 
        spec_data = sumspectra(radData,y_min,y_max,x_min,x_max,theta_extent, pixel_dOmega, Energy_arr)
        dNdE = spec_data[2]
        E_arr = spec_data[3]
        FWHM=GDA.find_FWHM(E_arr,dNdE,out=False)
        E_Imax=FWHM[0]
        BW=FWHM[2]
        #number of photons in cone and within BWlim
        mu_eff=sumspectra(radData, y_min, y_max, E_Imax*(1-BWlim/2), E_Imax*(1+BWlim/2), theta_extent, pixel_dOmega, Energy_arr)[1]#FWHM given by BWlim
        if E_Imax<1: # if the spectrum peaks at low energies, exclude this region from the evaluation by restricting the energy range
            x_min=10
            print("Low limit = "+str(x_min)+" keV")
            x_max=Emax
            y_min=-1*theta_cone
            y_max=theta_cone
            spec_data = sumspectra(radData,y_min,y_max,x_min,x_max,theta_extent, pixel_dOmega, Energy_arr)
            dNdE = spec_data[2]
            E_arr = spec_data[3]
            FWHM=GDA.find_FWHM(E_arr,dNdE,out=False)
            E_Imax=FWHM[0]
            BW=FWHM[2]
            mu_eff=sumspectra(radData, y_min, y_max, E_Imax*(1-BWlim/2), E_Imax*(1+BWlim/2), theta_extent, pixel_dOmega, Energy_arr)[1]
            print("")
            print("spectrum in "+str("%.1f" %(y_max*1e3))+" mrad"+"\n"+"Epeak="+str("%.2f" %(E_Imax))+" keV"+"\n"+"photons in "+str(BWlim*100)+" % BW ="+str(int(mu_eff)))
            #print("")
            print("Full photon number = "+str(np.nansum(dNdE)))
            print("Full BW = "+str("%.2f" %(BW*100))+ "%")
            print("")
            os.chdir("../")
        else:
            print("")
            print("spectrum in "+str("%.1f" %(y_max*1e3))+" mrad"+"\n"+"Epeak="+str("%.2f" %(E_Imax))+" keV"+"\n"+"photon number in "+str(BWlim*100)+" % BW ="+str(int(mu_eff)))
            #print("")
            print("Full photon number = "+str(np.nansum(dNdE)))
            print("Full BW = "+str("%.2f" %(BW*100))+ "%")
            print("")
            os.chdir("../")
        #PlotSpectrum(dNdE,Energy_arr)
    #plotBanana(data_x,extent)
    return dNdE,Energy_arr,data_x,data_y,extent
