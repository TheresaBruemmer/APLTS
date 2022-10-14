"""
This Module collects all Astra python methods
"""
#import sys
#sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS")
import APLTS.utilities.GeneralDataAnalysis as GDA
import numpy as np
import math
import os
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt


def waist(z,w0,zR,z0):
    """
    not ASTRA specific
    fit function for a (laser or electron) focus
    """
    return w0*np.sqrt(1+((z-z0)/zR)**2)

def Ekin(gamma):
    #not ASTRA specific, simply kinetic energy (MeV) from gamma
    return (gamma-1)*0.511

def generatorEkin_gammarange(gammamin,gammamax):
    #input for generator file. returns Ekin and sig_Ekin for uniform energy dist. for a given gamma range
    gammamean=(gammamax+gammamin)/2
    E_kin=Ekin(gammamean)
    sig_Ekin=(E_kin-Ekin(gammamin))/np.sqrt(3)
    #sigpos=(Ekin(gammamax)-E_kin)/np.sqrt(3)
    return E_kin,sig_Ekin,gammamean

def generatorEkin(gammae,enspread):
    #input for generator file. returns Ekin and sig_Ekin for uniform energy dist. for a given energy FWHM spread around gamma
    E_kin=Ekin(gammae)
    gamma_min = gammae*(1-0.5*enspread)
    gamma_max = gammae*(1+0.5*enspread)
    sig_Ekin=(Ekin(gamma_max)-E_kin)/np.sqrt(3)
    return E_kin,sig_Ekin, gamma_min,gamma_max


def L_H(llambda, alpha, gamma, num):
    #Astra laser namelist time step setting in ns
    L=llambda/(num*(1-(1-1/gamma**2)**(0.5)*math.cos(alpha))*((1-1/gamma**2))**(0.5)*3e8)*1e9
    return L

def Laser_zstart(gamma,zF):
    #Calculates laser start param to ensure laser-electron-bunch overlap at zF
    beta=np.sqrt(1.0-1./gamma**2)
    L_zstart=zF*(1./beta-1.)
    return L_zstart

def load_PhSp(PhSp_filename):
    """
    Loads phase space into arrays
    x,y,z: position
    px,py,pz: momentum
    t: time
    """
    data=np.loadtxt(str(PhSp_filename))
    x=np.zeros(len(data))
    y=np.zeros(len(data))
    z=np.zeros(len(data))
    px=np.zeros(len(data))
    py=np.zeros(len(data))
    pz=np.zeros(len(data))
    t=np.zeros(len(data))
    for i, line in enumerate(data):
        if i==0:
            x[i]=line[0]
            y[i]=line[1]
            z[i]=line[2]
            px[i]=line[3]
            py[i]=line[4]
            pz[i]=line[5]
            t[i]=line[6]
        else:
            x[i]=line[0]+x[0]
            y[i]=line[1]+y[0]
            z[i]=line[2]+z[0]
            px[i]=line[3]+px[0]
            py[i]=line[4]+py[0]
            pz[i]=line[5]+pz[0]
            t[i]=line[6]+t[0]
    return x,y,z,px,py,pz,t

def plot_PhSp(path,filename):
    """
    plots phase space data and saves to pdf files
    """
    x,y,z,px,py,pz,t=load_PhSp(path+"/"+filename)
    print("zmin="+str(np.nanmin(z)))
    print("zmax="+str(np.nanmax(z)))
    plt.plot(x,px,".")
    plt.savefig("xpx_"+str(filename)+".pdf")
    plt.clf()
    plt.plot(z,pz,".")
    plt.savefig("zpz_"+str(filename)+".pdf")
    plt.clf()
    plt.plot(x,pz,".")
    plt.savefig("xpz_"+str(filename)+".pdf")
    plt.clf()
    plt.plot(x,z,".")
    plt.savefig("xz_"+str(filename)+".pdf")
    plt.clf()
    plt.plot(x,y,".",alpha = 0.4,markeredgecolor="None")
    plt.savefig("xz_"+str(filename)+".pdf")
    plt.clf()

def plot_spec(phasespace_filename = None):
    """
    plots the bunch spectrum from phase space file
    """
    if phasespace_filename:
        x,y,z,px,py,pz,t=load_PhSp(phasespace_filename)
        plt.hist(pz)
        plt.xlabel("pz")
        plt.savefig("pz_"+str(phasespace_filename)+".pdf")
        plt.clf()
        gamma_e = np.sqrt(px**2+py**2+pz**2)/(511e3)
        plt.hist(gamma_e)
        plt.xlabel('gamma_e')
        plt.savefig('spec_'+str(phasespace_filename)+'.pdf')
        plt.clf()
    else:
        print("No Phase Space given")
        pass

        

def list_PhSps(path, beginning_of_name):
    """
    Returns a list of PhSp names which start with beginning_of_name in the directory path
    e.g.: Astra file Astra.in creates phase spaces with naming convention Astra.*.001, thus beginning_of_name=Astra or Astra.
    """
    files=[]
    allfiles=list(os.walk(path))[0][2]
    for file in allfiles:
        if file.startswith(beginning_of_name):
            if file.endswith("001"):#is a PhSp
                if not file.endswith("emit.001") and not file.endswith("track.001") and not file.endswith("Log.001"):
                    files.append(file)
    files=sorted(files)
    return files


def track_PhSps(path,beginning_of_name):
    """
    calculates evolution of rms bunch size (x,y) with propagation distance z for set of phase spaces 
    path: path to phase spaces
    names of phase spaces in a set all start with "beginning_of_name"
    """
    files=list_PhSps(path,beginning_of_name)
    sigma_x=[]
    sigma_y=[]
    pos=[]
    for file in files:
        filedata=load_PhSp(str(path)+"/"+str(file))
        x=filedata[0]
        y=filedata[1]
        z=filedata[2]
        sigma_x.append(GDA.rms(x))
        sigma_y.append(GDA.rms(y))
        pos.append(np.mean(z))
    sigmax_arr=np.array(sigma_x)
    sigmay_arr=np.array(sigma_y)
    z_arr=np.array(pos)
    return sigmax_arr,sigmay_arr,z_arr

def find_Focus(path,name,z_lowlim,plotTrack=False):
    """
    Fits waist function to rms bunch size evolution and prints focal params. Data is saved to file anf plot pdf
    """
    sigmax_arr,sigmay_arr,z_arr=track_PhSps(path,name)
    poptx,pcovx=curve_fit(waist,(z_arr[np.nanargmin(abs(z_arr-z_lowlim)):]),sigmax_arr[np.nanargmin(abs(z_arr-z_lowlim)):])
    print("sigma_x_min="+str(poptx[0]))
    print("at position z="+str(poptx[2]))
    file=open(str(path)+"/"+str(name)+"_Focus.txt","w")
    file.write("sigma_x_min="+str(poptx[0])+"\n"+"at position z="+str(poptx[2]))
    file.close()
    if plotTrack==True:
        z_arr_fine=np.linspace(z_arr[0],z_arr[-1],1000)
        plt.plot(z_arr,sigmax_arr,".",label="x")
        plt.plot(z_arr_fine,waist(z_arr_fine,*poptx))
        plt.xlabel("z")
        plt.savefig(str(path)+"/"+str(name)+"_sigma.pdf")
        np.savetxt(str(path)+"/"+str(name)+"_sigma.txt",(z_arr,sigmax_arr))


def plot_Track(path,name):
    """
    Plots bunch size evolution along propagation z for given PhSp set
    """
    sigmax_arr,sigmay_arr,z_arr=track_PhSps(path,name)
    plt.plot(z_arr,sigmax_arr,".",label="x")
    plt.plot(z_arr,sigmay_arr,".",label="y")
    plt.xlabel("z")
    plt.legend()
    plt.savefig(str(path)+"/"+str(name)+"_sigma.pdf")
    np.savetxt(str(path)+"/"+str(name)+"_sigma.txt",(z_arr,sigmax_arr))




def make_Trajectories(path,filename):
    """
    writes trajectory file for each MP from Astra track file
    ----------------------------------
    track files are of form
    
    numMp  z  x \n
    y  bz  bx \n
    by  t \n
    
    trajectory_numMP.txt is of form
    
    x y z bx by bz t \n
    ---------------------------------
    """
    factor=np.array([1,1,1,1,1,1,1e-9]) # to account for time in ns
    #print(filename)
    #print(sorted([elems for elems in os.listdir(path+"/../") if elems.startswith(filename)]))
    lst=sorted([elems for elems in os.listdir(path+"/../") if elems.startswith(filename)])
    print(lst)
    numMPs =0
    for Th_file in lst:
        newfilename = "../"+str(Th_file)
        changefilename = "../read_" + str(Th_file)
        temp=0#This accounts for the linenumber in the write file
        Traj=dict()#initiate the dict for each file to save memory
        #open the current file
        f = open(newfilename,'r')
        data = f.readlines()
        f.close()
        print(newfilename)
        linenumber = int(len(data))
        #write data into array
        #Always 3 lines belong to the trajectory of one MP at one time step.
        #First entry of first of these lines is the number of the MP
        newdata = np.zeros((int(linenumber/3),8))
        for i in range(0,len(newdata)):
            dat_a = np.array(data[i*3].split(),dtype=float)
            dat_b = np.array(data[i*3+1].split(),dtype=float)
            dat_c = np.array(data[i*3+2].split(),dtype=float)
            newdata[i]=np.concatenate((dat_a,dat_b,dat_c),axis=0)#*factor
        #only for first file to find number of macroparticles
        if numMPs ==0:
            numMPs = int(np.nanmax(newdata[:,0]))
            print("Number of macroparticles = "+str(numMPs))
        # Write trajectories
        Traj = dict()
        for ID in range(1,numMPs+1):
            Traj[ID]=newdata[newdata[:,0]==ID][:,1:]*factor
            fout=open("trajectory_"+str("%05d" %ID)+".txt",'a')
            for i,temp in enumerate(Traj[ID]):
                for j,elem in enumerate(Traj[ID][i]):
                    fout.write(str("%.15e" %elem)+"\t")
                fout.write("\n")
            fout.close()
        os.rename(newfilename,changefilename)


def read_generator_info(gen_file,return_originaldata=False):
    #Is there a routine reading generator file info?
    #E_kin = 43.999
    #sig_E_kin=0.069
    #gamma=E_kin/0.511+1
    #dE_kin = sig_E_kin*2.*np.sqrt(3.) #width of the uniform electron spectrum
    #Q_bunch=dE_kin*1e-12 # 1pC per MeV
    #print("Q1 = "+str(Q_bunch))
    #N_sim=5000
    gendict={}
    datafile=open(gen_file,'r')
    data=datafile.readlines()
    datafile.close()
    for line in data:
        splitline = line.split()
        for elem in splitline:
            datas=elem.split("=")
            if len(datas)>1:
                if datas[0]!="":
                    gendict[datas[0]]=datas[1]
    #gendata dictionary looks like this:
    #{'Add': 'FALSE,', 'N_add': '0,', 'IPart': '5000,', 'Species': "'electrons'", 'Probe': 'True,', 'All_Probe': 'T,', 'Noise_reduc': 'F,', 'Cathode': 'F', 'Q_total': '0.02', 'Ref_zpos': '0.e-6,', 'Ref_Ekin': '37.133', 'Dist_z': "'gauss',", 'sig_z': '1.5e-3,', 'C_sig_z': '3.', 'Dist_pz': "'u',", 'sig_Ekin': '0.07E3,', 'cor_Ekin': '0.0E0,', 'C_sig_Ekin': '3.', 'Dist_x': "'gauss',", 'sig_x': '1.00e-3,', 'C_sig_x': '3.', 'Dist_px': "'g',", 'Nemit_x': '1.0,', 'cor_px': '0.0E0,', 'C_sig_px': '3.', 'Dist_y': "'gauss',", 'sig_y': '1.00e-3,', 'C_sig_y': '3.', 'Dist_py': "'g',", 'Nemit_y': '1.0,', 'cor_py': '0.0E0,', 'C_sig_py': '3.'}
    #write a nicer dict in SI units:
    #symmetrical bunch is assumed, i.e. x and y data assumed to be same
    bunch_data={}
    bunch_data[E_kin]=float(gendict['Ref_Ekin'])*1e6 #MeV to eV
    bunch_data[sig_E_kin]=float(gendict['sig_Ekin'])*1e3 #keV to eV
    bunch_data[gamma]=bunch_data[E_kin]/0.511e6+1
    bunch_data[dE_kin]=bunch_data[sig_E_kin]*2.*np.sqrt(3.) #width of the uniform electron spectrum
    bunch_data[Q_bunch]=float(gendict['Q_total'])*1e-9 #nC to C
    bunch_data[N_sim]=int(gendict['IPart'])
    bunch_data[eps_n]=float(gendict['Nemit_x'])*1e-6 #mmmrad to m rad
    bunch_data[sigma_r]=float(gendict['sig_x'])*1e-3 #mm to m
    bunch_data[sigma_z]=float(gendict['sig_z'])*1e-3 #mm to m
    bunch_data[energydist]=gendict(['Dist_pz']) #gaussian or uniform?
    if return_originaldata==False:
        return bunch_data
    else:
        return bunch_data, gendict
