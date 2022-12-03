import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE

from pylops.basicoperators import *
from pylops.optimization.sparsity import *
from pylops.signalprocessing import FFT2D,DWT2D
import pylops

from time import time

csfont = {'fontname':'Times New Roman'}

# def PSNR(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = np.max(img1)
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
    print (fmt+str(time() - _tstart_stack.pop()))
    
tic()

#Basic file settings
# =======================================================
epsz = 8.854187817*1e-12      # F/m
muz = 4*np.pi*1e-7               # H/m              
c = (epsz*muz)**(-1/2)         # m/s
eta = np.sqrt(muz/epsz)          # ohm

Nx = 51
Ny = 51
Nz = 13
N = Nx*Ny

dx = 7.2e-3 #meters
dy = 7.2e-3 #meters
dz = 3e-3 #meters

Fint =  5e9         # Hz
Fend =  15e9       # Hz
Fstep = 0.5e9        # Hz

Aperture_size = 360e-3 #Determine the desired data range for recontruction
Object_size = 18e-3
R = 60e-3
Nf = int((Fend-Fint)/Fstep+1)
Freq = np.linspace(Fint,Fend,Nf)
fc = Freq[int((len(Freq)-1)/2)]
kf = 2*np.pi*Freq/(3e8);
Z0 = 50 #ohm

# =======================================================

# ======================================================
#Load E field and Voc data, both represented in real space basis
# ======================================================
folderpath = "./Singleblock/numpy"
Iin = np.load(os.path.join(folderpath,"Iin.npy"))
g = np.load(os.path.join(folderpath,"g.npy"))
Voc = np.load(os.path.join(folderpath,"Voc.npy"))

#foldervoc = "D:/Desktop/Imaging/downsampling_zero/Crossblock/"
#foldervoc = "D:/Desktop/Imaging/downsample/Singleblock/"
#Voc = np.load(os.path.join(foldervoc,"Voc_ds_4.npy"))
#Voc = scipy.io.loadmat(os.path.join(foldervoc,"Voc_ds_4.mat"))
#Voc = Voc["Voc_ds"]
# ======================================================
g = g[::2,::2,:]
Voc = Voc[::2,::2,10]

xratio = np.arange(0,Nx,1)
yratio = np.arange(0,Ny,1)

Zint=42e-3
Zend=78e-3
zratio = np.arange(7,8,1) #The actual value of z is determined by the setting of the Feko Simulation !
z_pos = (Zint + (zratio*dz))*1e3; #Predetermined inspected z-cut

fratio = np.arange(10,11,1)  
kf = kf[fratio]
theta = np.arctan((Nx-1)/2*dx/R) ; #Angle related to Scanning Aperture  

SampMax = 2*np.pi*(Nx-1)/Nx/2/dx #Highest sampling wavenumber (1/m)

wavenum = 2*np.pi*2*Freq/3e8

    
#Construct I Z matrix
# ======================================================
Vin = 1

Tx = Nx*dx #Real space X axis
Ty = Ny*dy #Real space Y axis

dfx = np.round(1/Tx,decimals=5)
dfy = np.round(1/Ty,decimals=5) #Use round for Floating-point error mitigation

xx,yy = np.meshgrid(np.arange(-(Nx-1)/2*dx,(Nx-1)/2*dx+dx,dx),np.arange(-(Ny-1)/2*dy,(Ny-1)/2*dy+dy,dy)) #Real space X,Y window
fxx,fyy = np.meshgrid(np.arange(-(Nx-1)/2*dfx,(Nx-1)/2*dfx+dfx,dfx),np.arange(-(Ny-1)/2*dfy,(Ny-1)/2*dfy+dfy,dfy)) #2pi convert to sampling wavenumber
FXX,FYY = np.meshgrid(2*np.pi*fxx[0,:],2*np.pi*fyy[:,0])
fzz = np.conj(np.sqrt(np.transpose(np.tile(np.reshape((2*Freq/c)**2,(1,1,1,len((2*Freq/c)**2))),(Ny,Nx,Nz,1)),(3,2,1,0))-np.tile((fxx**2 + fyy**2),(Nf,Nz,1,1)),dtype="complex128"))
# =================

if __name__ == "__main__":
    #go = 0
    #end = 51
    go = 12
    end = 39
    # ++++++++Ground Truth+++++++++++++++++++++++++++
    # gt = np.empty((Nx,Ny)); gt.fill(1)
    # #==Single Block===
    # gt[24:27,24:27] = 2
    # #=================
    # #==Cross Block====
    # #gt[23:28,25] = 2
    # #gt[25,23:28] = 2
    # #=================
    # #gt = (gt*255/np.max(gt)).astype(int)
    # gt = gt*100
    # +++++++++++++++++++++++++++++++++++++++++++++++
    
    g = g[:,:,zratio,fratio]
    G = np.fft.fft2(g,axes=(0,1))
    G = np.fft.fftshift(np.fft.fftshift(G,0),1)
    
    const = -1j*2*np.pi*Freq[fratio]*dz/Iin[:,:,fratio]
    Gprime = np.multiply(const,G)
    
    idmat = np.eye(Nx*Ny)
    idmat = np.tile(idmat,(len(Gprime[0,0,:]),1))
    
    Gmat = []
    for i in range(len(Gprime[1,1,:])):
        Gprimer = np.copy(Gprime[:,:,i]).reshape((Nx*Ny*len(fratio),1),order='F')
        temp = idmat*Gprimer
        Gmat.append(temp)
    Gmat = np.array(Gmat)
    Gmat = Gmat.transpose((1,2,0))
    Gmat = Gmat.reshape((Nx*Ny*len(fratio),-1))
    
    # ++++++++Ground Truth+++++++++++++++++++++++++++
    # T = np.fft.fft2(Voc,axes=(0,1))
    # T = np.fft.fftshift(np.fft.fftshift(T,0),1)  
    # Tvec = T.reshape((Nx*Ny*len(fratio),1))
            
    # X,res,rank,s = linalg.lstsq(Gmat,Tvec)
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # kc = (2*np.pi)*fc/(2.9979e8);
    # for k in range(len(z_pos)):
    #     theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
    #     limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
    #     for i in range(Ny):
    #         for j in range(Nx):
    #             if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
    #                 P[i,j,k] = 0
    # F = np.fft.ifftshift(np.fft.ifftshift(P,0),1)
    # f = np.fft.ifft2(F,axes = (0,1))
    # f = np.fft.fftshift(f,axes = (0,1))
    # gt = np.squeeze(np.abs((f/epsz/(dx*dy))+1))
    # gt = (gt*100).astype(int)
    
    gt = np.empty((Nx,Ny)); gt.fill(1)
    # # #==Single Block===
    gt[24:27,24:27] = 2
    # # #=================
    # # #==Cross Block====
    #gt[23:28,25] = 2
    #gt[25,23:28] = 2
    # # #=================
    # +++++++++++++++++++++++++++++++++++++++++++++++
    
    avg_psnr = []
    avg_ssim = []
    avg_epsr = []
    no_psnr = []
    no_ssim = []
    no_epsr = []
    Xaxis = []
    for comp in range (5,105,5):
        Xaxis.append(comp)
        cs_avg_psnr = []
        cs_avg_ssim = []
        cs_avg_epsr = []
        no_avg_psnr = []
        no_avg_ssim = []
        no_avg_epsr = []
        
        for i in range (20):
            #======Generate Sensing Matrix ==========
            comprate = comp*0.01
            Nsub = int(np.round(N*comprate))
            iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])
            mask = np.zeros(N)
            mask[iava] = 1
            Maskop = Diagonal(mask.ravel())
            Rop = Restriction(N, iava, dtype=np.complex128)
            #========================================
            
            #======Solve CS by SPGL1 ================
            FFT2op = FFT2D((Ny,Nx))
            Voc_holes = Maskop * Voc.ravel()
            #Voc_holes_visual =Maskop * Voc.ravel()
    
            Voc_filled, T_filled, info = spgl1(Maskop, Voc_holes,SOp=FFT2op , tau=0, iter_lim=200, iscomplex=True)

            Voc_filled = Voc_filled.reshape(Ny,Nx)
            Voc_holes_visual = Voc_holes.reshape(Ny,Nx)
            #========================================
            
            T = np.fft.fft2(Voc_filled,axes=(0,1))
            T = np.fft.fftshift(np.fft.fftshift(T,0),1)  
            Tvec = T.reshape((Nx*Ny*len(fratio),1))
            
            X,res,rank,s = linalg.lstsq(Gmat,Tvec)
            P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
            kc = (2*np.pi)*fc/(2.9979e8);
            for k in range(len(z_pos)):
                theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
                limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
                for i in range(Ny):
                    for j in range(Nx):
                        if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                            P[i,j,k] = 0
            P[int(Ny/2),int(Nx/2),:]=0
            F = np.fft.ifftshift(np.fft.ifftshift(P,0),1)
            f = np.fft.ifft2(F,axes = (0,1))
            f = np.fft.fftshift(f,axes = (0,1))
            image = (f/epsz/(dx*dy))+1
            image = np.squeeze(abs(image))

               
            psnr_CS = PSNR(gt[go:end,go:end],image[go:end,go:end],data_range=np.max(gt))
            ssim_CS = SSIM(gt[go:end,go:end],image[go:end,go:end])
            epsr_CS = np.abs(image[25,25])
            cs_avg_psnr.append(psnr_CS)
            cs_avg_ssim.append(ssim_CS)
            cs_avg_epsr.append(epsr_CS)
            
            T2 = np.fft.fft2(Voc_holes_visual,axes=(0,1))
            T2 = np.fft.fftshift(np.fft.fftshift(T2,0),1)  
            Tvec2 = T2.reshape((Nx*Ny*len(fratio),1))
            
            X2,res,rank,s = linalg.lstsq(Gmat,Tvec2)
            P2 = X2.reshape(-1,Nx,Ny).transpose(1,2,0)
            for k in range(len(z_pos)):
                theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
                limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
                for i in range(Ny):
                    for j in range(Nx):
                        if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                            P2[i,j,k] = 0
            P2[int(Ny/2),int(Nx/2),:]=0
            F2 = np.fft.ifftshift(np.fft.ifftshift(P2,0),1)
            f2 = np.fft.ifft2(F2,axes = (0,1))
            f2 = np.fft.fftshift(f2,axes = (0,1))
            image2 = (f2/epsz/(dx*dy))+1
            image2 = np.squeeze(abs(image2))

            
            psnr_no = PSNR(gt[go:end,go:end],image2[go:end,go:end],data_range=np.max(gt))
            ssim_no = SSIM(gt[go:end,go:end],image2[go:end,go:end])
            epsr_no = np.abs(image2[25,25])
            no_avg_psnr.append(psnr_no)
            no_avg_ssim.append(ssim_no)
            no_avg_epsr.append(epsr_no)
        cs_avg_psnr = sum(cs_avg_psnr)/len(cs_avg_psnr)
        cs_avg_ssim = sum(cs_avg_ssim)/len(cs_avg_ssim)
        cs_avg_epsr = sum(cs_avg_epsr)/len(cs_avg_epsr)
        no_avg_psnr = sum(no_avg_psnr)/len(no_avg_psnr)
        no_avg_ssim = sum(no_avg_ssim)/len(no_avg_ssim)
        no_avg_epsr = sum(no_avg_epsr)/len(no_avg_epsr)
        
        avg_psnr.append(cs_avg_psnr)
        avg_ssim.append(cs_avg_ssim)
        avg_epsr.append(cs_avg_epsr)
        no_psnr.append(no_avg_psnr)
        no_ssim.append(no_avg_ssim)
        no_epsr.append(no_avg_epsr)
    
    psnr_plot = np.array(avg_psnr)
    ssim_plot = np.array(avg_ssim)
    epsr_plot = np.array(avg_epsr)
    psnr2_plot = np.array(no_psnr)
    ssim2_plot = np.array(no_ssim)
    epsr2_plot = np.array(no_epsr)
    Xaxis = np.array(Xaxis)
    
    fig1,ax1 = plt.subplots(dpi=1200)
    ax1.set_xlabel('Compression Rate (%)',**csfont,fontsize=16)
    ax1.set_ylabel('PSNR (dB)', color='orangered',**csfont,fontsize=16)
    ax1.plot(Xaxis, psnr_plot, color='orangered',label='PSNR, CS recovered')
    ax1.plot(Xaxis, psnr2_plot,'--', color='orangered',label='PSNR, under-sampled')
    ax1.tick_params(axis='y', labelcolor='orangered',labelsize=13)
    ax1.tick_params(axis='x',labelsize=13)
    plt.yticks(np.arange(15,40,5)) 
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('SSIM', color='darkblue',**csfont,fontsize=16)
    ax2.plot(Xaxis, ssim_plot, color='darkblue',label='SSIM, CS recovered')
    ax2.plot(Xaxis, ssim2_plot,'--', color='darkblue',label='SSIM, under-sampled')
    ax2.tick_params(axis='y', labelcolor='darkblue',labelsize=13)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax2.legend(lines+lines2, labels+labels2, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2) #fancybox=True
    plt.xticks(np.arange(0,101,20))
    plt.yticks(np.arange(0.1,1,0.2)) 
    
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    fig1.tight_layout()
    plt.show()
    
    fig2,ax1 = plt.subplots(dpi=1200)
    ax1.set_xlabel('Compression Rate (%)',**csfont,fontsize=16)
    ax1.set_ylabel('Permittivity', color='darkgreen',**csfont,fontsize=16)
    ax1.plot(Xaxis, epsr_plot, color='darkgreen',label='Epsr, CS recovered')
    ax1.plot(Xaxis, epsr2_plot,'--', color='darkgreen',label='Epsr, under-sampled')
    ax1.tick_params(axis='y', labelcolor='darkgreen',labelsize=13)
    ax1.tick_params(axis='x',labelsize=13)
    plt.yticks(np.arange(0.8,2.1,0.2))
    plt.xticks(np.arange(0,110,20))
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc=0)
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.4)
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    fig1.tight_layout()
    plt.show()
    
    toc()