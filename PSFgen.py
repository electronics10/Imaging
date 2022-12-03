# Generate 3D Voc-based PSF, analogous to the vector E field matrix

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import linalg

csfont = {'fontname':'Times New Roman'}
#Basic file settings
# =======================================================
epsz = 8.854187817*1e-12      # F/m
muz = 4*np.pi*1e-7               # H/m              
c = (epsz*muz)**(-1/2)         # m/s
eta = np.sqrt(muz/epsz)          # ohm

Nx = 101
Ny = 101
Nz = 13

dx = 3.6e-3 #meters
dy = 3.6e-3 #meters
dz = 3e-3 #meters

Fint =  5e9         # Hz
Fend =  15e9       # Hz
Fstep = 0.5e9        # Hz

Aperture_size = 360e-3 #Determine the desired data range for recontruction
Object_size = 0.03
R = 66e-3
Nf = np.int((Fend-Fint)/Fstep+1)
Freq = np.linspace(Fint,Fend,Nf)
fc = Freq[int((len(Freq)-1)/2)]
kf = 2*np.pi*Freq/c;
Z0 = 50 #ohm

Vin = 1
# =======================================================

folderpath = "C:\\Users\\ELECTRONICS10\\Desktop\\Code_Data\\Singleblock3\\numpy"
Iin = np.load(os.path.join(folderpath,"Iin.npy"))
g = np.load(os.path.join(folderpath,"g.npy"))

psfpath = "C:\\Users\\ELECTRONICS10\\Desktop\\Code_Data\\PSF\\npy"
Voc_psf = np.load(os.path.join(psfpath,"PSF.npy"))

xratio = np.arange(0,Nx,1)
yratio = np.arange(0,Ny,1)

Zint=42e-3
Zend=78e-3
zratio = np.arange(0,13,1) #The actual value of z is determined by the setting of the Feko Simulation !
z_pos = (Zint + ((zratio)*dz))*1e3; #Predetermined inspected z-cut

fratio = np.arange(0,21,1)
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

if __name__ == "__main__":
    G = np.fft.fft2(g,axes=(0,1))
    G = np.fft.fftshift(np.fft.fftshift(G,0),1)
    
    const = -1j*2*np.pi*Freq[fratio]*dz/Iin[:,:,fratio]
    Gprime = np.multiply(const,G)
    
    volume = dx*dy*dz
    T = np.fft.fft2(Voc_psf,axes=(0,1))
    T = np.fft.fftshift(np.fft.fftshift(T,0),1)
    
    prime = 1j*Iin[:,:,fratio]/(2*np.pi*Freq[fratio]*epsz*volume)
    test = np.multiply(Voc_psf,prime)
    test = np.imag(test[:,:,10])
    PSF_spec = np.multiply(T,prime)
    #PSF_spec[50,50,:] = (PSF_spec[49,49,:]+PSF_spec[49,50,:]+PSF_spec[49,51,:]+PSF_spec[50,49,:]+PSF_spec[50,51,:]+PSF_spec[51,49,:]+PSF_spec[51,50,:]+PSF_spec[51,51,:])/8
    PSF_spec[50,50,:] = 0
    psf = np.fft.ifft2(PSF_spec,axes=(0,1))
    psf = np.fft.fftshift(np.fft.fftshift(psf,0),1)
    #psf = np.divide(Voc_psf,volume*epsz)
    
    # PSF_spec = np.fft.fft2(psf,axes=(0,1))
    # PSF_spec = np.fft.fftshift(np.fft.fftshift(PSF_spec,0),1)
    kz = []
    for f in range(len(fratio)):
        kz.append(np.sqrt(kf[f]**2-fxx**2-fxx**2))
    kz = np.array(kz).transpose(1,2,0)
    kz[np.isnan(kz)]=0
    kz_mat = []
    for z in range(-7,6,1):
        kz_mat.append(-1j*kz*z*dz)
    kz_mat = np.array(kz_mat).transpose(1,2,0,3)
    phase_mat = np.exp(kz_mat)
    
    
    transfer = []
    for z in range(len(phase_mat[0,0,:,0])):
        transfer.append(np.multiply(PSF_spec,phase_mat[:,:,z,:]))
    transfer = np.array(transfer).transpose(1,2,0,3)
    
    kernel = np.fft.fftshift(np.fft.fftshift(transfer,0),1)
    kernel = np.fft.ifft2(kernel,axes=(0,1))
    
    
    gamp = abs(g[:,:,7,10])
    psf_amp = abs(kernel[:,:,7,10])
    diff_amp = abs(gamp-psf_amp)/abs(gamp)*100
    
    
    fig = plt.figure(dpi=1200)
    plt.pcolormesh(xx*1000,yy*1000,np.abs(gamp),shading='auto',cmap='jet')
    plt.xticks(xx[0,:]*1000, fontsize=12)
    plt.yticks(yy[:,0]*1000, fontsize=12)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)",**csfont,fontsize=16)
    plt.ylabel("y (mm)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.clim(1,110)
    plt.colorbar()
    
    # fig1 = plt.figure()
    # plt.imshow(psf_amp)
    # plt.title("Amplitude of calculated PSF")
    # plt.colorbar()
    
    fig1 = plt.figure(dpi=1200)
    plt.pcolormesh(xx*1000,yy*1000,psf_amp,shading='auto',cmap='jet')
    plt.xticks(xx[0,:]*1000, fontsize=12)
    plt.yticks(yy[:,0]*1000, fontsize=12)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)",**csfont,fontsize=16)
    plt.ylabel("y (mm)",**csfont,fontsize=16)
    plt.clim(1,110)
    # plt.title("Amplitude of calculated PSF")
    plt.colorbar()
    
    fig2 = plt.figure(dpi=1200)
    plt.pcolormesh(xx*1000,yy*1000,diff_amp,shading='auto',cmap='gist_gray')
    plt.xticks(xx[0,:]*1000, fontsize=12)
    plt.yticks(yy[:,0]*1000, fontsize=12)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)",**csfont,fontsize=16)
    plt.ylabel("y (mm)",**csfont,fontsize=16)
    plt.title("Percentage Error (%)",**csfont,fontsize=16)
    plt.clim(1,100)
    plt.colorbar()
    
    gphase = np.angle(g[:,:,7,10], deg=True)
    psf_phase = np.angle(kernel[:,:,7,10], deg=True)
    diff_phase = abs(gphase-psf_phase)
    
    fig = plt.figure()
    plt.imshow(gphase)
    plt.title("Phase of simulated Einc dot Einc")
    plt.colorbar()
    
    fig1 = plt.figure()
    plt.imshow(psf_phase)
    plt.title("Phase of calculated PSF")
    plt.colorbar()
    
    fig2 = plt.figure()
    plt.imshow(diff_phase)
    plt.title("Phase Difference")
    plt.clim(1,100)
    plt.colorbar()
    
    Gamp = abs(G[:,:,7,10])
    PSF_amp = abs(transfer[:,:,7,10])
    Diff_amp = abs(Gamp-PSF_amp)/abs(Gamp)*100
    
    fig = plt.figure(dpi=1200)
    # plt.imshow(Gamp)
    # plt.title("Amplitude of simulated 2DFT{Einc dot Einc}")
    # plt.colorbar()
    plt.pcolormesh(FXX,FYY,np.abs(Gamp),shading='auto',cmap='jet')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.clim(0,8000)
    plt.axis('square')
    plt.colorbar()
    
    # fig1 = plt.figure()
    # plt.imshow(PSF_amp)
    # plt.title("Amplitude of calculated 2DFT{PSF}")
    # plt.colorbar()
    fig = plt.figure(dpi=1200)
    # plt.imshow(Gamp)
    # plt.title("Amplitude of simulated 2DFT{Einc dot Einc}")
    # plt.colorbar()
    plt.pcolormesh(FXX,FYY,np.abs(PSF_amp),shading='auto',cmap='jet')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.clim(0,8000)
    plt.axis('square')
    plt.colorbar()
    
    fig3 = plt.figure()
    plt.plot(PSF_amp[50,:])
    plt.title("Horizontal cut of the PSF spectrum amplitude")
    
    fig4 = plt.figure()
    plt.plot(Gamp[50,:])
    plt.title("Horizontal cut of the G spectrum amplitude")
    
    fig2 = plt.figure(dpi=1200)
    plt.pcolormesh(FXX,FYY,Diff_amp,shading='auto',cmap='gist_gray')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.title("Percentage Error (%)",**csfont,fontsize=16)
    plt.axis('square')
    #plt.title("Amplitude error percentage (%)")
    plt.clim(0,100)
    plt.colorbar()
    
    Gphase = np.angle(G[:,:,7,10], deg=True)
    PSF_phase = np.angle(transfer[:,:,7,10], deg=True)
    Diff_phase = abs(Gphase-PSF_phase)
    
    fig = plt.figure(dpi=1200)
    #plt.imshow(Gphase)
    #plt.title("Phase of simulated 2DFT{Einc dot Einc}")
    plt.pcolormesh(FXX,FYY,Gphase,shading='auto')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.clim(-180,180)
    plt.axis('square')
    plt.colorbar()
    
    fig1 = plt.figure(dpi=1200)
    #plt.imshow(PSF_phase)
    #plt.title("Phase of calculated 2DFT{PSF}")
    plt.pcolormesh(FXX,FYY,PSF_phase,shading='auto')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    plt.clim(-180,180)
    plt.axis('square')
    plt.colorbar()
    
    fig2 = plt.figure(dpi=1200)
    #plt.imshow(Diff_phase)
    #plt.title("Phase Difference")
    plt.pcolormesh(FXX,FYY,Diff_phase,shading='auto',cmap='gist_gray')
    plt.xticks(FXX[0,:], fontsize=12)
    plt.yticks(FYY[:,0], fontsize=12)
    plt.locator_params(axis='y', nbins=3)
    plt.locator_params(axis='x', nbins=3)
    plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
    plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
    #plt.title("Amplitude of simulated Einc dot Einc")
    #plt.clim(-150,150)
    plt.title("Phase difference (deg)",**csfont,fontsize=16)
    plt.axis('square')
    #plt.clim(1,100)
    plt.colorbar()
    