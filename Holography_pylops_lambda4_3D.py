import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from scipy import linalg
import math
import scipy.io
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from pylops.basicoperators import *
from pylops.optimization.sparsity import SPGL1
from pylops.signalprocessing import FFT2D,DWT2D
import pylops

from PIL import Image
from SSIM_PIL import compare_ssim
from time import time

_tstart_stack = []
csfont = {'fontname':'Times New Roman'}
def tic():
    _tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
    print (fmt+str(time() - _tstart_stack.pop()))
    
def spiral_points(arc=1, separation=1):
    """generate points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive 
    turnings
    - approximate arc length with circle arc at given distance
    - use a spiral equation r = b * phi
    """
    def p2c(r, phi):
        """polar to cartesian
        """
        return (r * math.cos(phi), r * math.sin(phi))

    # yield a point at origin
    yield (0, 0)

    # initialize the next point in the required distance
    r = arc
    b = separation / (2 * math.pi)
    # find the first phi to satisfy distance of `arc` to the second point
    phi = float(r) / b
    while True:
        yield p2c(r, phi)
        # advance the variables
        # calculate phi that will give desired arc length at current radius
        # (approximating with circle)
        phi += float(arc) / r
        r = b * phi

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

dx = 7.2e-3 #meters
dy = 7.2e-3 #meters
dz = 3e-3 #meters

Fint =  5e9         # Hz
Fend =  15e9       # Hz
Fstep = 0.5e9        # Hz

Aperture_size = 360e-3 #Determine the desired data range for recontruction
Object_size = 18e-3
R = 63e-3
Nf = int((Fend-Fint)/Fstep+1)
Freq = np.linspace(Fint,Fend,Nf)
fc = Freq[int((len(Freq)-1)/2)]
kf = 2*np.pi*Freq/(3e8)
Z0 = 50 #ohm

# =======================================================

# ======================================================
#Load E field and Voc data, both represented in real space basis
# ======================================================
folderpath = "Crossblock2/numpy"
Iin = np.load(os.path.join(folderpath,"Iin.npy"))
g = np.load(os.path.join(folderpath,"g.npy"))
Voc = np.load(os.path.join(folderpath,"Voc.npy"))

#foldervoc = "D:/Desktop/Imaging/noisy_Voc/Four_blocks/"
#Voc = np.load(os.path.join(foldervoc,"Voc_noisy_10.npy"))
#Voc = scipy.io.loadmat(os.path.join(foldervoc,"Voc_noisy_10.mat"))
#Voc = Voc["noisy_M"]
# ======================================================
g = g[::2,::2,:,:]
Voc = Voc[::2,::2,:]

xratio = np.arange(0,Nx,1)
yratio = np.arange(0,Ny,1)

Zint=42e-3
Zend=78e-3
zratio = np.arange(6,9,1) #The actual value of z is determined by the setting of the Feko Simulation !
z_pos = (Zint + (zratio*dz))*1e3; #Predetermined inspected z-cut

fratio = np.arange(3,18,1)  
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
#Conjugate for consistent to evanescent wave
# Removed positions
pick = 0
# =================

if __name__ == "__main__":
    #mag_go = 14
    #mag_end = 35
    
    mag_go = 0
    mag_end = 101
    
    Voc = Voc[mag_go:mag_end,mag_go:mag_end,fratio]
    Ny,Nx = Voc[:,:,0].shape
    
    ####==== Random Undersample ====###
    comprate = 0.3
    N = Nx*Ny
    Nsub = int(np.round(N*comprate))
    iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])
    mask = np.zeros(N)
    mask[iava] = 1
    Maskop = Diagonal(mask.ravel())
    mask = mask.reshape((Ny,Nx))
    Rop = Restriction(N, iava, dtype=np.complex128)
    ####============================###
    
    ####==== Random Line Undersample ====###
    # comprate = 0.5
    # N = Nx*Ny
    # Nsub = int(np.round(Nx*comprate))
    # iava = np.sort(np.random.permutation(np.arange(Nx))[:Nsub])
    # mask = np.zeros((Nx,Ny))
    # mask[iava,:] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####============================###
    ####==== Equidistance Spiral ====###
    # spiral = spiral_points(1.2,3)
    # sample = 2000
    # N = Nx*Ny
    # lst = []
    # for i in range(sample):
    #     lst.append(next(spiral))
    # lst = np.array(lst)+Nx/2
    # lst = np.round(lst).astype(int)
    # lst = lst[(0<lst[:,0]) & (lst[:,0]<Nx),:]
    # lst = lst[(0<lst[:,1]) & (lst[:,1]<Nx),:]
    # mask = np.zeros((Nx,Ny))
    # mask[lst[:,0],lst[:,1]] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####=============================###
    ####==== Spiral Undersample ====###
    # b = 0.65
    # N = Nx*Ny
    # mask = np.zeros((Nx,Ny))
    # theta = np.radians(np.linspace(0,360*14,3000))
    # r = b*theta
    # x_traj = np.round(r*np.cos(theta)+Nx/2).astype(int)
    # #x_traj = r*np.cos(theta)+Nx/2
    # #x_traj = np.round(x_traj+0.7*np.random.randn(len(x_traj))).astype(int)
    
    # y_traj = np.round(r*np.sin(theta)+Ny/2).astype(int)
    # #y_traj = r*np.sin(theta)+Ny/2
    # #y_traj = np.round(y_traj+0.7*np.random.randn(len(y_traj))).astype(int)
    # lst = np.stack((y_traj,x_traj)).transpose(1,0)
    # lst = lst[(0<lst[:,0]) & (lst[:,0]<Nx),:]
    # lst = lst[(0<lst[:,1]) & (lst[:,1]<Nx),:]
    # mask[lst[:,0],lst[:,1]] = 1
    # #mask[:,y_traj] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    # fig = plt.figure(dpi=1200)
    # plt.pcolormesh(xx*1000,yy*1000,mask,shading='auto',cmap='gist_gray')
    # plt.xticks(xx[0,:]*1000, fontsize=8)
    # plt.yticks(yy[:,0]*1000, fontsize=8)
    # #plt.clim(1,np.max(gt))
    # plt.locator_params(axis='y', nbins=7)
    # plt.locator_params(axis='x', nbins=7)
    # plt.xlabel("x (mm)",**csfont,fontsize=12)
    # plt.ylabel("y (mm)",**csfont,fontsize=12)
    # plt.axis('square')
    # plt.show
    ####============================###
    
    ####====Vogel Undersample ====####
    # n = 2000
    # radius = Nx/1.5*np.sqrt(np.arange(n) / float(n))
    # golden_angle = np.pi * (3 - np.sqrt(5))
    # theta = golden_angle * np.arange(n)
    # N = Nx*Ny
    # lst = np.zeros((n, 2))
    # lst[:,0] = np.cos(theta)
    # lst[:,1] = np.sin(theta)
    # lst *= radius.reshape((n, 1))
    # lst = np.round(lst+Nx/2).astype(int)
    # lst = lst[(0<lst[:,0]) & (lst[:,0]<Nx),:]
    # lst = lst[(0<lst[:,1]) & (lst[:,1]<Nx),:]
    # mask = np.zeros((Nx,Ny))
    # mask[lst[:,0],lst[:,1]] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####==========================####
    
    ####==== Squared-spiral Undersample ====###
    # N = Nx*Ny
    # repeates = 12
    # step = 2
    # directions = [(step,0),(0,step),(-step,0),(0,-step)]

    # moves = [(3+2*(i//2))*np.array(d) 
    #     for i,d in enumerate(itertools.chain(*itertools.repeat(directions, repeates)))]
    # points = (np.array([0,0]),*itertools.accumulate(moves))

    # coordinates = np.array(points).reshape(-1)
    # r1,r2 = coordinates.min(), coordinates.max()
    # n = r2-r1+1
    # mask = np.zeros((Nx,Ny))

    # for p,q in zip(points[0:-1],points[1:]):
    #     cv2.line(mask, tuple(p-r1), tuple(q-r1), (1,1,1))
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####====================================###
    
    ####==== Uniform undersample ====###
    # N = Nx*Ny
    # mask = np.zeros((Nx,Ny))
    # mask[0::2,0::2] = 1
    # #mask[1::3,:] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    
    ####=============================###
    #FFT2op = DWT2D((Ny,Nx), wavelet="db20", level=4)
    FFT2op = FFT2D((Ny,Nx))
    #Tvec_holes = FFT2op * Voc.ravel()
    
    A_mat = Rop * FFT2op.H  * np.identity(Nx*Ny)
    A_real = np.real(A_mat)
    A_imag = np.imag(A_mat)
    A_norm = np.linalg.norm(A_mat)
    
    Voc_holes = []
    Voc_holes_visual = []
    for f in range(len(fratio)):
        Voc_holes_freq = Rop * Voc[:,:,f].ravel()
        Voc_holes_visual_freq = Maskop * Voc[:,:,f].ravel()
        Voc_holes_visual_freq = Voc_holes_visual_freq.reshape((Ny,Nx))
        Voc_holes.append(Voc_holes_freq)
        Voc_holes_visual.append(Voc_holes_visual_freq)
    Voc_holes = np.array(Voc_holes)
    Voc_holes_visual = np.array(Voc_holes_visual).transpose((1,2,0))
    
    Voc_filled = []
    for f in range(len(fratio)):
        Voc_filled_temp, T_filled_temp, info = SPGL1(Rop, Voc_holes[f,:],SOp=FFT2op.H , tau=0, iter_lim=200, iscomplex=True)
        Voc_filled_temp = Voc_filled_temp.reshape((Ny,Nx))
        Voc_filled.append(Voc_filled_temp)
    Voc_filled = np.array(Voc_filled).transpose((1,2,0))
    
    fig = plt.figure()
    plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.abs(Voc[:,:,-1]),shading='auto')
    plt.xticks(xx[0,mag_go:mag_end]*1000)
    plt.yticks(yy[mag_go:mag_end,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Full Voc array")
    
    fig = plt.figure()
    plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.abs(Voc_holes_visual[:,:,-1]),shading='auto')
    plt.xticks(xx[0,mag_go:mag_end]*1000)
    plt.yticks(yy[mag_go:mag_end,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Undersampled Voc array")
    
    fig = plt.figure()
    plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.real(Voc_filled[:,:,-1]),shading='auto')
    plt.xticks(xx[0,mag_go:mag_end]*1000)
    plt.yticks(yy[mag_go:mag_end,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Recovered Voc array")
    
    diff = (np.abs(Voc[:,:,-1])-np.abs(Voc_filled[:,:,-1]))/np.abs(Voc[:,:,-1])*100
    fig = plt.figure()
    plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,diff,shading='auto',cmap='jet')
    plt.xticks(xx[0,mag_go:mag_end]*1000)
    plt.yticks(yy[mag_go:mag_end,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Error Percentage (%)")
    
    #T = np.fft.fft2(Voc_holes_visual,axes=(0,1))
    #T = np.fft.fft2(Voc,axes=(0,1))
    T = np.fft.fft2(Voc_filled,axes=(0,1))
    T = np.fft.fftshift(np.fft.fftshift(T,0),1)
    
    g = g[mag_go:mag_end,mag_go:mag_end,:,fratio]
    g = g[:,:,zratio,:]
    G = np.fft.fft2(g,axes=(0,1))
    G = np.fft.fftshift(np.fft.fftshift(G,0),1)
    
    const = -1j*2*np.pi*Freq[fratio]*dz/Iin[:,:,fratio]
    Gprime = np.multiply(const,G)
    ####====Solve Least squares====###
    A = np.transpose(Gprime,(3,2,0,1))
    A = np.concatenate((A,np.conj(A)),axis = 0)
    B = np.transpose(T,(2,0,1))
    B = np.concatenate((B,np.conj(B)),axis = 0)
    
    P = []
    for i in range(Ny):
        for j in range(Nx):
            x,res,rank,s = linalg.lstsq(A[:,:,i,j],B[:,i,j])
            P.append(x)
    P = np.array(P)
    P = P.reshape((Ny,Nx,len(zratio)))
    
    kc = (2*np.pi)*fc/c;
    for k in range(len(z_pos)):
        theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
        #limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
        limit = 2*kc*np.sin(theta)/((2*np.pi*(Nx-1))/(2*Nx*dx))
        for i in range(Ny):
            for j in range(Nx):
                if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                    P[i,j,k] = 0
    ####============================###
    F = np.fft.ifftshift(np.fft.ifftshift(P,0),1)
    f_raw = np.fft.ifft2(F,axes=(0,1))
    f_raw = np.fft.fftshift(np.fft.fftshift(f_raw,0),1)
    epsr = (f_raw/epsz/(dx*dy))+1
    
    for i in range(len(z_pos)):
        fig = plt.figure(dpi=1200)
        plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.real(epsr[:,:,i]),shading='auto',cmap='jet')
        plt.xticks(xx[0,mag_go:mag_end]*1000, fontsize=12)
        plt.yticks(yy[mag_go:mag_end,0]*1000, fontsize=12)
        #plt.clim(1,np.max(np.real(epsr)))
        plt.clim(1,2)
        #plt.clim(1,1.3)
        #Since we should have epsilon larger than 1
        #plt.colorbar()
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
        plt.xlabel("x (mm)",**csfont,fontsize=16)
        plt.ylabel("y (mm)",**csfont,fontsize=16)
        plt.axis('scaled')
        plt.axis("off")
        #plt.title("Object at "+str(int(z_pos[i]))+" mm")
    
    gt = np.empty((Nx,Ny)); gt.fill(1)
    #==60===
    #gt[24:27,21:24] = 2
    #gt[24:27,27:30] = 2
    #=================
    #==66===
    gt[21:24,24:27] = 2
    gt[27:30,24:27] = 2
    #=================
    
    #==Comparison range ==
    go = 12
    end = 39
    #go = 0
    #end = 51
    #=====================
    
    fig = plt.figure(dpi=1200)
    plt.pcolormesh(xx*1000,yy*1000,np.real(gt),shading='auto',cmap='jet')
    plt.xticks(xx[0,:]*1000, fontsize=12)
    plt.yticks(yy[:,0]*1000, fontsize=12)
    plt.clim(1,2)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)",**csfont,fontsize=16)
    plt.ylabel("y (mm)",**csfont,fontsize=16)
    #plt.title("Ground Truth")
    
    E = np.squeeze(np.abs(epsr[go:end,go:end,2]))
    print("CS Maximal calculated permittivity of object (real) = "+str(np.max(E)))
    print("CS PSNR = "+str(PSNR(gt[go:end,go:end],E,data_range=np.max(gt))))
    print("CS SSIM = "+str(SSIM(gt[go:end,go:end],E)))
 
toc()
plt.show()