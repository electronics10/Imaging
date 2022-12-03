import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
import math
import scipy.io

from pylops.basicoperators import *
from pylops.optimization.sparsity import SPGL1
from pylops.signalprocessing import FFT2D,DWT2D
import pylops
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE

from PIL import Image
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso,LassoCV,LassoLars,LassoLarsCV,LassoLarsIC,Ridge
from SSIM_PIL import compare_ssim
from time import time

csfont = {'fontname':'Times New Roman'}

def SNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    S = np.mean((img1)**2)
    #S = 1
    return 10 * math.log10(S/mse)

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
folderpath = "Crossblock2/numpy"
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
#Conjugate for consistent to evanescent wave
# Removed positions
pick = 0
# =================

if __name__ == "__main__":
    #mag_go = 14
    #mag_end = 35
    
    #mag_go = 0
    #mag_end = 51
    
    ####==== Random Undersample ====###
    comprate = 0.8
    N = Nx*Ny
    Nsub = int(np.round(N*comprate))
    iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])
    mask = np.zeros(N)
    mask[iava] = 1
    Maskop = Diagonal(mask.ravel())
    Rop = Restriction(N, iava, dtype=np.complex128)
    mask = mask.reshape((Ny,Nx))
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
    
    
    
    # fig5 = plt.figure()
    # plt.pcolormesh(mask, edgecolors='k', linewidth=1, cmap='Blues')
    # plt.axis('off')
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # #plt.colorbar(ticks=np.arange(np.min(mask),np.max(mask)+1))
    
    #FFT2op = DWT2D((Ny,Nx), wavelet="db20", level=4)
    FFT2op = FFT2D((Ny,Nx))
    
    A_mat = Rop * FFT2op.H  * np.identity(Nx*Ny)
    A_real = np.real(A_mat)
    A_imag = np.imag(A_mat)
    A_norm = np.linalg.norm(A_mat)
    #Tvec_holes = FFT2op * Voc.ravel()
    # test = FFT2op.todense()
    # test = np.exp(abs(np.real(test))*20)
    # fig5 = plt.figure()
    # plt.pcolormesh(test,cmap='bone')
    # plt.axis('off')

    Voc_holes =Maskop * Voc.ravel()
    
    #Voc_filledr, T_filledr, info = SPGL1(Maskop, Voc_holes.real,SOp=FFT2op , tau=0, iter_lim=200)
    #Voc_filledi, T_filledi, info = SPGL1(Maskop, Voc_holes.imag,SOp=FFT2op , tau=0, iter_lim=200)
    #Voc_filledr, T_filledr, info = SPGL1(Maskop, Voc_holes.real,SOp=DWT2op , tau=0, iter_lim=200)
    #Voc_filledi, T_filledi, info = SPGL1(Maskop, Voc_holes.imag,SOp=DWT2op , tau=0, iter_lim=200)
    Voc_filled, T_filled, info = SPGL1(Maskop, Voc_holes,SOp=FFT2op , tau=0, iter_lim=200, iscomplex=True)
    #T = T_filledr + 1j*T_filledi
    #Voc_filled = Voc_filledr+1j*Voc_filledi
    Voc_filled = Voc_filled.reshape(Ny,Nx)
    Voc_holes = Voc_holes.reshape(Ny,Nx)
    T = np.fft.fft2(Voc_filled,axes=(0,1))
    T = np.fft.fftshift(np.fft.fftshift(T,0),1)  
    Tvec = T.reshape((Nx*Ny*len(fratio),1))
    
    #T = T.reshape(Ny,Nx)
    #T = np.fft.fftshift(np.fft.fftshift(T,0),1)
    #Tvec = T.reshape((Nx*Ny*len(fratio),1),order='F')

    #g = g[mag_go:mag_end,mag_go:mag_end,zratio,fratio]
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
    
     # ============ Truncated SVD ====================
    # u, s, vt = linalg.svd(Gmat, full_matrices=False) #compute SVD without 0 singular values
    # fig = plt.figure(dpi=1200)
    # plt.plot(s)
    # plt.xlabel("Index i")
    # plt.ylabel("Amplitude of Singular Value")
    # #plt.title("Singular value of G' matrix")
    
    # k = 1000
    
    # u[:,k:-1] = 0;
    # s[k:-1] = 0;#10e-20;
    # vt[k:-1,:] = 0;
    
    # Gmatp = u @ np.diag(s) @ vt
    # X,res,rank,s = linalg.lstsq(Gmatp,Tvec)
    
    # # splus = np.linalg.inv(np.diag(s))
    # # v = np.transpose(vt)
    # # ut = np.transpose(u)
    # # X = v @ splus @ ut @ Tvec
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    
    # ======== Tikhonov regularization ==============
    
    # alph = 2e25
    # ridge_real = Ridge(alpha=alph)
    # ridge_imag = Ridge(alpha=alph)
    # ridge_real.fit(np.real(Gmat), np.real(Tvec))
    # ridge_imag.fit(np.imag(Gmat), np.imag(Tvec))
    
    # coef_real = ridge_real.coef_
    # coef_imag = ridge_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ====== Orthogonal Matching Pursuit ===========
    # n_nonzero_coefs = 80
    # omp_real = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    # omp_imag = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    # omp_real.fit(np.real(Gmat), np.real(Tvec))
    # coef_real = omp_real.coef_
    # idx_r, = coef_real.nonzero()
    
    # omp_imag.fit(np.imag(Gmat), np.imag(Tvec))
    # coef_imag = omp_imag.coef_
    # idx_i, = coef_imag.nonzero()
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ======= LASSO Regression ======================
    # alph = 1.5e7
    # lasso_real = Lasso(alpha=alph)
    # lasso_imag = Lasso(alpha=alph)
    # lasso_real.fit(np.real(Gmat), np.real(Tvec))
    # lasso_imag.fit(np.imag(Gmat), np.imag(Tvec))
    
    # coef_real = lasso_real.coef_
    # coef_imag = lasso_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ======= LASSO CV ======================
    # lasso_real = LassoCV(cv=5)
    # lasso_imag = LassoCV(cv=5)
    # lasso_real.fit(np.real(Gmat), np.real(Tvec).ravel())
    # lasso_imag.fit(np.imag(Gmat), np.imag(Tvec).ravel())
    
    # coef_real = lasso_real.coef_
    # coef_imag = lasso_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    
    # ============ Least square method ==============
    X,res,rank,s = linalg.lstsq(Gmat,Tvec)
    P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    count = np.ones((Ny,Nx))
    kc = (2*np.pi)*fc/(2.9979e8);
    for k in range(len(z_pos)):
        theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
        limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
        for i in range(Ny):
            for j in range(Nx):
                if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                    P[i,j,k] = 0
                    count[i,j] = 0
    # ===============================================
    
    
    # # ======== Position wise Least square ===========
    # A = np.transpose(Gprime,(2,0,1))
    # A = A[np.newaxis,:]
    # A = np.concatenate((A,np.conj(A)),axis = 0)
    # #B = np.transpose(T,(2,0,1))
    # B = T[np.newaxis,:]
    # B = np.concatenate((B,np.conj(B)),axis = 0)
    
    # P = []
    # for i in range(Ny):
    #     for j in range(Nx):
    #         x,res,rank,s = linalg.lstsq(A[:,:,i,j],B[:,i,j])
    #         P.append(x)
    # P = np.array(P)
    # P = P.reshape((Ny,Nx,len(zratio)))
    
    # kc = (2*np.pi)*fc/c;
    # for k in range(len(z_pos)):
    #     theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
    #     limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
    #     for i in range(Ny):
    #         for j in range(Nx):
    #             if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
    #                 P[i,j,k] = 0
    # # ===============================================
    
    gt = np.empty((Nx,Ny)); gt.fill(1)
    #==Single Block===
    #gt[24:27,24:27] = 2
    #=================
    #==Cross Block====
    gt[23:28,25] = 2
    gt[25,23:28] = 2
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
    #plt.clim(1,np.max(gt))
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)",**csfont,fontsize=16)
    plt.ylabel("y (mm)",**csfont,fontsize=16)
    #plt.title("Ground Truth")
    
    Tvec[int(np.size(Tvec)/2)]=0
    P[int(Ny/2),int(Nx/2),:]=0
    T[int(Ny/2),int(Nx/2)]=0
    F = np.fft.ifftshift(np.fft.ifftshift(P,0),1)
    f_raw = np.fft.ifft2(F,axes = (0,1))
    f_raw = np.fft.fftshift(f_raw,axes = (0,1))
    epsr = (f_raw/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r 
    
    # for i in range(len(z_pos)):
    #     fig = plt.figure(dpi=1200)
    #     #plt.pcolormesh(FXX[mag_go:mag_end,mag_go:mag_end],FYY[mag_go:mag_end,mag_go:mag_end],np.abs(P[:,:,i]),shading='auto') #[42:61,42:61]
    #     #plt.xticks(2*np.pi*fxx[0,mag_go:mag_end])
    #     #plt.yticks(2*np.pi*fyy[mag_go:mag_end,0])
    #     plt.pcolormesh(FXX,FYY,np.abs(P[:,:,i]),shading='auto',cmap='jet') #[42:61,42:61]
    #     plt.xticks(2*np.pi*fxx[0,:])
    #     plt.yticks(2*np.pi*fyy[:,0])
    #     #plt.clim(0,5e-15)
    #     #plt.clim(0,6e-14)
    #     plt.colorbar()
    #     plt.locator_params(axis='y', nbins=7)
    #     plt.locator_params(axis='x', nbins=7)
    #     plt.xlabel('kx Position (1/m)',**csfont)
    #     plt.ylabel("ky Position (1/m)",**csfont)
        #plt.title("|F(kx,ky,z = "+str(int(z_pos[i]))+" mm)| of object")
            
        #Visualize object matrix
    for i in range(len(z_pos)):
        fig = plt.figure(dpi=1200)
        #plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.real(epsr[:,:,i]),shading='auto')
        #plt.xticks(xx[0,mag_go:mag_end]*1000)
        #plt.yticks(yy[mag_go:mag_end,0]*1000)
        plt.pcolormesh(xx*1000,yy*1000,np.abs(epsr[:,:,i]),shading='auto',cmap='jet')
        plt.xticks(xx[0,:]*1000, fontsize=12)
        plt.yticks(yy[:,0]*1000, fontsize=12)
        plt.clim(1,2)
        #plt.clim(1,np.max(abs(epsr)))
        #plt.clim(1,1.45)
        #Since we should have epsilon larger than 1
        #plt.colorbar()
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
        plt.xlabel("x (mm)",**csfont,fontsize=16)
        plt.ylabel("y (mm)",**csfont,fontsize=16)
        plt.axis('scaled')
        plt.axis("off")
        #plt.title("Object at "+str(int(z_pos[i]))+" mm")
        #plt.axis('scaled')
        
    # fig = plt.figure()
    # plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end],yy[mag_go:mag_end,mag_go:mag_end],gt,shading='auto')
    # plt.xticks(xx[0,mag_go:mag_end])
    # plt.yticks(yy[mag_go:mag_end,0])
    # plt.colorbar()
    # plt.locator_params(axis='y', nbins=7)
    # plt.locator_params(axis='x', nbins=7)
    # plt.xlabel("x Position (m)")
    # plt.ylabel("y Position (m)")
    # plt.title("Ground truth at 63 mm")
    
    fig2 = plt.figure()
    fig = plt.figure()
    plt.plot(np.abs(Tvec))
    plt.title("Abs value of T vector")
    
    fig3 = plt.figure()
    fig = plt.figure()
    plt.plot(np.imag(Tvec))
    plt.title("Imag value of T vector")
    
    # Xmax_sq = 4
    # avg = 1/(Nx*Ny)*np.sum((abs(epsr)-gt)**2)
    # psnr = 10*np.log10(Xmax_sq/avg)
    # print("The PSNR between image and Ground truth "+str(psnr)+" dB")
    
    # norm_epsr = abs(epsr[:,:,-1])*255/np.max(abs(epsr[:,:,-1]))
    # norm_epsr = norm_epsr.astype(np.uint8)
    # img1 = Image.fromarray(norm_epsr)
    # gti = (gt*255/2).astype(np.uint8)
    # img2 = Image.fromarray(gti)
    # ssim = compare_ssim(img1,img2)
    # print("The SSIM between image and Ground truth "+str(ssim))
    E = np.squeeze(np.abs(epsr[go:end,go:end]))
    print("CS Maximal calculated permittivity of object (real) = "+str(np.max(E)))
    print("CS PSNR = "+str(PSNR(gt[go:end,go:end],E,data_range=np.max(gt))))
    print("CS SSIM = "+str(SSIM(gt[go:end,go:end],E)))
    print("CS MSE = "+str(MSE(gt[go:end,go:end],E)))
    print("CS SNR = "+str(SNR(gt[go:end,go:end],E)))
    print("CS NRMSE = "+str(NRMSE(gt[go:end,go:end],E)))
    # fig4 = plt.figure()
    # plt.pcolormesh(FXX[mag_go:mag_end,mag_go:mag_end],FYY[mag_go:mag_end,mag_go:mag_end],np.abs(T),shading='auto') #[42:61,42:61]
    # plt.xticks(2*np.pi*fxx[0,mag_go:mag_end])
    # plt.yticks(2*np.pi*fyy[mag_go:mag_end,0])
    # plt.locator_params(axis='y', nbins=9)
    # plt.locator_params(axis='x', nbins=9)
    # plt.title("Spectrum of Voc (abs)")
    # plt.colorbar()
    
    T2 = np.fft.fft2(Voc_holes,axes=(0,1))
    T2 = np.fft.fftshift(np.fft.fftshift(T2,0),1)  
    Tvec2 = T2.reshape((Nx*Ny*len(fratio),1))
    Tvec2[int(Ny*Nx/2)]=0
    # ============ Least square method ==============
    X,res,rank,s = linalg.lstsq(Gmat,Tvec2)
    P2 = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    
    kc = (2*np.pi)*fc/(2.9979e8);
    for k in range(len(z_pos)):
        theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
        #limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
        limit = 2*kc*np.sin(theta)/((2*np.pi)/(Nx*dx))
        #limit = 2*kc*np.sin(theta)/((2*np.pi*(Nx-1))/(2*Nx*dx))
        for i in range(Ny):
            for j in range(Nx):
                #if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                if (i-(Ny-1)/2)**2+(j-(Nx-1)/2)**2>limit**2:
                    P2[i,j,k] = 0
    # ===============================================
    P2[int(Ny/2),int(Nx/2),:]=0
    F2 = np.fft.ifftshift(np.fft.ifftshift(P2,0),1)
    f_raw2 = np.fft.ifft2(F2,axes = (0,1))
    f_raw2 = np.fft.fftshift(f_raw2,axes = (0,1))
    epsr2 = (f_raw2/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r 
    
    for i in range(len(z_pos)):
        fig = plt.figure(dpi=1200)
        #plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.real(epsr[:,:,i]),shading='auto')
        #plt.xticks(xx[0,mag_go:mag_end]*1000)
        #plt.yticks(yy[mag_go:mag_end,0]*1000)
        plt.pcolormesh(xx*1000,yy*1000,np.abs(epsr2[:,:,i]),shading='auto',cmap='jet')
        plt.xticks(xx[0,:]*1000, fontsize=12)
        plt.yticks(yy[:,0]*1000, fontsize=12)
        plt.clim(1,2)
        #plt.clim(1,np.max(abs(epsr)))
        #plt.clim(1,1.45)
        #Since we should have epsilon larger than 1
        #plt.colorbar()
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
        plt.xlabel("x (mm)",**csfont,fontsize=16)
        plt.ylabel("y (mm)",**csfont,fontsize=16)
        #plt.title("Object at "+str(int(z_pos[i]))+" mm")
        plt.axis('scaled')
        plt.axis("off")
    E2 = np.squeeze(np.abs(epsr2[go:end,go:end]))
    print("NO-CS Maximal calculated permittivity of object (real) = "+str(np.max(E2)))
    print("NO-CS PSNR = "+str(PSNR(gt[go:end,go:end],E2,data_range=np.max(gt))))
    print("NO-CS SSIM = "+str(SSIM(gt[go:end,go:end],E2)))
    print("NO-CS MSE = "+str(MSE(gt[go:end,go:end],E2)))
    print("NO-CS SNR = "+str(SNR(gt[go:end,go:end],E2)))
    print("NO-CS NRMSE = "+str(NRMSE(gt[go:end,go:end],E2)))
    toc()
    plt.show()
    
    
    
    