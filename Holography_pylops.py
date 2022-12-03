import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
from scipy.sparse.linalg import svds
#import cv2
#import itertools

from pylops.basicoperators import *
from pylops.optimization.sparsity import SPGL1
from pylops.signalprocessing import FFT2D,DWT2D
import pylops

from PIL import Image
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso,LassoCV,LassoLars,LassoLarsCV,LassoLarsIC
from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,ElasticNetCV
from SSIM_PIL import compare_ssim
from time import time

csfont = {'fontname':'Times New Roman'}
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
#folderpath = "D:/Desktop/Imaging/Python/Bendata"
#folderpath = "D:/Desktop/Imaging/PSF/npy"
Iin = np.load(os.path.join(folderpath,"Iin.npy"))
g = np.load(os.path.join(folderpath,"g.npy"))
Voc = np.load(os.path.join(folderpath,"Voc.npy"))
#Voc = np.load("D:/Desktop/Imaging/RRBF/Voccrossblock_interp_factor4_RRBF.npy")
# ======================================================
#psfpath = "D:/Desktop/Imaging/PSF/npy"
#g = np.load(os.path.join(psfpath,"kernel.npy"))
#g = g.transpose(1,0,2,3)
#G = np.load(os.path.join(psfpath,"transfer.npy"))
#Voc = np.load(os.path.join(psfpath,"PSF.npy"))

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
    #mag_go = 32
    #mag_end = 74
    #mag_go = 12
    #mag_end = 63
    
    
    mag_go = 0
    mag_end = 101
    #Voc = Voc[:,:,10]
    #Voc = Voc[::2,::2,10]
    Voc = Voc[mag_go:mag_end,mag_go:mag_end,10]
    # Voc = Voc[::4,::4]
    # Voc = np.repeat(Voc,4,axis=0)
    # Voc = np.repeat(Voc,4,axis=1)
    Ny,Nx = Voc.shape
    
    
    ####==== Random Undersample ====###
    # comprate = 0.999
    # N = Nx*Ny
    # Nsub = int(np.round(N*comprate))
    # iava = np.sort(np.random.permutation(np.arange(N))[:Nsub])
    # mask = np.zeros(N)
    # mask[iava] = 1
    # Maskop = Diagonal(mask.ravel())
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####============================###
    
    ####==== Random Line Undersample ====###
    # comprate = 0.5
    # N = Nx*Ny
    # Nsub = int(np.round(Nx*comprate))
    # iava = np.sort(np.random.permutation(np.arange(Nx))[:Nsub])
    # mask = np.zeros((Nx,Ny))
    # mask[iava,:] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==0)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####============================###
    
    ####==== Spiral Undersample ====###
    # b = 0.75
    # N = Nx*Ny
    # mask = np.zeros((Nx,Ny))
    # theta = np.radians(np.linspace(0,360*10.9,2000))
    # r = b*theta
    # x_traj = np.round(r*np.cos(theta)+Nx/2).astype(int)
    # y_traj = np.round(r*np.sin(theta)+Ny/2).astype(int)
    # mask[x_traj,y_traj] = 1
    # #mask[:,y_traj] = 1
    # masker = mask.ravel()
    # iava = np.where(masker==1)[0]
    # Maskop = Diagonal(masker)
    # Rop = Restriction(N, iava, dtype=np.complex128)
    ####============================###
    
    ####==== Squared-spiral Undersample ====###
    # repeates = 5
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
    N = Nx*Ny
    mask = np.zeros((Nx,Ny))
    mask[0::1,0::1] = 1
    #mask[1::3,:] = 1
    masker = mask.ravel()
    iava = np.where(masker==1)[0]
    Maskop = Diagonal(masker)
    Rop = Restriction(N, iava, dtype=np.complex128)
   
    ####=============================###
    
    
    #DWT2op = DWT2D((Ny,Nx), wavelet="db20", level=4)
    FFT2op = FFT2D((Ny,Nx))
    #Tvec_holes = FFT2op * Voc.ravel()
    Voc_holes = Rop * Voc.ravel()
    Voc_holes_visual =Maskop * Voc.ravel()

    Voc_filled, T_filled, info = SPGL1(Rop, Voc_holes, SOp=FFT2op.H , tau=0, iter_lim=200, iscomplex=True)

    A = Rop * FFT2op.H  * np.identity(Nx*Ny)
    A_real = np.real(A)
    A_imag = np.imag(A)
    A_norm = np.linalg.norm(A)
    Voc_filled = Voc_filled.reshape(Ny,Nx)
    Voc_holes_visual = Voc_holes_visual.reshape(Ny,Nx)
    #T = np.fft.fft2(Voc_holes_visual,axes=(0,1))
    T = np.fft.fft2(Voc_filled,axes=(0,1))
    T = np.fft.fftshift(np.fft.fftshift(T,0),1)  
    Tvec = T.reshape((Nx*Ny*len(fratio),1))
    
    # T_filled = T_filled.reshape(Ny,Nx)
    # T_filled = np.fft.fftshift(np.fft.fftshift(T_filled,0),1)  
    # Tvec = T_filled.reshape((Nx*Ny*len(fratio),1))
    
    #T = T.reshape(Ny,Nx)
    #T = np.fft.fftshift(np.fft.fftshift(T,0),1)
    #Tvec = T.reshape((Nx*Ny*len(fratio),1),order='F')

    g = g[mag_go:mag_end,mag_go:mag_end,zratio,fratio]
    
    #g = g[::4,::4,:]
    #g = np.repeat(g,4,axis=0)
    #g = np.repeat(g,4,axis=1)
    #g[1::2,:,0] = 0
    #g[:,1::2,0] = 0
    G = np.fft.fft2(g,axes=(0,1))
    G = np.fft.fftshift(np.fft.fftshift(G,0),1)
    print(G)
    
    #G = G[mag_go:mag_end,mag_go:mag_end,zratio,fratio]    
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
    
    #  # ============ Truncated SVD ====================
    # u, s, vt = np.linalg.svd(Gmat, full_matrices=False)
    # fig = plt.figure()
    # fig = plt.figure(dpi=1200)
    # plt.plot(s)
    # plt.xlabel("Index i",**csfont,fontsize=16)
    # plt.ylabel("Amplitude of Singular Values (a.u.)",**csfont,fontsize=16)
    # #plt.title("Singular value of G' matrix")
    
    # k = 1800
    
    # u[:,k:-1] = 0;
    # s[k:-1] = 0;
    # vt[k:-1,:] = 0;
    
    # Gmatp = u @ np.diag(s) @ vt
    # X,res,rank,s = linalg.lstsq(Gmatp,Tvec)
    
    # # ##splus = np.linalg.inv(np.diag(s))
    # # #splus = 1.0/np.diag(s)
    # # #splus[np.isinf(splus)] = 0
    # # #v = np.transpose(vt)
    # # #ut = np.transpose(u)
    # # #XX = v @ splus @ ut @ Tvec
    # # #P = XX.reshape(-1,Nx,Ny).transpose(1,2,0)
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    

    # ===============================================
    # # =========== Randomized Truncated SVD ==========
    # k = 100
    # r = 400
    # Omega = np.random.randn(Gmat.shape[1],r)
    # u_r, s_r, vt_r = rsvd(np.real(Gmat), Omega)
    # u_i, s_i, vt_i = rsvd(np.imag(Gmat), Omega)
    # # u_r[:,k:-1] = 0
    # # u_i[:,k:-1] = 0
    # # s_r[k:-1] = 0
    # # s_i[k:-1] = 0
    # # vt_r[k:-1,:] = 0
    # # vt_i[k:-1,:] = 0
    # Gmat_r = u_r @ np.diag(s_r) @ vt_r
    # Gmat_i = u_i @ np.diag(s_i) @ vt_i
    # Gmatp = Gmat_r+1j*Gmat_i
    # Gmatp = np.diag(Gmatp)
    # Gmatp = np.diag(Gmatp)
    # #test1 = abs(np.diag(Gmat))
    # #test2 = abs(np.diag(Gmatp))
    # #plt.plot((test1-test2)*100/test1)
    # X,res,rank,s = linalg.lstsq(Gmatp,Tvec)
    
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # # ===============================================
    # ============ Selective SVD ====================
    # u, s, vt = np.linalg.svd(Gmat, full_matrices=False) #compute SVD without 0 singular values
    # proj = proj = np.dot(np.transpose(u),Tvec)
    
    # plt.plot(abs(proj))
    # plt.xlabel("index i")
    # plt.ylabel("Amplitude")
    # plt.title("Projection of T to ui (abs)")
    
    # #thrld =0.07
    # thrld = 0.07
    # kept = Nx*Ny
    # for i in range(len(proj)):
    #     if abs(proj[i])<thrld:
    #         u[:,i] = 0;
    #         s[i] = 10e-20;
    #         vt[i,:] = 0;
    #         kept = kept-1
    # ratio = kept/(Nx*Ny)*100
    # print("The ratio of kept columns is: "+str(ratio)+"%")
    # Gmatp = u @ np.diag(s) @ vt
    # X,res,rank,s = linalg.lstsq(Gmatp,Tvec)
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # =========Randomized Selective SVD =============
    # r = 300
    # Omega = np.random.randn(Gmat.shape[1],r)
    # u_r, s_r, vt_r = rsvd(np.real(Gmat), Omega)
    # u_i, s_i, vt_i = rsvd(np.imag(Gmat), Omega)
    # proj = np.dot(np.transpose(u_r+1j*u_i),Tvec)
    # plt.plot(abs(proj))
    # plt.xlabel("index i")
    # plt.ylabel("Amplitude")
    # plt.title("Projection of T to ui (abs)")
    
    # #thrld =0.02
    # thrld = 0.1
    # kept = Nx*Ny
    # for i in range(len(proj)):
    #     if abs(proj[i])<thrld:
    #         u_r[:,i] = 0
    #         u_i[:,i] = 0
    #         s_r[i] = 0
    #         s_i[i] = 0
    #         vt_r[i,:] = 0
    #         vt_i[i,:] = 0
    #         kept = kept-1
    # ratio = kept/(Nx*Ny)*100
    # print("The ratio of kept columns is: "+str(ratio)+"%")
    # Gmat_r = u_r @ np.diag(s_r) @ vt_r
    # Gmat_i = u_i @ np.diag(s_i) @ vt_i
    # Gmatp = Gmat_r+1j*Gmat_i
    # X,res,rank,s = linalg.lstsq(Gmatp,Tvec)
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    
    # ======== Tikhonov regularization ==============
    # alph = 0
    # ridge_real = Ridge(alpha=alph,solver='sparse_cg')
    # ridge_imag = Ridge(alpha=alph,solver='sparse_cg')
    # ridge_real.fit(np.real(Gmat), np.real(Tvec))
    # ridge_imag.fit(np.imag(Gmat), np.imag(Tvec))
    
    # coef_real = ridge_real.coef_
    # coef_imag = ridge_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ======== Tikhonov CV ==============
    # ridge_real = RidgeCV(cv=4)
    # ridge_imag = RidgeCV(cv=4)
    # ridge_real.fit(np.real(Gmat), np.real(Tvec).ravel())
    # ridge_imag.fit(np.imag(Gmat), np.imag(Tvec).ravel())
    
    # coef_real = ridge_real.coef_
    # coef_imag = ridge_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ====== Orthogonal Matching Pursuit ===========
    # n_nonzero_coefs = 120
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
    # alph = 3e7
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
    # #X = coef_abs*np.exp(1j*coef_angle)
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ======= Elastic Net ======================
    # alph = 10e7
    # net_real = ElasticNet(alpha=alph)
    # net_imag = ElasticNet(alpha=alph)
    # net_real.fit(np.real(Gmat), np.real(Tvec))
    # net_imag.fit(np.imag(Gmat), np.imag(Tvec))
    
    # coef_real = net_real.coef_
    # coef_imag = net_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ======= ElasticNet CV ======================
    # net_real = ElasticNetCV(cv=5)
    # net_imag = ElasticNetCV(cv=5)
    # net_real.fit(np.real(Gmat), np.real(Tvec).ravel())
    # net_imag.fit(np.imag(Gmat), np.imag(Tvec).ravel())
    
    # coef_real = net_real.coef_
    # coef_imag = net_imag.coef_
    
    # X = coef_real+1j*coef_imag
    # P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    # ===============================================
    # ============ Least square method ==============
    X,res,rank,s = linalg.lstsq(Gmat,Tvec)
    P = X.reshape(-1,Nx,Ny).transpose(1,2,0)
    
    # kc = (2*np.pi)*fc/(2.9979e8);
    # for k in range(len(z_pos)):
    #     theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
    #     limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
    #     for i in range(Ny):
    #         for j in range(Nx):
    #             if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
    #                 P[i,j,k] = 0
    # ===============================================
    
    # ======== Position wise Least square ===========
    # A = np.transpose(Gprime,(2,0,1))
    # A = A[np.newaxis,:]
    # #A = np.concatenate((A,np.conj(A)),axis = 0)
    # #B = np.transpose(T,(2,0,1))
    # B = T[np.newaxis,:]
    # #B = np.concatenate((B,np.conj(B)),axis = 0)
    
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
    # ===============================================
    
    #Tvec[int(np.size(Tvec)/2)]=0
    gt = np.empty((101,101)); gt.fill(1)
    gt[48:53,48:53] = 2
    gt = gt[mag_go:mag_end,mag_go:mag_end]
    #T[int(Ny/2),int(Nx/2)]=0
    P[int(Ny/2),int(Nx/2),:]=0
    F = np.fft.ifftshift(np.fft.ifftshift(P,0),1)
    f_raw = np.fft.ifft2(F,axes = (0,1))
    f_raw = np.fft.fftshift(f_raw,axes = (0,1))
    epsr = (f_raw/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r 
    for i in range(len(z_pos)):
        fig = plt.figure(dpi=1200)
        plt.pcolormesh(FXX[mag_go:mag_end,mag_go:mag_end],FYY[mag_go:mag_end,mag_go:mag_end],np.abs(P[:,:,i]),shading='auto',cmap='jet') #[42:61,42:61]
        plt.xticks(2*np.pi*fxx[0,mag_go:mag_end], fontsize=13)
        plt.yticks(2*np.pi*fyy[mag_go:mag_end,0], fontsize=13)
        plt.clim(0,4e-15)
        #plt.clim(0,6e-14)
        plt.colorbar()
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=9)
        plt.xlabel("kx (rad/m)",**csfont,fontsize=16)
        plt.ylabel("ky (rad/m)",**csfont,fontsize=16)
        #plt.title("|F(kx,ky,z = "+str(int(z_pos[i]))+" mm)| of object")
            
        #Visualize object matrix
    for i in range(len(z_pos)):
        fig = plt.figure(dpi=1200)
        plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,np.real(epsr[:,:,i]),shading='auto',cmap='jet')
        plt.xticks(xx[0,mag_go:mag_end]*1000, fontsize=13)
        plt.yticks(yy[mag_go:mag_end,0]*1000, fontsize=13)
        #plt.clim(1,2)
        plt.clim(1,np.max(np.real(epsr)))
        #Since we should have epsilon larger than 1
        plt.colorbar()
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
        plt.xlabel("x (mm)",**csfont,fontsize=16)
        plt.ylabel("y (mm)",**csfont,fontsize=16)
        #plt.title("Object at "+str(int(z_pos[i]))+" mm")
        
    fig = plt.figure()
    plt.pcolormesh(xx[mag_go:mag_end,mag_go:mag_end]*1000,yy[mag_go:mag_end,mag_go:mag_end]*1000,gt,shading='auto')
    plt.xticks(xx[0,mag_go:mag_end]*1000)
    plt.yticks(yy[mag_go:mag_end,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Ground truth at 63 mm")
    
    fig2 = plt.figure()
    fig = plt.figure()
    plt.plot(np.abs(Tvec))
    plt.title("Abs. value of the reshaped spectrum of Voc")
    
    fig3 = plt.figure()
    fig = plt.figure()
    plt.plot(np.imag(Tvec))
    plt.title("Imag value of T vector")
    
    # Xmax_sq = 4
    # avg = 1/(Nx*Ny)*np.sum((abs(epsr)-gt)**2)
    # psnr = 10*np.log10(Xmax_sq/avg)
    # print("The PSNR between image and Ground truth "+str(psnr)+" dB")
    
    norm_epsr = abs(epsr[:,:,-1])*255/np.max(abs(epsr[:,:,-1]))
    norm_epsr = norm_epsr.astype(np.uint8)
    img1 = Image.fromarray(norm_epsr)
    gti = (gt*255/2).astype(np.uint8)
    img2 = Image.fromarray(gti)
    ssim = compare_ssim(img1,img2)
    print("The SSIM between image and Ground truth "+str(ssim))
    print("Maximal calculated permittivity of object (real) = "+str(np.max(np.real(epsr))))
    
    fig4 = plt.figure()
    fig = plt.figure()
    plt.pcolormesh(FXX[mag_go:mag_end,mag_go:mag_end],FYY[mag_go:mag_end,mag_go:mag_end],np.abs(T),shading='auto') #[42:61,42:61]
    plt.xticks(2*np.pi*fxx[0,mag_go:mag_end])
    plt.yticks(2*np.pi*fyy[mag_go:mag_end,0])
    plt.locator_params(axis='y', nbins=9)
    plt.locator_params(axis='x', nbins=9)
    plt.title("Spectrum of Voc (abs)")
    plt.colorbar()
    plt.show()
    toc()