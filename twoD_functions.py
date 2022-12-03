import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pylops
from pylops.optimization.sparsity import SPGL1

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE

from twoD_settings import*

###===========preprocess================
def get_ground_truth(target):
    target = str(target).lower()
    ground = np.ones((Nx,Ny))
    if target == "crossblock": # 2D Cross Block
        a = int(Nx/2-4/samp)
        b = int(Nx/2+6/samp)
        ground[a:b,Nx//2] = 2
        ground[Nx//2,a:b] = 2
    elif target == "singleblock": # 2D Single Block
        a = int(Nx/2-2/samp)
        b = int(Nx/2+3/samp)
        ground[a:b, a:b] = 2
    # ##==3D Cross Block (Nx, Ny = 51, 51)===
    # # 60mm
    # ground[24:27,21:24] = 2
    # ground[24:27,27:30] = 2
    # #=================
    # # 66mm
    # ground[21:24,24:27] = 2
    # ground[27:30,24:27] = 2
    # ##===================
    return ground

def loadFEKO(target):
    target = str(target).lower()
    if target == "crossblock": folderpath = "Crossblock3/numpy"
    elif target == "singleblock": folderpath = "Singleblock/numpy"
    # load
    Iin = np.load(os.path.join(folderpath,"Iin.npy")) # Iin(x,y,f) #101*101*21
    g = np.load(os.path.join(folderpath,"g.npy")) # |Einc|^2=g(x,y,z,f) #101*101*13*21
    Voc = np.load(os.path.join(folderpath,"Voc.npy")) # Voc(x,y,f) #101*101*21
    # manipuate by different sampling distance
    Iin = Iin[::samp,::samp,:]
    g = g[::samp,::samp,:]
    Voc = Voc[::samp,::samp,:]
    return Iin, g, Voc

def voc_to_Tvec(Voc, fratio):
    Voc = Voc[:,:,fratio] # crop Voc(Nx*Ny*21) to Voc(Nx*Ny*1)
    Tmat = np.fft.fft2(Voc,axes=(0,1))
    Tmat = np.fft.fftshift(np.fft.fftshift(Tmat,0),1) 
    Tvec = Tmat.ravel()
    return Tvec, Tmat

def g_to_Gmatrix(g, Iin):
    g = g[:,:,zratio,fratio] # g(Nx*Ny*13*21) to g(Nx*Ny*1*1)
    Gmat = np.fft.fft2(g,axes=(0,1))
    Gmat = np.fft.fftshift(np.fft.fftshift(Gmat,0),1)
    const = -1j*2*np.pi*Freq[fratio]*dz/Iin[:,:,fratio] # not important
    Gprime = np.multiply(const,Gmat)
    idmat = np.eye(Nx*Ny)
    Gmat = idmat*np.ravel(Gprime, order='F')
    return Gmat
###==========================================

###===============(pseudo)inverse and regularization====================
def tsvd(Gmat):
    u, s, vt = np.linalg.svd(Gmat, full_matrices=False)
    k = 1800
    u[:,k:-1] = 0
    s[k:-1] = 0
    vt[k:-1,:] = 0
    Gmat = u @ np.diag(s) @ vt
    return Gmat

def least_square_argument(Gmat, Tvec):
    arg = np.linalg.lstsq(Gmat,Tvec,rcond=None)[0]
    angular_spectrum = arg.reshape(-1,Nx,Ny).transpose(1,2,0)
    return angular_spectrum

def twok_filter(angular_spectrum):
    kc = (2*np.pi)*fc/c
    for k in range(len(z_pos)):
        theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
        limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
        for i in range(Ny):
            for j in range(Nx):
                if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                    angular_spectrum[i,j,k] = 0

def angspectrum_to_image(angular_spectrum):
    F = np.fft.ifftshift(np.fft.ifftshift(angular_spectrum,0),1)
    f_raw = np.fft.ifft2(F,axes = (0,1))
    f_raw = np.fft.fftshift(f_raw,axes = (0,1))
    epsr = (f_raw/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r
    return epsr
###======================================

###=============plot===============
def plot_ground_truth(ground):
    fig = plt.figure()
    plt.pcolormesh(xx[:,:]*1000,\
        yy[:,:]*1000,ground,shading='auto')
    plt.xticks(xx[0,:]*1000)
    plt.yticks(yy[:,0]*1000)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x Position (mm)")
    plt.ylabel("y Position (mm)")
    plt.title("Ground truth at 63 mm")
    return fig

def plot_angular_spectrum(angular_spectrum):
    fig = plt.figure()
    plt.pcolormesh(FXX[:,:],FYY[:,:],\
    np.abs(angular_spectrum[:,:,0]),shading='auto',cmap='jet')
    plt.xticks(2*np.pi*fxx[0,:])
    plt.yticks(2*np.pi*fyy[:,0])
    plt.clim(0,4e-15)
    #plt.clim(0,6e-14)
    plt.colorbar()
    plt.locator_params(axis='y', nbins=9)
    plt.locator_params(axis='x', nbins=9)
    plt.xlabel("kx (rad/m)")
    plt.ylabel("ky (rad/m)")
    plt.title("|F(kx,ky,z = "+str(int(z_pos[0]))+" mm)| of object")
    return fig

def plot_epsr(epsr):
    fig = plt.figure()
    plt.pcolormesh(xx[:,:]*1000,\
        yy[:,:]*1000,np.real(epsr[:,:,0]),\
            shading='auto',cmap='jet')
    plt.xticks(xx[0,:]*1000)
    plt.yticks(yy[:,0]*1000)
    #plt.clim(1,2)
    plt.clim(1,np.max(np.real(epsr)))
    #Since we should have epsilon larger than 1
    plt.colorbar()
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Object at "+str(int(z_pos[0]))+" mm")
    return fig

def plot_Voc_spectrum(Tmat):
    fig = plt.figure()
    plt.pcolormesh(FXX[:,:],FYY[:,:],\
        np.abs(Tmat[:,:,0]),shading='auto')
    plt.xticks(2*np.pi*fxx[0,:])
    plt.yticks(2*np.pi*fyy[:,0])
    plt.colorbar()
    plt.locator_params(axis='y', nbins=9)
    plt.locator_params(axis='x', nbins=9)
    plt.title("Spectrum of Voc (abs)")
    return fig

def SNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    S = np.mean((img1)**2)
    #S = 1
    return 10 * math.log10(S/mse)

def print_stats(epsr,ground):
    E = np.squeeze(np.abs(epsr[:,:]))
    print("Maximal permittivity = "+str(np.max(E)))
    print("PSNR = "+str(PSNR(ground[:,:],E,data_range=np.max(ground))))
    print("SSIM = "+str(SSIM(ground[:,:],E)))
    print("MSE = "+str(MSE(ground[:,:],E)))
    print("SNR = "+str(SNR(ground[:,:],E)))
    print("NRMSE = "+str(NRMSE(ground[:,:],E)))

def show_plot():
    plt.show()
###==================================

###========undersampling=============
def undersampling(Voc, comprate=1):
    print("compressed rate = ",comprate)
    mask = np.random.choice(2, Nx*Ny, p=[1-comprate, comprate])
    mask = np.reshape(mask, (Ny, Nx))
    Voc_holed = Voc.transpose(2,0,1)*mask
    Voc_holed = Voc_holed.transpose(1,2,0)
    return Voc_holed, mask

def compressed_sensing(Voc_holed, mask):
    Maskop = pylops.basicoperators.Diagonal(np.ravel(mask))
    FFT2op = pylops.signalprocessing.FFT2D((Ny,Nx))
    for f in fratio:
        holed_freq = Maskop * np.ravel(Voc_holed[:,:,f])
        filled_freq = SPGL1(Maskop, holed_freq, SOp=FFT2op,\
            tau=0, iter_lim=200, iscomplex=True)[0]
        Voc_holed[:,:,f] = np.reshape(filled_freq, (Ny,Nx))
    Voc_filled = Voc_holed
    return Voc_filled

