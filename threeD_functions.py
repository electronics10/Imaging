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

from threeD_settings import*

###===========preprocess================
def get_ground_truth():
    # ##=========2D===============
    # ground = np.ones((Nx,Ny))
    # if target == "crossblock": # 2D Cross Block
    #     a = int(Nx/2-4/samp)
    #     b = int(Nx/2+6/samp)
    #     ground[a:b,Nx//2] = 2
    #     ground[Nx//2,a:b] = 2
    # elif target == "singleblock": # 2D Single Block
    #     a = int(Nx/2-2/samp)
    #     b = int(Nx/2+3/samp)
    #     ground[a:b, a:b] = 2
    # ##=========================
    ##==3D Cross Block (Nx, Ny = 51, 51)===
    ground = np.ones((Nx,Ny,len(zratio)))
    step = int(6/samp)
    a = int((Nx-9)/samp)
    b = a + step
    c = a + 2*step
    d = a + 3*step
    try:
        # 60mm
        ground[b:c,a:b,np.where(zratio==6)[0][0]] = 2
        ground[b:c,c:d,np.where(zratio==6)[0][0]] = 2
        #=================
        # 66mm
        ground[a:b,b:c,np.where(zratio==8)[0][0]] = 2
        ground[c:d,b:c,np.where(zratio==8)[0][0]] = 2
        ##===================
    except IndexError: pass
    return ground

def loadFEKO(folderpath):
    # load
    Iin = np.load(os.path.join(folderpath,"Iin.npy")) # Iin(x,y,f) #101*101*21
    g = np.load(os.path.join(folderpath,"g.npy")) # |Einc|^2=g(x,y,z,f) #101*101*13*21
    Voc = np.load(os.path.join(folderpath,"Voc.npy")) # Voc(x,y,f) #101*101*21
    # manipuate by different sampling distance
    Iin = Iin[::samp,::samp,:]
    g = g[::samp,::samp,:]
    Voc = Voc[::samp,::samp,:]
    return Iin, g, Voc

def voc_to_Tvec(Voc):
    Tmat = np.zeros((Nx, Ny, len(fratio)), dtype=np.complex128)
    Tvec = np.zeros((Nx*Ny, len(fratio)), dtype=np.complex128)
    for f_index in range(len(fratio)):
        f = fratio[f_index]
        Voc_f = Voc[:,:,f] # crop Voc(Nx*Ny*21) to Voc(Nx*Ny*1)
        Tmat_f = np.fft.fft2(Voc_f,axes=(0,1))
        Tmat_f = np.fft.fftshift(np.fft.fftshift(Tmat_f,0),1) 
        Tmat[:,:,f_index] = Tmat_f
        Tvec[:,f_index] = Tmat_f.ravel()
    return Tvec, Tmat

def g_to_Gmatrix(g, Iin):
    Gmat = np.zeros((Nx*Ny, Nx*Ny, len(zratio),len(fratio)), dtype=np.complex128)
    for f_index in range(len(fratio)):
        f = fratio[f_index]
        const = -1j*2*np.pi*Freq[f]*dz/Iin[:,:,f]
        for z_index in range(len(zratio)):
            z = zratio[z_index]
            g_z_f = g[:,:,z-5,f] #######!!!!!Four_blocks z = z-5 !!!!!
            Gmat_z_f = np.fft.fft2(g_z_f,axes=(0,1))
            Gmat_z_f = np.fft.fftshift(np.fft.fftshift(Gmat_z_f,0),1)
            Gprime_z_f = np.multiply(const,Gmat_z_f)
            idmat = np.eye(Nx*Ny)
            Gmat_z_f = idmat*np.ravel(Gprime_z_f, order='F')
            Gmat[:,:,z_index,f_index] = Gmat_z_f
    return Gmat
###==========================================

###===============(pseudo)inverse and regularization====================
def tsvd(Gmat):
    k = 1800 # heuristic
    for f_index in range(len(fratio)):
        for z_index in range(len(zratio)):
            u, s, vt = np.linalg.svd(Gmat[:,:,z_index,f_index], full_matrices=False)
            u[:,k:-1] = 0
            s[k:-1] = 0
            vt[k:-1,:] = 0
            Gmat[:,:,z_index,f_index] = u @ np.diag(s) @ vt
    return Gmat

def least_square_argument(Gmat, Tvec):
    angular_spectrum = np.zeros((Nx, Ny, len(zratio), len(fratio)))
    for z_index in range(len(zratio)):
        for f_index in range(len(fratio)):
            arg = np.linalg.lstsq(Gmat[:,:,z_index,f_index],Tvec[:,f_index],rcond=None)[0]
            angular_spectrum[:,:,z_index,f_index] = arg.reshape(Nx,Ny)
    return angular_spectrum

def twok_filter(angular_spectrum):
    kc = (2*np.pi)*fc/c
    for z_index in range(len(zratio)):
        theta = np.arctan(Aperture_size/2/(z_pos[z_index]*1e-3))
        limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
        for i in range(Ny):
            for j in range(Nx):
                if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                    angular_spectrum[i,j,z_index,:] = 0

def angspectrum_to_image(angular_spectrum):
    F = np.fft.ifftshift(np.fft.ifftshift(angular_spectrum,0),1)
    f_raw = np.fft.ifft2(F,axes = (0,1))
    f_raw = np.fft.fftshift(f_raw,axes = (0,1))
    epsr = (f_raw/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r
    return epsr
###======================================

###=============plot===============
def plot_ground_truth(ground):
    fig_list = []
    for z_index in range(len(zratio)):
        fig = plt.figure()
        plt.pcolormesh(xx[:,:]*1000,\
            yy[:,:]*1000,ground[:,:,z_index],shading='auto')
        plt.xticks(xx[0,:]*1000)
        plt.yticks(yy[:,0]*1000)
        plt.colorbar()
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=7)
        plt.xlabel("x Position (mm)")
        plt.ylabel("y Position (mm)")
        plt.title("Ground truth at" +str(int(z_pos[z_index]))+ "mm")
        fig_list.append(fig)
    return fig_list

def plot_angular_spectrum(angular_spectrum):
    fig_list = []
    for z_index in range(len(zratio)):
        fig_list_f = []
        for f_index in range(len((fratio))):
            fig = plt.figure()
            plt.pcolormesh(FXX[:,:],FYY[:,:],\
            np.abs(angular_spectrum[:,:,z_index,f_index]),shading='auto',cmap='jet')
            plt.xticks(2*np.pi*fxx[0,:])
            plt.yticks(2*np.pi*fyy[:,0])
            plt.clim(0,4e-15)
            #plt.clim(0,6e-14)
            plt.colorbar()
            plt.locator_params(axis='y', nbins=9)
            plt.locator_params(axis='x', nbins=9)
            plt.xlabel("kx (rad/m)")
            plt.ylabel("ky (rad/m)")
            f = fratio[f_index]
            plt.title("|F(kx,ky,z = "+str(int(z_pos[z_index]))+" mm)| of object at Freq = "+str(Freq[f])+"Hz")
            fig_list_f.append(fig)
        fig_list.append(fig_list_f)
    return fig_list

def plot_epsr(epsr):
    fig_list = []
    for z_index in range(len(zratio)):
        fig_list_f = []
        for f_index in range(len((fratio))):
            fig = plt.figure()
            plt.pcolormesh(xx[:,:]*1000,\
                yy[:,:]*1000,np.real(epsr[:,:,z_index,f_index]),\
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
            f = fratio[f_index]
            plt.title("Object at "+str(int(z_pos[z_index]))+" mm at Freq = "+str(Freq[f])+"Hz")
            fig_list_f.append(fig)
        fig_list.append(fig_list_f)
    return fig_list

def plot_Voc_spectrum(Tmat):
    fig_list = []
    for f_index in range(len(fratio)):
        fig = plt.figure()
        plt.pcolormesh(FXX[:,:],FYY[:,:],\
            np.abs(Tmat[:,:,f_index]),shading='auto')
        plt.xticks(2*np.pi*fxx[f_index,:])
        plt.yticks(2*np.pi*fyy[:,f_index])
        plt.colorbar()
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=9)
        f = fratio[f_index]
        plt.title("Spectrum of Voc (abs) at Freq = "+str(Freq[f]))
        fig_list.append(fig)
    return fig

def SNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    S = np.mean((img1)**2)
    # S = 1
    return 10 * math.log10(S/mse)

def print_stats(epsr,ground):
    for z_index in range(len(zratio)):
        ground_z = ground[:,:,z_index]
        for f_index in range(len(fratio)):
            temp = epsr[:,:,z_index,f_index]
            E = np.squeeze(np.abs(temp[:,:]))
            print("zpos = ", z_pos[z_index])
            print("Maximal calculated permittivity of object (real) = "+str(np.max(E)))
            print("PSNR = "+str(PSNR(ground_z[:,:],E,data_range=np.max(ground_z))))
            print("SSIM = "+str(SSIM(ground_z[:,:],E)))
            print("MSE = "+str(MSE(ground_z[:,:],E)))
            print("SNR = "+str(SNR(ground_z[:,:],E)))
            print("NRMSE = "+str(NRMSE(ground_z[:,:],E)))
            print("\n")

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
