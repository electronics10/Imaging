import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
#from readFEKO import *
    
#Basic file settings
# =======================================================
epsz = 8.854187817*1e-12      # F/m
muz = 4*np.pi*1e-7               # H/m              
c = (epsz*muz)**(-1/2)         # m/s
eta = np.sqrt(muz/epsz)          # ohm

Nx = 101
Ny = 101
Nz = 7

dx = 7.2e-3 #meters
dy = 7.2e-3 #meters
dz = 3e-3 #meters

Fint =  8.5e9         # Hz
Fend =  11.5e9       # Hz
Fstep = 0.5e9        # Hz

Aperture_size = 180e-3 #Determine the desired data range for recontruction
Object_size = 0.36
R = 60e-3
Nf = np.int((Fend-Fint)/Fstep+1)
Freq = np.linspace(Fint,Fend,Nf)
fc = Freq[int((len(Freq)-1)/2)]
kf = 2*np.pi*Freq/(3e8);
Z0 = 50 #ohm

Vin = 1
# =======================================================

#Open FEKO files by using readFEKO library
# ======================================================
#s1ppath = "/Users/clement_yang/Desktop/AAL/Previous Data Set/Hsu-Chi/Die_Overlap_Res_Wiredipoley_s360r3.6f5to15r0.5_quarter"
#s1ppath = "C:/Users/sychen/Desktop/Imaging/Die_Overlap_Res_Wiredipoley_s360r3.6f5to15r0.5_quarter"
#S11,Zeq = readFEKO.get_S11(s1ppath, np.int((Nx+1)/2), np.int((Ny+1)/2), Nf)

#fileEinc = "C:/Users/sychen/Desktop/Imaging/C.-H/Single probe/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7.out"
#fileEinc = "/Users/clement_yang/Desktop/AAL/Previous Data Set/Hsu-Chi/C.-H./Single probe/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7.out"
#Ex,Ey,Ez = readFEKO.get_Einc(fileEinc, Nx, Ny, Nz, Nf)
#Iin,Zr = readFEKO.get_Zr_Iin(fileEinc,Nf)

# ======================================================
#Load Feko data in .npy form
# ======================================================
#folderpath = "D:/Desktop/Imaging/Crossblock3/numpy"
#folderpath = "D:/Desktop/Imaging/Python/Meatdata"
#folderpath = "D:/Desktop/Imaging/Conference_2/numpy/"
folderpath = "C:\\Users\\ELECTRONICS10\\Desktop\\Code_Data\\Four_blocks"
#folderpath = "D:/Desktop/Imaging/Single probe_Ben-20200204T171546Z-001/Single probe_Ben/Die_4_Doc/numpy"
#folderpath = "/Users/clement_yang/Desktop/AAL/Previous Data Set/StackProx"
#folderpath = "D:\Desktop\Imaging\Python\FR4_cross"
#folderpath = "D:/Desktop/Imaging/Bigblock2/numpy"
S11 = np.load(os.path.join(folderpath,"S11.npy"))
Zeq = np.load(os.path.join(folderpath,"Zeq.npy"))
Ex = np.load(os.path.join(folderpath,"Ex.npy"))
Ey = np.load(os.path.join(folderpath,"Ey.npy"))
Ez = np.load(os.path.join(folderpath,"Ez.npy"))
Iin = np.load(os.path.join(folderpath,"Iin.npy"))
Zr = np.load(os.path.join(folderpath,"Zr.npy"))
# ======================================================
xratio = np.arange(0,Nx,1)
yratio = np.arange(0,Ny,1)

Zint=54e-3
Zend=72e-3
zratio = np.arange(0,7,1) #The actual value of z is determined by the setting of the Feko Simulation !
z_pos = (Zint + ((zratio)*dz))*1e3; #Predetermined inspected z-cut

fratio = np.arange(0,7,1)
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

# ======================================================

#Plot characteristic S11 of antenna in the frequency range
# =======================================================
# S11fig = plt.figure()
# plt.plot(Freq/1e9, 20*np.log10(abs((np.ravel(Zr)-Z0))/(np.ravel(Zr)+Z0)), linewidth=2, marker='o') #Get S11 from Zr
# plt.xlabel('frequency (GHz)');
# plt.ylabel('|S\N{SUBSCRIPT ONE}\N{SUBSCRIPT ONE}| (dB)')
# plt.show()
# =======================================================

def Plot_S11_plane(i): #Plot the S11 plane with respect to different frequencies

    if i == 'half': #Input data is half plane S11
        S11_half = S11[0:(Ny+1)/2-1,xratio,fratio]
        Zeq_half = Zeq[0:(Ny+1)/2-1,xratio,fratio]
        S11_full = np.vstack((np.flip(S11_half,0),S11_half))
        np.delete(S11_full,51,0)
        Zeq_full = np.vstack((np.flip(Zeq_half,0),Zeq_half))
        np.delete(Zeq_full,51,0)
        
        for i in range(len(fratio)):
            fig = plt.figure()
            plt.imshow(abs(S11_full[:,:,i]),cmap="winter",interpolation="nearest")
            plt.title("S11 Figure at "+str(Freq[i]/10e8)+" GHz")
            
        return S11_full,Zeq_full
            
    elif i == 'quarter': #Input data is quarter plane S11
        S11_quarter = S11[0:np.int((Ny+1)/2),0:np.int((Ny+1)/2),fratio]
        Zeq_quarter = Zeq[0:np.int((Ny+1)/2),0:np.int((Ny+1)/2),fratio]
        
        S11_right = np.vstack((np.flip(S11_quarter,0),S11_quarter)) #Vertical flip and con
        S11_left = np.vstack((np.flip(np.flip(S11_quarter,0),1),np.flip(S11_quarter,1)))
        S11_full = np.hstack((S11_left,S11_right))
        S11_full = np.delete(S11_full,51,0)
        S11_full = np.delete(S11_full,51,1)
        
        Zeq_right = np.vstack((np.flip(Zeq_quarter,0),Zeq_quarter)) #Vertical flip and con
        Zeq_left = np.vstack((np.flip(np.flip(Zeq_quarter,0),1),np.flip(Zeq_quarter,1)))
        Zeq_full = np.hstack((Zeq_left,Zeq_right))
        Zeq_full = np.delete(Zeq_full,51,0)
        Zeq_full = np.delete(Zeq_full,51,1)
        
        #D3
        # Zeq_full[:,0,:] = Zeq_full[:,1,:]
        # Zeq_full[:,26,:] = Zeq_full[:,27,:]
        # Zeq_full[:,25,:] = Zeq_full[:,24,:]
        #fig = plt.figure()
        #plt.imshow(abs(S11_full[:,:,10]),cmap="winter",interpolation="nearest")
        #plt.title("epilon=10, S11 Figure at "+str(Freq[10]/10e8)+" GHz")
            
        return S11_full,Zeq_full
            
    elif i == "full": #Input data is full plane S11
        S11_full = S11
        Zeq_full = Zeq
        for i in range(len(fratio)):
            fig = plt.figure()
            plt.imshow(abs(S11_full[:,:,i]),cmap="winter",interpolation="nearest")
            plt.title("S11 Figure at "+str(Freq[i]/10e8)+" GHz")
        return S11_full,Zeq_full
            
S11_full,Zeq_full = Plot_S11_plane("full")


class algorithm:
    
    def AngularSpectrum(): #Obtain the angular spectrum (K space domain)of Einc (Orientation,ky,kx,kz,f)
        #Here Orientation = Ex, Ey, Ez
        
        RevExinc = np.rot90(Ex,k=2)
        RevEyinc = np.rot90(Ey,k=2)
        RevEzinc = np.rot90(Ez,k=2)
        
        #Born approximation is utilized to approximate Einc = Etot
        #Should not work with different formulation method
        g = np.multiply(RevExinc,RevExinc)+np.multiply(RevEyinc,RevEyinc)+np.multiply(RevEzinc,RevEzinc)
        g = (g[:,:,zratio,:])[:,:,:,fratio]
        
        # Shift the g function within space domain to the center of the dipole antenna
        G = np.fft.fft2(g,axes=(0,1))
        G = np.fft.fftshift(np.fft.fftshift(G,0),1)
        #Take FFT of g function and shift zero frequency to the center of the matrix
        
        Gprime = np.multiply(-1j*2*np.pi*Freq[fratio]*dz/Iin[:,:,fratio],G) #combine -j*w*dz/Iin(f) term to G
        return g,G,Gprime
    
    # Some notes about variable namings:
        # 1. Examine the Condition number of Gprime matrix
        #    =======>  stored in condnum(Ny,Nx)
        # 2. Examing the Singular value by svd operation
        #    =======>  stored in s(Ny,Nx,min(Nf,Nz)) 
        #    (Singular value is the diagonal entries of S matrix) 
        # 3. Solve the x:
        #    =======>  stored in f_raw(Nx,Ny,Nz)
        # 4. Error examination:
        #    =======>  stored in error(Nx,Ny)

    def Plot_condnum():
        condnum = np.linalg.cond(np.transpose(A,(2,3,1,0)),None)
        fig = plt.figure()
        #Visualize Condition Number
        
        fig = plt.figure()
        plt.pcolormesh(FXX,FYY,condnum)
        plt.xticks(2*np.pi*fxx[0,:])
        plt.yticks(2*np.pi*fyy[:,0])
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=9)
        plt.colorbar()
        plt.title("Condition Number of Gprime matrix (kx,ky) ")
        
        fig = plt.figure()
        plt.pcolormesh(FXX,FYY,condnum)
        plt.xticks(2*np.pi*fxx[0,:])
        plt.yticks(2*np.pi*fyy[:,0])
        plt.colorbar()
        plt.clim(0,100)
        plt.locator_params(axis='y', nbins=9)
        plt.locator_params(axis='x', nbins=9)
        plt.title("Condition Number of Gprime matrix with value limit (kx,ky)")
        
        # fig = plt.figure()
        # plt.pcolormesh(X,Y,condnum)
        # plt.xticks(2*np.pi*fxx[0,:])
        # plt.yticks(2*np.pi*fyy[:,0])
        # plt.colorbar()
        # plt.clim(0,100)
        # plt.hold()
        
        # plt.title("Condition Number of Gprime matrix with contour and limit")
   

    def F_matrix_NoComp(Gprime,Hermitian):
        #Solve object contrast function without considering the effect of propagation phase
        U,singularvalue,V = np.linalg.svd(Gprime, full_matrices=True)
         
        if Hermitian == "0":
            A = np.transpose(Gprime,(3,2,0,1))
            B = np.transpose(T,(2,0,1))
        
        elif Hermitian == "1":
        #Lossless assumption of the material, complex permittivity is real valued
        #Build auxiliary equations: F(-kx,-ky,zn) = Fbar(kx,ky,zn)
            A = np.transpose(Gprime,(3,2,0,1))
            A = np.concatenate((A,np.conj(A)),axis = 0)
            B = np.transpose(T,(2,0,1))
            B = np.concatenate((B,np.conj(B)),axis = 0)
            
            
        return A,B


    def F_matrix_PhaseComp(Gprime, Hermitian):
        #Solve object contrast function considering the effect of propagation phase
        #Only consider the z axis propagation in material
        U,singularvalue,V = np.linalg.svd(Gprime, full_matrices=True)
        
        epsr_maxZ = np.max(np.max(epsr,axis=0),axis=0)
        ermax = -np.sort(-epsr_maxZ)
        nmax = np.sqrt(ermax)
        
        Atemp = np.transpose(Gprime,(3,2,0,1))
        B = np.transpose(T,(2,0,1))
            
        kzprime = 2*np.pi*(fzz[fratio,:,:,:])[:,zratio,:,:]
            
        total_phase = np.empty((len(fratio),len(zratio),Ny,Nx),dtype="complex128")
        for i in range(len(nmax)):
            total_phase[:,i,:,:] = -1j*kzprime[:,i,:,:]*dz/2*(nmax[i]-1)
                
        A = Atemp*np.exp(total_phase)
            
        if Hermitian == "0":
            A = A
            B = B
        
        elif Hermitian == "1":
        #Lossless assumption of the material, complex permittivity is real valued
        #Build auxiliary equations: F(-kx,-ky,zn) = Fbar(kx,ky,zn)
            A = np.concatenate((A,np.conj(A)),axis = 0)
            B = np.concatenate((B,np.conj(B)),axis = 0)
            
            
        return A,B

    def F_matrix_EnhancedPhaseComp(Gprime,Type,Hermitian):
        #Phase compensation with radiation angle considerations
        #The maximum error is defined by regular \reconstruction, please run it first !
        
        #Type 1: Compensate for only object plane
        #Type 2: Compensate phase including the plane behind object
        #Type 3: Compensate phase for all planes behind object
        #Type 4: Compensate phase for Layer Structure and all the planes behind object
        
        threshold = 0.6
        n = er**0.5
        
        A = np.transpose(Gprime,(3,2,0,1))
        B = np.transpose(T,(2,0,1))

        if Type == '1':
            for f in range(len(fratio)):
                kz = np.conj(np.sqrt((2*kf[f])**2-FXX**2-FYY*2),dtype='complex128')
                kz[np.isnan(kz)] = 0
                for z in range(len(zratio)):
                    if np.max(abs(epsr[:,:,z]))>threshold*er:
                        ratio = kz/(2*kf[f])
                        ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*0.5))))]=10000
                        phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                        phi[phi>100] = 0
                        A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                
        elif Type == '2':
            for f in range(len(fratio)):
                flag = 0
                kz = np.conj(np.sqrt((2*kf[f])**2-FXX**2-FYY*2),dtype='complex128')
                kz[np.isnan(kz)] = 0
                for z in range(len(zratio)):
                    if np.logical_or(np.max(abs(epsr[:,:,z]))>threshold*er,flag):
                        if np.max(abs(epsr[:,:,z]))>threshold*er:
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*(flag+0.5)))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            phi[phi>100] = 0
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                            flag = 1
                        else:
                            flag = 0
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*0.5))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            phi[phi>100] = 0
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
        elif Type == '3':
            for f in range(len(fratio)):
                flag = 0
                kz = np.conj(np.sqrt((2*kf[f])**2-FXX**2-FYY*2),dtype='complex128')
                kz[np.isnan(kz)] = 0
                for z in range(len(zratio)):
                    if np.logical_or(np.max(abs(epsr[:,:,z]))>threshold*er,flag):
                        if np.max(abs(epsr[:,:,z]))>threshold*er:
                            flag = 1
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*0.5))))]=10000
                            ratio[np.logical_and(ratio<np.cos(np.pi/2),ratio<np.cos(np.arctan(Object_size/2/(dz*0.5))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            phi[phi>100] = 0
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                            
                        else:
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*(flag+0.5)))))]=10000
                            ratio[np.logical_and(ratio<np.cos(np.pi/2),ratio<np.cos(np.arctan(Object_size/2/(dz*(flag+0.5)))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                            flag = flag+1
                            
        elif Type == '4':
            have_object = np.zeros(len(zratio))
            for z in range(len(zratio)):
                if np.max(epsr[:,:,z])>threshold*er:
                    have_object[z] = 1
                    
            print(have_object)
            for f in range(len(fratio)):
                kz = np.conj(np.sqrt((2*kf[f])**2-FXX**2-FYY*2),dtype='complex128')
                kz[np.isnan(kz)] = 0
                flag = 0
                for z in range(len(zratio)):
                    if np.logical_or(have_object[z]==1,flag):
                        if have_object[z]==1:
                            flag = 1
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*0.5))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            phi[phi>100] = 0
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                        else:
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*(flag+0.5)))))]=10000
                            phi = 2*kf[f]*(n-1)*(dx*ratio/2)
                            phi[phi>100] = 0
                            A[f,z,:,:] = A[f,z,:,:]*np.exp(-1j*phi)
                            flag = 1
                    overlap=0
                                                
                    for j in range(z+1,len(zratio)):
                        if have_object[z]+have_object[j]==2:
                            overlap = 1
                            layer_diff = j-z
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*(layer_diff+0.5)))))]=10000
                            phi = 2*kf[f]*(n-1)*(dz/ratio)
                            phi[phi>100] = 0
                            A[f,j,:,:] =  A[f,j,:,:]*np.exp(-1j*phi)
                        elif(overlap):
                            layer_diff = j-z
                            ratio = kz/(2*kf[f])
                            ratio[np.logical_and(ratio<np.cos(np.arctan(Aperture_size/2/z_pos[z])),ratio<np.cos(np.arctan(Object_size/2/(dz*(layer_diff+0.5)))))]=10000
                            phi = 2*kf[f]*(n-1)*(dz/ratio)
                            phi[phi>100] = 0
                            A[f,j,:,:] =  A[f,j,:,:]*np.exp(-1j*phi)
                            
        if Hermitian == "0":
            A = A
            B = B
        
        elif Hermitian == "1":
        #Lossless assumption of the material, complex permittivity is real
        #Build auxiliary equations: F(-kx,-ky,zn) = Fbar(kx,ky,zn)
            A = np.concatenate((A,np.conj(A)),axis = 0)
            B = np.concatenate((B,np.conj(B)),axis = 0)
        
        return A,B
    
    
    def Solve_LSTSQ_Error(A,B):
        #Begin solving the linear equations by least square method and error analysis
        X = []
        Error = []
        for i in range(Nx):
            for j in range(Ny):
                x,res,rank,s = linalg.lstsq(A[:,:,i,j],B[:,i,j])
                X.append(x)
                #Error analysis by Hsu-Chi (function)
                err = np.arccos(abs(np.inner(np.inner(A[:,:,i,j],x),B[:,i,j]))/(np.linalg.norm(np.inner(A[:,:,i,j],x))*np.linalg.norm(B[:,i,j])))*180/np.pi
                Error.append(err)
        X = np.array(X)
        Error = np.array(Error)
        #return A,B,X
        X = X.reshape((Ny,Nx,len(zratio)))
        Error = Error.reshape((Ny,Nx))
        
        # F = np.fft.ifftshift(np.fft.ifftshift(X,0),1)
        # f = np.fft.ifft2(F,axes = (0,1))
        # f_raw = np.roll(f,np.int(-(Ny-1)/2),axis=0)
        # f_raw = np.roll(f_raw,np.int(-(Nx-1)/2),axis=1)
        # epsr = (f_raw/epsz/(dx*dy))+1 #Permittivity of real space object, epsilon r
                            
        return X
    
    
    def plot_results():
        #Visualize F matrix
        F = np.fft.ifftshift(np.fft.ifftshift(X,0),1)
        f = np.fft.ifft2(F,axes = (0,1))
        f_raw = np.roll(f,np.int(-(Ny-1)/2),axis=0)
        f_raw = np.roll(f_raw,np.int(-(Nx-1)/2),axis=1)
        epsr = (f_raw/epsz/(dx*dy))+1 #Relative permittivity of real space object, epsilon r
        for i in range(len(z_pos)):
            fig = plt.figure()
            plt.pcolormesh(FXX,FYY,np.abs(X[:,:,i]))
            plt.xticks(2*np.pi*fxx[0,:])
            plt.yticks(2*np.pi*fyy[:,0])
            #plt.clim(0,4e-13)
            plt.clim(0,6e-15)
            plt.colorbar()
            plt.locator_params(axis='y', nbins=9)
            plt.locator_params(axis='x', nbins=9)
            plt.title("|F(kx,ky,z = "+str(float(z_pos[i]))+" mm)| of "+Title)
            
        #Visualize object matrix
        for i in range(len(z_pos)):
            fig = plt.figure()
            plt.pcolormesh(xx,yy,abs(epsr[:,:,i]))
            plt.xticks(xx[0,:])
            plt.yticks(yy[:,0])
            #plt.clim(1,2)
            plt.clim(1,np.max(abs(epsr)))
            #Since we should have epsilon larger than 1
            plt.colorbar()
            plt.locator_params(axis='y', nbins=7)
            plt.locator_params(axis='x', nbins=7)
            plt.title(Title+" at "+str(float(z_pos[i]))+" mm")
        return epsr
        
        #Error analysis by Hsu-Chi continued
        # fig = plt.figure()
        # plt.pcolormesh(FXX,FYY,Error)
        # plt.xticks(2*np.pi*fxx[0,:])
        # plt.yticks(2*np.pi*fyy[:,0])
        # #plt.clim(0,45)
        # plt.colorbar()
        # plt.locator_params(axis='y', nbins=7)
        # plt.locator_params(axis='x', nbins=7)
        # plt.title("Total Imaging Error")
        
class Filter:
    
    def quick_filter(X,rad,Nx,Ny,z_pos):
    #Zero out values out of specified k-circle radius
    #rad = 18 for clearest reconstruction
        for k in range(len(z_pos)):
            for i in range(Ny):
                for j in range(Nx):
                    if ((i-(Ny-1)/2)/rad)**2+((j-(Nx-1)/2)/rad)**2>1:
                        X[i,j,k] = 0
                        
    def aperture_filter(X,Nx,Ny,z_pos):
    # Aperture_size: Determine the desired data range for recontruction
    # Matrix mimics the calculations of MATLAB freqspace function
        kc = (2*np.pi)*fc/(2.9979e8);
        for k in range(len(z_pos)):
            theta = np.arctan(Aperture_size/2/(z_pos[k]*1e-3))
            limit = 2*kc*np.sin(theta)/((2*np.pi)/dx/(Nx)*((Nx-1)/2))
            for i in range(Ny):
                for j in range(Nx):
                    if ((i-(Ny-1)/2)/(limit*(Ny-1)/2))**2+((j-(Nx-1)/2)/(limit*(Nx-1)/2))**2>1:
                        X[i,j,k] = 0

class savefile:
    
    def savecurr(name,file,path):
        savepath = path
        np.save(os.path.join(savepath,name),file)
        
        
        
if __name__ == "__main__":
    
    g,G,Gprime = algorithm.AngularSpectrum()
    #g = np.load(os.path.join(folderpath,"g.npy"))
    #psfpath = "D:/Desktop/Imaging/PSF/npy"
    #g = np.load(os.path.join(psfpath,"kernel.npy"))
    
    repZr = np.tile(Zr,[Ny,Nx,1])
    Voc = Vin*(repZr[:,:,fratio] + Z0)/repZr[:,:,fratio]*(Zeq_full - repZr[:,:,fratio])/(Zeq_full + Z0)
    #The open circuit voltage of antenna at given frequencies         
    #Voc = np.roll(Voc,np.int(-(Ny-1)/2),axis=0)
    #Voc = np.roll(Voc,np.int(-(Nx-1)/2),axis=1)
    # Shift the Voc function within space domain to the center of the dipole antenna
    T = np.fft.fft2(Voc,axes=(0,1))
    T = np.fft.fftshift(np.fft.fftshift(T,0),1)
    #Take FFT of Voc function and shift zero frequency to the center of the array
    A,B = algorithm.F_matrix_NoComp(Gprime,"0")
    
    #algorithm.Plot_condnum()
    X = algorithm.Solve_LSTSQ_Error(A, B)
    #X[26,:,:] = 0
    Filter.aperture_filter(X,Nx,Ny,z_pos)
    Title = "Uncompensated object"
    epsr = algorithm.plot_results()
    
    er = np.max(np.real(epsr))
    
    # ========================================================
    #Phase compensation with radiation angle considerations
        #The maximum error is defined by regular \reconstruction, please run it first !
        
        #Type 1: Compensate for only object plane
        #Type 2: Compensate phase including the plane behind object
        #Type 3: Compensate phase for all planes behind object
        #Type 4: Compensate phase for Layer Structure and all the planes behind object
    # ========================================================
    
    C,B = algorithm.F_matrix_PhaseComp(Gprime, "1")
    #D,B = algorithm.F_matrix_EnhancedPhaseComp(Gprime,"1","1")
    #E,B = algorithm.F_matrix_EnhancedPhaseComp(Gprime,"2","1")
    #F,B = algorithm.F_matrix_EnhancedPhaseComp(Gprime,"3","1")
    #G,B = algorithm.F_matrix_EnhancedPhaseComp(Gprime,"4","1")
    
    #Plot_condnum()
    X = algorithm.Solve_LSTSQ_Error(C, B)
    #X[50,:,:] = 0
    Filter.aperture_filter(X,Nx,Ny,z_pos)
    Title = "Simple PC object"
    algorithm.plot_results()
    
    #Plot_condnum()
    # X = algorithm.Solve_LSTSQ_Error(D, B)
    # X[50,:,:] = 0
    # Filter.aperture_filter(X,Nx,Ny,z_pos)
    # Title = "PC at object plane"
    # algorithm.plot_results()
    
    #algorithm.Plot_condnum()
    # X = algorithm.Solve_LSTSQ_Error(E, B)
    # X[50,:,:] = 0
    # Filter.aperture_filter(X,Nx,Ny,z_pos)
    # Title = "PC of plane behind object"
    # algorithm.plot_results()
    
    # #algorithm.Plot_condnum()
    # X = algorithm.Solve_LSTSQ_Error(F, B)
    # Filter.aperture_filter(X,Nx,Ny,z_pos)
    # Title = "PC all planes behind object"
    # algorithm.plot_results()
    
    # #algorithm.Plot_condnum()
    #X = algorithm.Solve_LSTSQ_Error(G, B)
    #Filter.aperture_filter(X,Nx,Ny,z_pos)
    #Title = "PC layered structure"
    #algorithm.plot_results()
    
    epsr = abs(np.transpose(epsr[35:65,35:65,:],(0,2,1)))
    #epsr[epsr<1.2]=1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cmap = plt.get_cmap("Purples")
    norm= plt.Normalize(epsr.min(), epsr.max())
    ax.voxels(np.ones_like(epsr), facecolors=cmap(norm(epsr)))

    plt.show()