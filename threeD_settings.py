import numpy as np


###==============File Settings===========
epsz = 8.854187817*1e-12      # F/m
muz = 4*np.pi*1e-7               # H/m              
c = (epsz*muz)**(-1/2)         # m/s
eta = np.sqrt(muz/epsz)          # ohm

Nx = 51
Ny = Nx
samp = -(101//-Nx) # ceiling division
Nz = 13

dx = 3.6e-3*samp #meters
dy = 3.6e-3*samp #meters
dz = 3e-3 #meters

Fint =  5e9         # Initial Frequency (Hz)
Fend =  15e9       # End Frequency (Hz)
Fstep = 0.5e9        # Frequency Step (Hz)

Aperture_size = 360e-3 #Determine the desired data range for recontruction
Object_size = 18e-3
R = 60e-3
Nf = int((Fend-Fint)/Fstep+1)
Freq = np.linspace(Fint,Fend,Nf)
fc = Freq[int((len(Freq)-1)/2)]
kf = 2*np.pi*Freq/(3e8)
Z0 = 50 #ohm

xratio = np.arange(0,Nx,1)
yratio = np.arange(0,Ny,1)

Zint=42e-3  # Initial z-axis postion
Zend=78e-3
zratio = np.arange(6,9,1) #The actual value of z is determined by the setting of the Feko Simulation !
z_pos = (Zint + (zratio*dz))*1e3; #Predetermined inspected z-cut

fratio = np.arange(10,11,1)

kf = kf[fratio]
theta = np.arctan((Nx-1)/2*dx/R) ; #Angle related to Scanning Aperture  

SampMax = 2*np.pi*(Nx-1)/Nx/2/dx #Highest sampling wavenumber (1/m)
wavenum = 2*np.pi*2*Freq/3e8
#=====================


###========Plot Settings==========
Tx = Nx*dx #Real space X axis
Ty = Ny*dy #Real space Y axis
dfx = np.round(1/Tx,decimals=5)
dfy = np.round(1/Ty,decimals=5) #Use round for Floating-point error mitigation
xx,yy = np.meshgrid(np.arange(-(Nx-1)/2*dx,(Nx-1)/2*dx+dx,dx),\
    np.arange(-(Ny-1)/2*dy,(Ny-1)/2*dy+dy,dy)) #Real space X,Y window
fxx,fyy = np.meshgrid(np.arange(-(Nx-1)/2*dfx,(Nx-1)/2*dfx+dfx,dfx),\
    np.arange(-(Ny-1)/2*dfy,(Ny-1)/2*dfy+dfy,dfy)) #2pi convert to sampling wavenumber
FXX,FYY = np.meshgrid(2*np.pi*fxx[0,:],2*np.pi*fyy[:,0])
fzz = np.conj(np.sqrt(np.transpose(np.tile(np.reshape((2*Freq/c)**2,\
    (1,1,1,len((2*Freq/c)**2))),(Ny,Nx,Nz,1)),\
        (3,2,1,0))-np.tile((fxx**2 + fyy**2),(Nf,Nz,1,1)),dtype="complex128"))
#===========================

