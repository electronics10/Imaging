import numpy as np
import os
from natsort import natsorted
import re

class readFEKO:
    
    def get_S11(filepath,Nx,Ny,Nf):
        # Input a total of Nx*Ny .s1p files ! 
        
        Z0 = 50
        NumOfFile = Nx*Ny
        
        fil = natsorted([i for i in os.listdir(filepath) if i.endswith('.s1p')]) #sort all files ending with .s1p according to serial number
        fillength = len(fil)
        
        if (fillength != NumOfFile): print("Warning: Mismatch between input file number and sorted file number !")
        # Mismatch will cause reshape error !    
        s11 = []
        frq = []
        s11p = []
        count = 0
        for pos in fil:
            count += 1
            if count>NumOfFile: break
            reads1p = open(os.path.join(filepath, pos))
            reader = reads1p.read()  
            reader = reader.split('<S11\n')[1].replace('\n',' ').split()
            s11.append(reader[1::3])
            s11p.append(reader[2::3])
        frq.append(reader[0::3]) #Frequency should remain the same for all data.
        S11 = np.array(s11,dtype=float)*np.exp(np.array(s11p,dtype=float)*1j/180*np.pi) #Convert to complex value
        
        S11 =  np.reshape(S11,(Ny,Nx,Nf))
        Zeq = Z0*(S11 + 1)/(1 - S11)
        
        reads1p.close()
        return S11,Zeq #Three dimensional complex arrays
    
    def get_Zr_Iin(filename,Nf):
        # Input the .out file !
        # ==============================================
        #           Antenna is at the Origin
        #     Acquire the Antenna's Characteristics
        #             in different Freq.
        # ==============================================
        #   Zr[1,1,Nf] = (Impedance[1,1], Frequency) = (Impedance, Freq)
        #   Iin[1,1,Nf] = (Iin[1,1], Frequency) = (Iin, Freq)
        lines = []
        with open(filename,'rt') as fid:
            for scan in fid:
                lines.append(scan.rstrip('\n'))
        CurrPat = re.compile(".*Current   in A.*")
        Curr = list(filter(CurrPat.match,lines))
        Curr = [i.split('Current   in A')[1] for i in Curr]
        ZPat = re.compile(".*Impedance in Ohm.*")
        Z =  list(filter(ZPat.match,lines))
        Z = [i.split('Impedance in Ohm')[1] for i in Z]
        
        Ilist = np.array(str(Curr).replace("[", "").replace("]", "").replace(",", " ").replace("'","").split(),dtype=float)
        Zlist = np.array(str(Z).replace("[", "").replace("]", "").replace(",", " ").replace("'","").split(),dtype=float)
      
        Iin = Ilist[0::4]+Ilist[1::4]*1j
        Zr = Zlist[0::4]+Zlist[1::4]*1j
        
        Iin = np.reshape(Iin,(1,1,Nf))
        Zr = np.reshape(Zr,(1,1,Nf))        
        return Iin,Zr
        
    def get_Einc(filename,Nx,Ny,Nz,Nf):
        # Input the .out file !
        # =================================================================
        #            Get Einc of antenna placed at the origin
        #            at different freq.
        # ================================================================
        # (x,y) representation:  (-0.18, -0.18)       ....      (0.18, -0.18)  
        #                             .....                         ...
        #                        (-0.18, 0.18)        ....      (0.18, 0.18)
        # (Ny,Nx)
        #
        # Nx = 101
        # Ny = 101
        # Nz = 13
        # Nf = 21
        readE = open(filename)
        lines = readE.read()
        Ematch = re.findall('phase      magn.     phase\n(.*?)\n read from buffer:',lines,re.DOTALL)
        Eraw = []
        for freqData in Ematch:
            Eraw.append(np.float64(freqData.split()))
        Eraw = np.array(Eraw)
        # ---------Array Formatting, handle with care !-----------------
        Eraw = Eraw.reshape(Nf,1,Nx*Ny*Nz*10) #frequency as Z-axis
        Eraw = Eraw.reshape(Nf,Nx*Ny*Nz,10) #Same format as .out file, with frequency as Z-axis
        #-------------------------------------------------------------
        
        Ex = Eraw[:,:,4].reshape(Nf,Nx*Ny*Nz,1)*np.exp(Eraw[:,:,5].reshape(Nf,Nx*Ny*Nz,1)*1j/180*np.pi)
        Ey = Eraw[:,:,6].reshape(Nf,Nx*Ny*Nz,1)*np.exp(Eraw[:,:,7].reshape(Nf,Nx*Ny*Nz,1)*1j/180*np.pi)
        Ez = Eraw[:,:,8].reshape(Nf,Nx*Ny*Nz,1)*np.exp(Eraw[:,:,9].reshape(Nf,Nx*Ny*Nz,1)*1j/180*np.pi)
        
        Ex = Ex.reshape(Nf,Nz,Ny,Nx)
        Ey = Ey.reshape(Nf,Nz,Ny,Nx)
        Ez = Ez.reshape(Nf,Nz,Ny,Nx)
        
        Ex = np.transpose(Ex,(2,3,1,0))
        Ey = np.transpose(Ey,(2,3,1,0))
        Ez = np.transpose(Ez,(2,3,1,0)) #Swap array axis to match get_S11 format
        #Format: [Ny,Nx,Nz,Nf]

        return Ex,Ey,Ez #Four dimensional complex arrays
    
if __name__ == "__main__":
    #Basic file settings
    # =======================================================
    Nx = 101    
    Ny = 101
    Nz = 7
    
    dx = 7.2e-3
    dy = 7.2e-3
    dz = 3e-3
    
    Fint =  8.5e9         # Hz
    Fend =  11.5e9       # Hz
    Fstep = 0.5e9        # Hz
    
    R = 60e-3
    Nf = np.int((Fend-Fint)/Fstep+1)
    Freq = np.linspace(Fint,Fend,Nf)
    Z0 = 50
    
    epsz = 8.854187817*1e-12      # F/m
    muz = 4*np.pi*1e-7               # H/m              
    c = (epsz*muz)**(-1/2)         # m/s
    eta = np.sqrt(muz/epsz)          # ohm
    # =======================================================
    

    #filepath = "/Users/clement_yang/Desktop/AAL/Previous Data Set/Hsu-Chi/Die_Overlap_Res_Wiredipoley_s360r3.6f5to15r0.5_quarter"
    #filepath = "C:/Users/sychen/Desktop/Imaging/Single probe_Ben-20200204T171546Z-001/Single probe_Ben/Die_4_Doc/"
    filepath = 'D:/Desktop/Imaging/Conference/'
    #pos = "Die_Overlap_Res_Wiredipoley_s360r3.6f5to15r0.5_quarter_opt_1_SParameter1.s1p"
    #S11,Zeq = readFEKO.get_S11(filepath, np.int((Nx+1)/2), np.int((Ny+1)/2), Nf)
    S11,Zeq = readFEKO.get_S11(filepath, Nx, Ny, Nf)
    #pcs = np.abs(S11[:,:,1])
    #fileEinc = "/Users/clement_yang/Desktop/AAL/Previous Data Set/Hsu-Chi/C.-H./Single probe/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7.out"
    #fileEinc = "C:/Users/sychen/Desktop/Imaging/C.-H/Single probe/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7/E_inc_dipolex_s360r3.6d42to78r3f5to15r0.5_v7.out"
    #fileEinc = 'D:\Desktop\Imaging\Singleblock\Singleblock_s360r3.6f5to15r0.5_quarter.out'
    #fileEinc = 'D:/Desktop/Imaging/Conference_2/Conference.out'
    #Ex,Ey,Ez = readFEKO.get_Einc(fileEinc, Nx, Ny, Nz, Nf)
    #Iin,Zr = readFEKO.get_Zr_Iin(fileEinc,Nf)
    
    
    #Save data in .npy form
    # ======================================================
    #savepath = "/Users/clement_yang/Desktop/AAL/Previous Data Set/BenData"
    # savepath = "D:\Desktop\Imaging\Single probe_Ben-20200204T171546Z-001\Single probe_Ben\Die_4_Doc\numpy"
    # np.save(os.path.join(savepath,"S11"),S11)
    # np.save(os.path.join(savepath,"Zeq"),Zeq)
    # np.save(os.path.join(savepath,"Iin"),Iin)
    # np.save(os.path.join(savepath,"Zr"),Zr)
    # np.save(os.path.join(savepath,"Ex"),Ex)
    # np.save(os.path.join(savepath,"Ey"),Ey)
    # np.save(os.path.join(savepath,"Ez"),Ez)
