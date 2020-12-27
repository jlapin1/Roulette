"""
120319: Changed the starting rho and propagation method. Should be standardized going forward.
        rho_n+1 = exDtot * rho_n * exDtot_i
        G[n+1] = trace( rho*rho_det )
"""
import numpy as np
import scipy.signal as ss
import scipy.sparse as sp
import scipy.linalg as la
from scipy.optimize import fmin
from copy import copy
from time import time
import threading
import queue
from math import factorial as fact
import os
import sys
spgen = sp.csc_matrix
cos = np.cos
sin = np.sin
pi = np.pi
d2r = pi/180
i = np.complex64(np.complex(0,1))
hbar=1.055e-34

class config():
    def __init__(self):
        self.v = {
                'ficoor':"C:/Users/joell/Documents/Python/Nevzorov/SPIN/ala.txt",
                'xind':0,
                'dlim':3.0,
                'fisym':"C:/Users/joell/Documents/Python/Nevzorov/SPIN/sym.inp",
                'fiRNG':"C:/Users/joell/Documents/Python/Nevzorov/SPIN/RNG.inp",
                'new':1,
                'firefpath':"C:/Users/joell/Documents/Python/Nevzorov/SPIN/Loop/refinement/",
                'refrng':[0.3,0.3],
                
                'fopath':"C:/Users/joell/Documents/Python/Nevzorov/SPIN/Loop/",
                'foPlow':1,
                'foPlowraw':0,
                'fowid':0,
                'foreport':1,
                'foRNG':1,
                'fosgm':1,
                'foHS':0,
                'foconf':1,
                'shutdown':0,
                
                'gamS':-2.718E7,
                'gamI':2.675E8,
                'Hhml':[20E3,10E3,5E3],
                'sgmr':0,
                'sgmstd':1,
                'sigs':[2,-2,2,4,1,-3,2,3],
                
                'L':6,
                'SwrfA':[1,1,1,1,1,1],
                'IwrfA':[1,1,0,0,1,1],
                
                'temp':1,
                'Ws':[1,8],
                'Wab':3,
                'Wathr':-4,
                'Wap':1E5,
                
                'runs':1,
                'incr':200,
                'steps':100,
                
                'numln':[4,5,7],
                'fus':[1.0,1.0,1.0],
                
                'wrf':58.14E3,
                'W0':500E6,
                'Nm1':256,
                'dnu':0,
                'z':0.5
                
                }
        self.conf()
        self.sym = np.loadtxt(self.v['fisym'], dtype='int', delimiter=',')
        self.RNG = np.loadtxt(self.v['fiRNG'], dtype='float32', delimiter=',')
        self.readcoords()
        self.qc()
        
    def conf(self):
        files = os.listdir()
        name = files[np.argmax(['.conf' in m for m in files])]
        with open(name, "r") as f:
            for line in f:
                ln = line.strip().split("\t") # tab separated fields
                if ln[0][0]!='#':
                    ln2 = [] # array which will hold only non-whitespace entries
                    for m in ln:
                        if m!='':
                            ln2.append(m)
                    if len(ln2)!=3:
                        sys.exit("QCError: Please make sure all non-commented lines in the configuration file contain 3 fields: varname datatype value")
                    self.v[ln[0]] = ln[-1]
            f.seek(0)
            self.v['text'] = f.read()
            
    
    def readcoords(self):
        
        self.v['gamS'] = float(self.v['gamS']) # conversion
        self.v['gamI'] = float(self.v['gamI']) # conversion
        with open(self.v['ficoor'],'r') as f:
            f.readline()
            self.restyp=[];self.gamma=[];self.coords=[]
            for line in f:
                buf = line.split()
                self.restyp.append(buf[0])
                if 'H' in buf[0]:
                    self.gamma.append(self.v['gamI'])
                else:
                    self.gamma.append(self.v['gamS'])
                self.coords.append([float(m) for m in buf[-3:]])
        self.coords = np.array(self.coords)
        self.gamma = np.array(self.gamma)
        Nmax,_ = self.coords.shape
        
        self.v['xind'] = int(self.v['xind']) # conversion
        self.v['dlim'] = float(self.v['dlim']) # conversion
        inds = []
        for m in range(0,Nmax,1):
            r_jk = ((self.coords[self.v['xind'],0]-self.coords[m,0])**2 + (self.coords[self.v['xind'],1]-self.coords[m,1])**2 + (self.coords[self.v['xind'],2]-self.coords[m,2])**2)**0.5 
            if r_jk<self.v['dlim']:
                inds.append(m)
                if r_jk==0:
                    xind = len(inds)-1
        self.X = self.coords[inds]
        self.v['xind'] = xind
        self.gamma = self.gamma[inds]
        self.N = len(inds)
                    
    def qc(self):
        tick=0
        # Input options
        if os.path.isfile(self.v['ficoor'])==False:
            print("QCError: Structure file %s doesn't exist"%(self.v['ficoor']));tick+=1
        if (self.v['xind']<0) & (self.v['xind']>=self.N):
            print("QCError: Index of x-atom must be between 0-%d"%(self.N-1));tick+=1
#        else:
#            if self.restyp[self.v['xind']]=='H':
#                print("QCError: Index of x-atom refers to an I spin");tick+=1
        if os.path.isfile(self.v['fisym'])==False:
            print("QCError: Symmetry file %s doesn't exist"%(self.v['fisym']));tick+=1
        if os.path.isfile(self.v['fiRNG'])==False:
            print("QCError: Symmetry file %s doesn't exist"%(self.v['fisym']));tick+=1
        self.v['new'] = bool(int(self.v['new'])) # conversion
        if self.v['new']==False:
            if os.path.isdir(self.v['firefpath'])==False:
                print("QCError: Path to refinement files %s is not a directory"%(self.v['firefpath']));tick+=1
        self.v['refrng'] = [float(m) for m in self.v['refrng'].split(",")] # conversion
        
        # Output options
        if os.path.isdir(self.v['fopath'])==False:
            print('QCError: Output path %s is not a directory'%(self.v['fopath']));tick+=1
        self.v['foPlow'] = bool(int(self.v['foPlow'])) # conversion
        self.v['foPlowraw'] = bool(int(self.v['foPlowraw'])) # conversion
        self.v['fowid'] = bool(int(self.v['fowid'])) # conversion
        self.v['foreport'] = bool(int(self.v['foreport'])) # conversion
        self.v['foRNG'] = bool(int(self.v['foRNG'])) # conversion
        self.v['fosgm'] = bool(int(self.v['fosgm'])) # conversion
        self.v['foHS'] = bool(int(self.v['foHS'])) # conversion
        self.v['foconf'] = bool(int(self.v['foconf'])) # conversion
        self.v['shutdown'] = bool(int(self.v['shutdown'])) # conversion
        
        # Hamiltonian settings
        self.v['Hhml'] = np.sort([np.float32(m) for m in self.v['Hhml'].split(",")])[::-1] # conversion
        # if len(self.v['Hhml'])!=3:
        #     print('QCError: Max couplings list Hhml must contain 3 comma separated floats');tick+=1
        self.v['sgmr'] = bool(int(self.v['sgmr'])) # conversion
        self.v['sgmstd'] = np.float32(self.v['sgmstd']) # conversion
        if self.v['sgmstd']<0:
            print('QCError: sgmstd must be >= 0');tick+=1
        self.v['sigs'] = np.array([float(m) for m in self.v['sigs'].split(",")]) # conversion
        if (self.v['sgmr']==False) & (len(self.v['sigs'])<self.N):
            print("QCError: For sgmr=0, sigs list must contain %d floats");tick+=1
        
        # PS Architecture
        self.v['L'] = int(self.v['L']) # Conversion
        self.v['SwrfA'] = [np.float32(m) for m in self.v['SwrfA'].split(",")] # conversion
        if len(self.v['SwrfA'])!=self.v['L']:
            print("QCError: S-channel wrf amplitude (SwrfA) must contain %d comma separated booleans"%(self.v['L']));tick+=1
        self.v['IwrfA'] = [np.float32(m) for m in self.v['IwrfA'].split(",")] # conversion
        if len(self.v['IwrfA'])!=self.v['L']:
            print("QCError: I-channel wrf amplitude (IwrfA) must contain %d comma separated booleans"%(self.v['L']));tick+=1
        if np.shape(self.RNG)[0]!=14:
            print("QCError: RNG input file must contain 14 rows (Top 7 -> floor vals; Bot 7 -> ceil vals)");tick+=1
        if np.shape(self.RNG)[1]!=self.v['L']:
            print("QCError: RNG input file must contain %d columns (same as L)"%(self.v["L"]));tick+=1
        if np.shape(self.sym)[0]!=7:
            print("QCError: sym input file must contain 7 rows");tick+=1
        if np.shape(self.sym)[1]!=self.v['L']:
            print("QCError: sym input file must contain %d columns (same as L)"%(self.v["L"]));tick+=1
        
        # Temperature settings
        self.v['temp'] = int(self.v['temp']) # conversion
        if self.v['temp'] not in [0,1]:
            print("QCError: temp can only be 0 (schedule) or 1 (adaptive)");tick+=1
        self.v['Ws'] = [np.float32(m) for m in self.v['Ws'].split(",")] # conversion
        self.v['Wab'] = np.float32(self.v['Wab']) # conversion
        self.v['Wathr'] = np.float32(self.v['Wathr']) # conversion
        self.v['Wap'] = np.float32(self.v['Wap']) # conversion
        if self.v['temp']==0:
            if len(self.v['Ws'])!=2:
                print("QCError: Ws must contain 2 comma separated floats: T_start,T_end");tick+=1
        else:
            if self.v['Wab']<=0:
                print("QCError: Baseline adaptive temperature Wab must be > 0");tick+=1
            if self.v['Wap']<1:
                print("QCError: Adaptive temperature parameter Wap must be > 1");tick+=1
        
        # MC settings
        self.v['runs'] = int(self.v['runs']) # conversion
        if self.v['runs']<1:
            print("QCError: runs must be > 0");tick+=1
        self.v['incr'] = int(self.v['incr']) # conversion
        if self.v['incr']<1:
            print("QCError: incr must be > 0");tick+=1
        self.v['steps'] = int(self.v['steps']) # conversion
        if self.v['steps']<1:
            print("QCError: steps must be > 0");tick+=1
        
        # Score settings
        self.v['numln'] = [int(m) for m in self.v['numln'].split(",")] # conversion
        if sum(np.argsort(self.v['numln'])==[0,1,2])!=3:
            print("QCError: Number line cutoffs must be in ascending order");tick+=1
        self.v['fus'] = [float(m) for m in self.v['fus'].split(',')] # conversion
        
        # Miscellaneous options
        self.v['wrf'] = np.float32(self.v['wrf']) # conversion
        if self.v['wrf']<=0:
            print("QCError: wrf must be > 0");tick+=1
        self.v['W0'] = np.float32(self.v['W0']) # conversion
        if self.v['W0']<=0:
            print("QCError: Carrier frequency W0 must be > 0");tick+=1
        self.v['Nm1'] = int(self.v['Nm1']) # conversion
        if self.v['Nm1']<=0:
            print("QCError: t1 points Nm1 must be an int between 0 < Nm1 < 1024");tick+=1
        self.v['dnu'] = np.float32(self.v['dnu']) # conversion
        if self.v['dnu']<0:
            print("QCError: Linebroadening parameter dnu must be >= 0");tick+=1
        self.v['z'] = np.float32(self.v['z']) # conversion
        if self.v['z']<0:
            print("QCError: Score penalty coefficient z must be >= 0");tick+=1
        
        # sym.inp
        if (sum(0==self.sym[4])!=0) | (sum(1==self.sym[4])!=0):
            print("QCError: timing variable(s) assigned Full symmetry (0)\n");tick+=1
        if (sum(0==self.sym[5])!=0) | (sum(1==self.sym[6])!=0):
            print("QCError: rf amplitude variable(s) assigned Full symmetry (0)\n");tick+=1
        
        if tick>0:
            sys.exit("Please fix %d input error(s)"%(tick))

class scobj():
    def __init__(self):
        self.pm = 0
        self.pmraw = 1E10
        self.wid = 1E10
        self.phase = np.zeros((7, L))
    
    def copyfrom(self, src, phase=False):
        self.pm = src.pm
        self.pmraw = src.pmraw
        self.wid = src.wid
        if phase==True:
            self.phase = copy(src.phase)
   
def dagger(A):
    return A.conjugate().transpose()

def operator(spmat, pos, tot):
    dt = spmat.dtype
    if sp.issparse(spmat)==True:
        kron = sp.kron
        eye = sp.eye
    else:
        kron = np.kron
        eye = np.eye
    
    if pos==0:
        return kron(spmat, eye(2**(tot-1), dtype=dt) )
    elif pos==tot-1:
        return kron(eye(2**(tot-1), dtype=dt), spmat)
    else:
        return kron(kron(sp.eye(2**pos, dtype=dt), spmat), eye(2**(tot-pos-1), dtype=dt) )

def operator2(spmat1, spmat2, pos1, pos2, tot):
    kron = sp.kron if (sp.issparse(spmat1)==True) else np.kron
    ib = pos2-pos1
    mat1 = operator(spmat1, pos1, pos1+ib)
    mat2 = operator(spmat2, 0, tot-pos2)
    return kron(mat1, mat2)

def rotaxis(ang, u):
    R = np.zeros((3,3))
    u = u/np.linalg.norm(u)
    cos = np.cos(ang)
    sin = np.sin(ang)
    R[0,0] = cos + (u[0]**2)*(1-cos)
    R[1,1] = cos + (u[1]**2)*(1-cos)
    R[2,2] = cos + (u[2]**2)*(1-cos)
    R[0,1] = u[0]*u[1]*(1-cos) - u[2]*sin
    R[1,0] = u[0]*u[1]*(1-cos) + u[2]*sin
    R[0,2] = u[0]*u[2]*(1-cos) + u[1]*sin
    R[2,0] = u[0]*u[2]*(1-cos) - u[1]*sin
    R[1,2] = u[1]*u[2]*(1-cos) - u[0]*sin
    R[2,1] = u[1]*u[2]*(1-cos) + u[0]*sin
    return R

def func(parms, coords, coup):
    rot = parms[0]
    axis = parms[1:]
    axis /= np.linalg.norm(axis)
    outs = INTAXMat(coords, rot=rot, axis=axis)
    return abs(coup-outs[-1]/2/pi)

def findrots(coords, coup, its=10):
    low = [1E10, []]
    for m in range(its):
        rot = np.random.uniform(0, 180)
        axis = [np.random.uniform() for n in range(3)]
        start = [rot]+axis
        parms,score,_,_,_ = fmin(func, start, args=(coords, coup), full_output=True, disp=False)
        if score<low[0]:
            low[0] = score
            low[1] = list(parms)
    low.append(INTAXMat(coords, low[1][0], low[1][1:]))
    return low

def INTAXMat(X, rot=0, axis=[1,0,0]):
    Y = X.dot(rotaxis(rot*d2r, axis))
    a = np.zeros((N,N))
    m=0
    while m<N:
        n = m
        while n<N:
            r_mn = ( (Y[m,0]-Y[n,0])**2 + (Y[m,1]-Y[n,1])**2 + (Y[m,2]-Y[n,2])**2 )**0.5
            if r_mn!=0:
                cos_theta = (Y[m,2]-Y[n,2])/r_mn
                a[m,n] = 1E-7*gamma[m]*gamma[n]*hbar*(3*cos_theta**2 - 1) / (r_mn*1E-10)**3
                a[n,m] = a[m,n]
            n+=1
        m+=1
    [a_max, N_max] = [np.max(abs(a[C.v['xind'],:])), np.argmax(abs(a[C.v['xind'],:]))]
    return a, N_max, a_max

def pade(A, q=4):
    
    tick = 0
    Apr = copy(A)
    
    # Step 1: If ||A||_1 < 0.5, proceed to step 4
    a1n = np.max(np.sum(abs(Apr), axis=0))
    if a1n>0.5:
        # Step 2: Determine minimum integer m such that ||A||_1 / m < 0.5
        # - find m = 2^n; divide original matrix by m, do n number of matrix multiplication in step 6
        n=1
        m=2
        while (a1n/m)>0.5:
            n+=1
            m = 2**n
        m = 2**n
        # Step 3: Compute A' = A/m
        Apr /= m
        tick = 1
    
    # Step 4: compute Nq and Dq
    Nq = np.zeros((sz, sz), dtp)
    Dq = np.zeros((sz, sz), dtp)
    a1 = np.eye(sz, dtype=dtp)
    a2 = np.eye(sz, dtype=dtp)
    for j in range(q+1):
        mul = ( fact(2*q-j)*fact(q) ) / ( fact(2*q)*fact(j)*fact(q-j) )
        if j>0:
            a1 = a1.dot(Apr)
            a2 = a2.dot(-1*Apr)
        Nq += mul*a1
        Dq += mul*a2
    
    # Step 5: Compute e^A', which is denoted Rq
    Rq = la.inv(Dq).dot(Nq)
    
    # Step 6: If steps 2 and 3 were undertaken, compute e^A = (e^A')^n
    if tick==1:
        for _ in range(int(n)):
            Rq = Rq.dot(Rq)
    
    return Rq

def cpinv(A):
    R = np.zeros(A.shape, A.dtype)
    Ar = A.real
    Ai = A.imag
    r0 = la.inv(Ar).dot(Ai)
    y00 = la.inv(Ai.dot(r0) + Ar)
    y10 = -1*r0.dot(y00)
    R.real = y00
    R.imag = y10
    return R

def propg(dev, steps, sx, rho, exD1, exD2):
    g = np.zeros((steps), dtype=dtp)
    exD1i = dagger(exD1)
    exD2i = dagger(exD2)
    
    hold   = np.empty((sz,sz), dtype=dtp)
    
    flag = 0
    for n1 in range(steps):
        hold = np.dot(sx, rho)
        g[n1] = np.trace(hold)
        if flag==0:
            hold = np.dot(rho, exD1i)
            rho = np.dot(exD1 , hold)
            flag = 1
        else:
            hold = np.dot(rho, exD2i)
            rho = np.dot(exD2 , hold)
            flag = 0
    
    return g

def propgeo(dev, steps, Det, rho, exD1, Q):
    g = np.zeros((steps), dtype=dtp)
    
    exD1i = dagger(exD1).astype(dtp)
    
#        print("Running on device %d (%3.1f%% full)"%(cp.cuda.device.get_device_id(), (mp.used_bytes()/mp.total_bytes())*100))
    for n1 in range(steps):
        g[n1] = np.trace( Det.dot(rho) )
        rho = exD1.dot(rho).dot(exD1i)
        
    Q.put(g)

def expmdev(dev, A, q):
    
    SZ,_ = A.shape
    
    tick = 0
    Apr = A

    # Step 1: If ||A||_1 < 0.5, proceed to step 4
    a1n = np.max(np.sum(abs(A), axis=0))
    if a1n>0.5:
        # Step 2: Determine minimum integer m such that ||A||_1 / m < 0.5
        # - find m = 2^n; divide original matrix by m, do n number of matrix multiplication in step 6
        n=1
        m=2
        while (a1n/m)>0.5:
            n+=1
            m = 2**n
        m = 2**n
        # Step 3: Compute A' = A/m
        Apr = (Apr/m).astype(dtp)
        tick = 1

    Ad = np.array(Apr, dtype=dtp)
    hold1 = np.zeros((SZ,SZ), dtype=dtp)
#        print("Running on device %d (%3.1f%% full)"%(cp.cuda.device.get_device_id(), (mp.used_bytes()/mp.total_bytes())*100))
    for i in range(q+1):
        hold2 = np.eye(SZ, dtype=dtp)
        if i>0:
            for j in range(i):
                hold2 = hold2.dot(Ad)
        hold2 /= fact(i)
        hold1 += hold2
    
    if tick==1:
        for _ in range(int(n)):
           hold1 = hold1.dot(hold1)
        
        
    
    return hold1

def expmdev2(dev, A, q, Q):
    M_cycle, N_cycle = cur.phase[:2].shape#phase1.shape
    exD = np.eye(sz, dtype=dtp)
    
    for k in range(N_cycle):
        tick = 0
        Apr = A[k].todense().astype(dtp)
    
        # Step 1: If ||A||_1 < 0.5, proceed to step 4
        a1n = np.max(np.sum(abs(Apr), axis=0))
        if a1n>0.5:
            # Step 2: Determine minimum integer m such that ||A||_1 / m < 0.5
            # - find m = 2^n; divide original matrix by m, do n number of matrix multiplication in step 6
            n=1
            m=2
            while (a1n/m)>0.5:
                n+=1
                m = 2**n
            m = 2**n
            # Step 3: Compute A' = A/m
            Apr /= m
            tick = 1
        
        Ad = np.array(Apr, dtype=dtp)
        
        hold1 = np.zeros((sz,sz), dtype=dtp)
        for i in range(q+1):
            hold2 = np.eye(sz, dtype=dtp)
            if i>0:
                for j in range(i):
                    hold2 = hold2.dot(Ad)
            hold2 /= fact(i)
            hold1 += hold2
        
        if tick==1:
            for _ in range(int(n)):
               hold1 = hold1.dot(hold1)
        
#            if k==0:
#                print("     Running on device %d (%3.1f%% full)"%(cp.cuda.device.get_device_id(), (mp.used_bytes()/mp.total_bytes())*100))
        
        exD = hold1.dot(exD)
    
    Q.put(exD)

def pretot():
    
    sgm = 2*pi*W0*1E-6*sGm
        
    # SETTING UP AND CALCULATION OF THE INTERACTION MATRICES
    Iz = spgen(0.5*np.array([[1,0],[0,-1]], dtype=dtp))
    Iy = spgen((1/(2*i))*np.array([[0,1],[-1,0]], dtype=dtp))
    Ix = spgen(0.5*np.array([[0,1],[1,0]], dtype=dtp))
    Ip = spgen(np.array([[0,1],[0,0]], dtype=dtp))
    Im = dagger(Ip).tocsc()
    
    Sx = spgen((sz,sz), dtype=dtp);Sy = spgen((sz,sz), dtype=dtp);Sz = spgen((sz,sz), dtype=dtp)
    Hsgm = spgen((sz,sz), dtype=dtp)
    Ixtot = spgen((sz, sz), dtype=dtp);Iytot = spgen((sz, sz), dtype=dtp);Iztot = spgen((sz, sz), dtype=dtp)
    Hdiplike = spgen((sz,sz), dtype=dtp);Hdipunlike = spgen((sz,sz), dtype=dtp);Hdiptot = spgen((sz, sz), dtype=dtp)
    
    m=0
    while m<N:
        Iyk = operator(Iy, m, N)
        Izk = operator(Iz, m, N)
        Ixk = operator(Ix, m, N)
        if gamma[m]==C.v['gamI']: # Proton
            Ixtot += Ixk
            Iytot += Iyk
            Iztot += Izk
        elif gamma[m]==C.v['gamS']:
            Sx += Ixk
            Sy += Iyk
            Sz += Izk
        Hsgm += sgm[m]*Izk
        n=m+1
        while n<N:
            if gamma[m]==gamma[n]: # Homonuclear
                Hjk = operator2(Iz,Iz,m,n,N)-0.25*(operator2(Ip,Im,m,n,N)+operator2(Im,Ip,m,n,N))
                Hdiplike += a[m,n]*Hjk
            else: # Heteronuclear
                Hjk = operator2(Iz, Iz, m, n, N)
                Hdipunlike += a[m,n]*Hjk
            Hdiptot += a[m,n]*Hjk
            
            n+=1
        m+=1
    
    Sd = operator(Ix, C.v['xind'], N)
    
    return Hdiptot, Hsgm, Sx, Sy, Ixtot, Iytot, Iztot, Sd

def PHASE(Hdiptot, phase, t, w1):
    Hall = []
    M_cycle, N_cycle = phase.shape
    for m_cycle in range(N_cycle):
        
        p1 = phase[0,m_cycle]
        p2 = phase[1,m_cycle]
        HrfX = np.cos((pi/2)*p1)*Sx + np.sin((pi/2)*p1)*Sy
        HrfH = np.cos((pi/2)*p2)*Ixtot + np.sin((pi/2)*p2)*Iytot
        
        H_cycle = (dW[m_cycle]*Iztot) + Hsgm + Hdiptot + (w1[0,m_cycle]*w_rf*HrfX) + (w1[1,m_cycle]*w_rf*HrfH)
        Hall.append(-i*t90*t[m_cycle]*H_cycle)
        
    return Hall

def getXD(Hall1, Hall2):
    outq1 = queue.Queue();outq2 = queue.Queue()
    t1 = threading.Thread(target=expmdev2, args=(0, Hall1, 7, outq1))
    t2 = threading.Thread(target=expmdev2, args=(1, Hall2, 7, outq2))
    t1.start();t2.start()
    t1.join();t2.join()
    exDtot1 = outq1.get()
    exDtot2 = outq2.get()
    return exDtot1, exDtot2

def mostprom(freq, ft, start, end, MAXW):
    stepsz = freq[1]-freq[0]
    MIN = min(freq)
    maxw = int(np.ceil(MAXW/stepsz))
    
    # This could happen with large scf
    if MIN>start:
        return 0,0,1
    
    ind11 = int(np.floor((start-MIN)/stepsz))
    ind12 = int(np.ceil((end-MIN)/stepsz))
    inds1,dic1 = ss.find_peaks(ft[ind11:ind12], width=(0, maxw))
    if len(inds1)==0:
        pkh = 1E-10
        pkw = 1E10
        pkf = abs(end-start)/2
    else:
        ind1 = inds1[np.argmax(dic1['prominences'])]+ind11
        pkw = stepsz*ss.peak_widths(ft, [ind1], rel_height=0.5)[0][0]
        pkh = ft[ind1]
        pkf = freq[ind1]
    
    return pkh, pkw, pkf

def allinone(Hdip, sf, ef):
    Hall1 = PHASE(Hdip, cur.phase[:2], cur.phase[4], cur.phase[5:])
    Hall2 = PHASE(Hdip, cur.phase[2:4], cur.phase[4], cur.phase[5:])    

    exDtot1, exDtot2 = getXD(Hall1, Hall2)  
    
    # Normalized initial density matrix
    rho01 = exDtot1.dot(rho00).dot(dagger(exDtot1))
    exDe = exDtot2.dot(exDtot1)
    exDo = exDtot1.dot(exDtot2)
    
    outq1 = queue.Queue();outq2 = queue.Queue()
    t1 = threading.Thread(target=propgeo, args=(0, Nm1//2, rho_det, rho00, exDe, outq1))
    t2 = threading.Thread(target=propgeo, args=(1, Nm1//2, rho_det, rho01, exDo, outq2))
    t1.start();t2.start()
    t1.join();t2.join()
    
    Gee = np.zeros((Nm1), dtype=dtp)
    Gee[:Nm1:2] = outq1.get()
    Gee[1:Nm1:2] = outq2.get()
    
    Nzf = 1025 # zerofill size
    Gee[0] /= 2
    Gee = np.append(Gee, np.zeros((Nzf-Nm1)))
    
    Nt, = Gee.shape
    dwell = t90*sum(cur.phase[4])
    timedom = np.linspace(0, (Nt-1)*dwell, Nt)
    # Apodization
    G_conv = np.multiply(Gee,np.exp(-timedom*pi*dnu))
    FT = np.fft.fftshift(np.fft.fft(G_conv)).real
    FT *= (1E4*dwell if dcorr==1 else 1)
    FREQ = np.linspace(-1/2/(dwell*scf), 1/2/(dwell*scf), Nzf)
    # NORMALIZATION
#    spac = FREQ[1]-FREQ[0]
#    FT *= 1E4/(spac*sum(((FT[:-1:2])+(FT[1::2]))))
    
    if sf=='max': #edit
        sf = min(FREQ)+100
    p, pkw, pf = mostprom(FREQ, FT, sf, ef, 700)
#    if np.argmax(FT)==512:
#        p*=1E-10
    return p, pkw, pf# edit

def run3():
    
    pms=[];pws=[];pfs=[]
    for M in range(len(Hdips)):
        (pm, pw, pf) = allinone(Hdips[M], 'max', -1000)
        pm = (1E-10 if pm<0 else pm)
        pw *= abs(amaxs[M]/2/pi/pf)
        pms.append(pm)
        pws.append(pw)
        pfs.append(pf)
    
    fac = 1
    for M in range(ntargs-1):
        dif = ratios[M] - pfs[M]/pfs[M+1]
        fac *= np.exp(-(dif / z)**2)
    fac =  fac**(1/(ntargs-1))
    raw = np.prod(pms)**(1/ntargs)
    scor = -1*fac*raw
    
    width = np.prod(pws)**(1/ntargs)
    
    return scor, raw, width
    
def AccRej(arg):
    if arg==0:
        # Alternatively phase1 = copy(hold.phase[:2]); phase2 = copy(hold.phase[2:4]); t = copy(phase[4])
        if (sym[ind1, ind2]==0) | (sym[ind1, ind2]==1):
            cur.phase[ind1%2,     [ind2, -(ind2+1)]] = hold.phase[ind1%2,     [ind2, -(ind2+1)]]
            cur.phase[(ind1%2)+2, [ind2, -(ind2+1)]] = hold.phase[(ind1%2)+2, [ind2, -(ind2+1)]]
        elif (sym[ind1, ind2]==2) | (sym[ind1, ind2]==3):
            cur.phase[ind1, [ind2, -(ind2+1)]] = hold.phase[ind1, [ind2, -(ind2+1)]]
        elif (sym[ind1, ind2]==4) | (sym[ind1, ind2]==5):
            cur.phase[[(ind1%2), (ind1%2)+2], ind2] = hold.phase[[(ind1%2), (ind1%2)+2], ind2]
        else:
            cur.phase[ind1, ind2] = hold.phase[ind1, ind2]
    else:
        # Alternatively hold.phase = np.row_stack((phase1,phase2,t))
        if (sym[ind1, ind2]==0) | (sym[ind1, ind2]==1):
            hold.phase[ind1%2,     [ind2, -(ind2+1)]] = cur.phase[ind1%2,     [ind2, -(ind2+1)]]
            hold.phase[(ind1%2)+2, [ind2, -(ind2+1)]] = cur.phase[(ind1%2)+2, [ind2, -(ind2+1)]]
        elif (sym[ind1, ind2]==2) | (sym[ind1, ind2]==3):
            hold.phase[ind1, [ind2, -(ind2+1)]] = cur.phase[ind1, [ind2, -(ind2+1)]]
        elif (sym[ind1, ind2]==4) | (sym[ind1, ind2]==5):
            hold.phase[[(ind1%2), (ind1%2)+2], ind2] = cur.phase[[(ind1%2), (ind1%2)+2], ind2]
        else:
            hold.phase[ind1, ind2] = cur.phase[ind1, ind2]

def fsps(arg):
    rnd = (np.random.uniform(RNG[ind1,ind2,0], RNG[ind1,ind2,1]) if ind1<5 else (1 if cur.phase[ind1,ind2]==0 else 0)) # change if w1 continuous
    rnd2 = ( (rnd-2) if (rnd>2) else (4+rnd-2) )
    if (arg==0) | (arg==1): # full (anti)symmetry
        rnd2 = rnd if arg==1 else rnd2
        cur.phase[(ind1%2), [ind2, -(ind2+1)]] = [rnd, rnd2]
        cur.phase[(ind1%2)+2, [ind2, -(ind2+1)]] = [rnd2, rnd]
    elif (arg==2) | (arg==3): # partial (anti)symmetry between mirrored subdwells
        rnd2 = rnd if arg==3 else rnd2
        cur.phase[ind1, [ind2, -(ind2+1)]] = [rnd, rnd2]
    elif (arg==4) | (arg==5): # partial (anti)symmetry between odd and even dwell
        rnd2 = rnd if arg==5 else rnd2
        cur.phase[[(ind1%2),(ind1%2)+2], ind2] = [rnd, rnd2]
    else: # no symmetry
        cur.phase[ind1, ind2] = rnd

def numln(ran):
    if ran < C.v['numln'][0]:
        ind = np.random.randint(4)
        vtyp = 0
    elif ran < C.v['numln'][1]:
        ind = 4
        vtyp = 1
    else:
        ind = 5+np.random.randint(2)
        vtyp = 2
    return ind, vtyp

def sendmsg():
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    
    email_sender = 'jlapin@ncsu.edu'
    email_receiver = 'jlapin@ncsu.edu'
    subject = 'Simulation'
    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    
    body = 'Runtime = %f'%(time()-START)
    msg.attach(MIMEText(body, 'plain'))
    
    attachment = open(C.v['fopath']+"report.txt", 'r')
    part = MIMEBase('application', 'octet_stream')
    part.set_payload(attachment.read())
    attachment.close()
    part.add_header('Content-Disposition', "attachment; filename = %s"%('report.txt'))
    
    msg.attach(part)
    text = msg.as_string()
    
    connection = smtplib.SMTP('smtp.gmail.com', 587)
    connection.starttls()
    connection.login(email_sender, 'ktckbpspgcldjtfe')
    connection.sendmail(email_sender, email_receiver, text)
    connection.quit()
rep = []
dtp = 'complex64'
C = config()
if C.v['foconf']==True:
    with open(C.v['fopath']+"configfile",'w') as f:
        f.write(C.v['text'])
X = C.X#X[inds]
N = C.N#len(inds)
sz = 2**N
rep.append("N = %d\n"%(N))
print(rep[-1])

# Interaction parameters for NH system
gamma = C.gamma

############################

# PHASES, OFFSETS, AMPLITUDES, AND DURATIONS FOR THE PULSE SEQUENCE
scf = 1.0 # Scaling factor relative to the apparent dwell
L = C.v['L'] # Number of subdwells
dW = np.zeros((L), dtype='float32') # proton offsets

# Score objects
cur = scobj()
hold = scobj()
low = scobj()
raw = scobj()
wSO = scobj()

# rf amplitude
w_rf = 2*pi*C.v['wrf']
t90 = pi/(2*w_rf)

# Miscellaneous simulation parameters
Nm1 = C.v['Nm1']   # number of data points
W0 = C.v['W0']  # carrier frequency
SGM = C.v['sgmstd']    # sigma coefficient
dnu = np.float32(C.v['dnu'])     # linebroadening, Hz
z = C.v['z'] # ratio penalty coefficient
fus = C.v['fus'] # uphill facilitator
dcorr = 1   # dwell intensity correction

# Where (or in what ratios) to expect the signals
ntargs = len(C.v['Hhml'])
ratios=[]
for m in range(ntargs-1):
    ratios.append(C.v['Hhml'][m]/C.v['Hhml'][m+1])

difs=[];rots=[];axes=[];coords=[];Nmaxs=[];amaxs=[]
for m,n in enumerate(C.v['Hhml']):
    out = findrots(X, n)
    
    difs.append(out[0])
    rots.append(out[1][0])
    axes.append(out[1][1:])
    coords.append(out[-1][0])
    Nmaxs.append(out[-1][1])
    amaxs.append(out[-1][-1])
    
    rep.append("Target %d: %f %f %f %f"%(m, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2])) # need good precision to reproduce results
    print("Coupling Target %d: %5d Hz --> Found %5d Hz at %6.2f rotation about [%5.2f %5.2f %5.2f] axis"%(
            m+1, n, amaxs[-1]/2/pi, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2]))
if C.v['foreport']==True:
    with open(C.v['fopath']+'report.txt', 'w') as f:
        f.write("\n".join(rep))
print()

# Simulation search restrictions, parameters, etc.
dim1 = L
sym = C.sym[:,:dim1]
RNG = np.zeros((7, dim1, 2)) # Sampling range
RNG[:,:,0] = C.RNG[:7,:dim1]
RNG[:,:,1] = C.RNG[7:,:dim1]
active = RNG[:,:,0]!=RNG[:,:,1]

# Refinement or new optimization
new = C.v['new']
if new==False:
    # Grab all csv files in the "Loop/" directory
    files = [m for m in os.listdir(C.v['firefpath']) if m[-3:]=='csv']
    runs = len(files)
else:
    runs = C.v['runs']
    files = []
pms = np.zeros((runs, 4));pms[:,0] = np.arange(runs);allsgm = np.zeros((runs,N))

####################
## Start the MCSA ##
####################
START = time()
for l in range(runs):
    if new==False:
            cur.phase = np.loadtxt(C.v['firefpath']+"%s"%(files[l]), delimiter=",", skiprows=1)
            # setting the random RNG to the pulse sequence to be refined
            if C.v['refrng'][0]>=0:
                RNG[:,:,0] = cur.phase;RNG[:,:,1] = cur.phase
                RNG[:4,:,0] -= C.v['refrng'][0];RNG[:4,:,1] += C.v['refrng'][0]
                RNG[4,:,0] -= C.v['refrng'][1];RNG[4,:,1] += C.v['refrng'][1]
                # timings cannot go below 0
                for n,m in enumerate(RNG[4,:,0]):
                    RNG[4,n,0] = (0 if m<0 else m)
    else:
        files.append("Plow%d.csv"%(l))
        for m in range(5):
            for n in range(L):
                cur.phase[m,n] = np.random.uniform(RNG[m,n%dim1,0], RNG[m,n%dim1,1])
                cur.phase[5,n] = C.v['SwrfA'][n]
                cur.phase[6,n] = C.v['IwrfA'][n]
    
    # Setup hamiltonians to be used for various sgm, amax
    if C.v['sgmr']==True:
        sGm = np.random.normal(0, SGM, (N,))
    else:
        sGm = SGM*C.v['sigs'][:N]
    allsgm[l] = sGm
    
    Hdips=[]
    for m in range(len(coords)):
        a = coords[m]
        Hdip, Hsgm, Sx, Sy, Ixtot, Iytot, Iztot, Sd = pretot()
        Hdips.append(Hdip)
    rho_det = Sd.todense()
    rho00 = ((Sd-Ixtot)/(2**(N-1))).todense()
    
    # Initialize score objects
    cur.pm,cur.pmraw,cur.wid = run3()
    hold.copyfrom(cur, phase=True)
    low.copyfrom(cur, phase=True) # best real score
    wSO.copyfrom(cur, phase=True);wSO.wid = 1E10 # lowest width
    raw.copyfrom(cur, phase=True) # best raw score
    
    Temp = C.v['temp'] # 0 for schedule; 1 for adaptive
    incr  = C.v['incr']
    steps = C.v['steps']
    WW = (np.linspace(C.v['Ws'][0], C.v['Ws'][1], incr, dtype='float32') if Temp==0 else C.v['Wab']);W = WW
    scors = np.zeros((incr, 2));hs = np.zeros((7,dim1));mags = np.zeros((2,7,dim1));mags[1]=1E-10
    for m in range(incr): # Temperature loop
        if Temp==0:
            W = WW[m]
        print("m = %d; l = %d (%s)\nW = %f"%(m,l,files[l],W))
        accepth = 0
        acceptl = 0
        reject = 0
        desum = 0
        
        for n in range(steps): # random steps loop
#            sys.stdout.write("\r%d/%d"%(n,steps))
            # Choose random index (ind1, ind2) - This code is verbose because of how the variables relate
            # - while loop prevents moves on variables intended to be constant (a la RNG.txt)
            PASS=0
            while PASS==0:
                ind1,vtyp = numln( np.random.randint(C.v['numln'][-1]) ) # phase or timing
                ind2 = np.random.randint(0, dim1) # which subdwell
                if active[ind1, ind2]:
                    PASS=1
            # If a phase was chosen and its channel is off, find another subdwell
            if (ind1<4) & (sum(cur.phase[(ind1%2)+5])!=0):
                cnt=0
                while (cur.phase[(ind1%2)+5,ind2]==0) & (cnt<50):
                    ind2 = np.random.randint(0, dim1)
                    cnt+=1
            
            # Impose symmetry
            fsps(sym[ind1, ind2]) # function automatically operates on cur
            # cur.phase[5:] = np.round(cur.phase[5:])
            
            # Score new sequence
            cur.pm,cur.pmraw,cur.wid = run3()
            dE = cur.pm - hold.pm
            
            # HS statistics
            if (C.v['foHS']==True) & (dE!=0):
                mags[0,ind1,ind2]+=abs(dE);mags[1,ind1,ind2]+=1
            # Facilitate uphill steps
            if dE>0:
                dE*=fus[vtyp]
            # Adaptive temperature variable
            if Temp==1:
                desum+=(dE if dE>0 else 0)
            # Record keeping
            if cur.pm < low.pm: # Lowest score
                low.copyfrom(cur, phase=True)
            if (cur.pm < C.v['Wathr']) & (cur.pmraw > raw.pmraw): # lowest raw score
                raw.copyfrom(cur, phase=True)
            if (cur.pm < 2*C.v['Wathr']) & (cur.wid < wSO.wid): # lowest width
                wSO.copyfrom(cur, phase=True)
            
            # Update cur and hold states
            if dE<0: # if less -> accept
                hold.copyfrom(cur)
                AccRej(1)
                acceptl += 1
            elif dE==0: # if equal do a coin flip
                if np.random.rand()>0.5: # accept coin flip
                    hold.copyfrom(cur)
                    AccRej(1)
                    accepth += 1
                    if C.v['foHS']==True:
                        hs[ind1,ind2] += 1
                else: # reject coin flip
                    AccRej(0)
                    reject += 1
            else: # apply metropolis criterion
                if np.exp(-dE*W)>np.random.rand(): # accept metropolis
                    hold.copyfrom(cur)
                    AccRej(1)
                    accepth += 1
                    if C.v['foHS']==True:
                        hs[ind1,ind2] += 1
                else: # reject metropolis
                    AccRej(0)
                    reject += 1
            
        # End of temp increment
        # print status
        print("   Steps accepted (+|=): %2d"%(accepth))
        print("   Steps accepted (-)  : %2d"%(acceptl))
        print("   Steps rejected      : %2d"%(reject))
        print("   Current score/raw/wid: %7.3f (%6.3f/%4d)"%(hold.pm, hold.pmraw, hold.wid))
        print("   Low score/raw/wid    : %7.3f (%6.3f/%4d)"%(low.pm, low.pmraw, low.wid))
        print("   Global low raw/wid   :         (%6.3f/%4d)\n"%(raw.pmraw, wSO.wid))
        # Save Plow after every increment
        if C.v['foPlow']==True:
            np.savetxt(C.v['fopath']+"%s"%(files[l]), low.phase, delimiter=",", 
                   header='%s; sgmstd=%d; w_rf=%.2f; dnu=%.0f; pm/raw/wid=%.3f/%.3f/%d'%(
                           os.path.split(sys.argv[0])[1],SGM, w_rf/2000/pi, dnu, low.pm, low.pmraw, low.wid))
        scors[m,0] = hold.pm;scors[m,1] = low.pm
        
        """
        update temperature
        - Logarithm must be tuned to accept a comfortable number of higher
           energy steps, but not so much so that it pops out of deep energy wells
        """
        if Temp==1:
            desum/=steps-acceptl
            W = (C.v['Wab'] if hold.pm>C.v['Wathr'] else -np.log(C.v['Wap']**-1)/desum)
        
    # End of run
    # Save scores for the run
    pms[l,1] = low.pm;pms[l,2] = low.pmraw;pms[l,3] = low.wid
    # Print out other lows or statistics
    if C.v['foPlowraw']==True: # low raw score parameters
        if raw.pmraw!=low.pmraw:
            np.savetxt(C.v['fopath']+'raw%s'%(files[l]), raw.phase, delimiter=",", 
                       header='%s; sgmstd=%d; w_rf=%.2f; dnu=%.0f; pm/raw/wid=%.3f/%.3f/%d'%(
                               os.path.split(sys.argv[0])[1],SGM, w_rf/2000/pi, dnu, raw.pm, raw.pmraw, raw.wid))
    if C.v['fowid']==True: # low width parameters
        if wSO.wid!=low.wid:
            np.savetxt(C.v['fopath']+"wid%s"%(files[l]), wSO.phase, delimiter=",", 
                       header='%s; sgmstd=%d; w_rf=%.2f; dnu=%.0f; pm/raw/wid=%.3f/%.3f/%d'%(
                               os.path.split(sys.argv[0])[1],SGM, w_rf/2000/pi, dnu, wSO.pm, wSO.pmraw, wSO.wid))
    if C.v['foHS']==True: # Uphill statistics: magnitudes and # of high steps
        MAG = np.divide(mags[0],mags[1])
        Hs = open("hs.txt"%(l), 'a')
        Hs.write("%s\n"%(files[l]))
        Hs.write("#: "+" ".join(["%6d"%(M) for M in np.arange(1,MAG.shape[1]+1,1)])+"\n")
        for M,n in zip([MAG,hs],["%6.3f","%6d"]):
            for o in range(M.shape[0]):
                Hs.write("%d: "%(o)+" ".join([n%(p) for p in M[o]])+"\n")
        Hs.close()
# End of program
if C.v['shutdown']==True: # shutdown computer
    os.system("shutdown /s /t 60")
if C.v['foRNG']==True: # save random number ranges for variables
    np.savetxt(C.v['fopath']+"RNG.txt", np.row_stack((RNG[:,:,0], RNG[:,:,1])), fmt='%.1f', delimiter=",")
if bool(int(C.v['fosgm']))==True: # save sigmas for reproduceability
    np.savetxt(C.v['fopath']+"sgm.txt", allsgm, fmt='%.3f', delimiter=",")
if bool(int(C.v['foreport']))==True: # save report with scores and misc. info
    rep.append("Total runtime: %.2f"%(time()-START))
    header="\n".join(rep)+"\n"
    for m in files:
        header+=m+" "
    np.savetxt(C.v['fopath']+"report.txt", pms, fmt='%5d %7.2f %7.2f %7.0f', delimiter=" ", header=header)
#np.savetxt("C:/Users/joell/Documents/Python/Nevzorov/SPIN/scors.csv", scors, fmt='%f', delimiter=",")
sendmsg() # send email of report
print("\nTotal runtime: %.2f"%(time()-START))