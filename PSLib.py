# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 07:56:32 2020

@author: nevlab
"""
import numpy as np
import scipy.signal as ss
import scipy.sparse as sp
import scipy.linalg as la
from copy import copy
import sys
import os
spgen = sp.csc_matrix
cos = np.cos
sin = np.sin
pi = np.pi
d2r = np.pi/180
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
                'IwrfA':[1,1,1,1,1,1],
                
                'Ws':[1E-5,1E-1],
                'Wathr':4000,
                'Wap':0.05,
                'que':200,
                'Wae':1.0,
                
                'runs':1,
                'incr':200,
                'steps':100,
                'numln':[4,5,7],
                
                'fus':[1.0,1.0,1.0],
                'wts':[1,100],
                'om':[3,2],
                'z':0.834,
                'pkl':0.85,
                'ipent':1.0,
                'ipene':1.2,
                
                'wrf':58.14E3,
                'W0':500E6,
                'Nm1':256,
                'dnu':0
                
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
            print("QCWarning: S-channel wrf amplitude (SwrfA) must contain %d comma separated booleans. Defaulting to all-on"%(self.v['L']))
            self.v['SwrfA'] = [np.float32(1) for m in range(self.v['L'])]
        self.v['IwrfA'] = [np.float32(m) for m in self.v['IwrfA'].split(",")] # conversion
        if len(self.v['IwrfA'])!=self.v['L']:
            print("QCWarning: I-channel wrf amplitude (IwrfA) must contain %d comma separated booleans. Defaulting to all-on"%(self.v['L']))
            self.v['IwrfA'] = [np.float32(1) for m in range(self.v['L'])]
        if np.shape(self.RNG)[0]!=14:
            print("QCError: RNG input file must contain 14 rows (Top 7 -> floor vals; Bot 7 -> ceil vals)");tick+=1
        if np.shape(self.RNG)[1]!=self.v['L']:
            print("QCWarning: RNG input file should contain %d columns (same as L). Defaulting to RNG"%(self.v["L"]))
            self.RNG = np.zeros((14,self.v['L']));self.RNG[4]=0.5;self.RNG[7:12]=4;self.RNG[-2:]=1;self.RNG[5:7,[0,-1]]=1
            print(self.RNG)
        if np.shape(self.sym)[0]!=7:
            print("QCError: sym input file must contain 7 rows");tick+=1
        if np.shape(self.sym)[1]!=self.v['L']:
            print("QCWarning: sym input file should contain %d columns (same as L). Defaulting to sym"%(self.v["L"]))
            self.sym = 2*np.ones((7, self.v['L']));self.sym[-3:] = 3
            print(self.sym)
        
        # Temperature settings
        self.v['Ws'] = [np.float32(m) for m in self.v['Ws'].split(",")] # conversion
        if len(self.v['Ws'])!=2:
            print("QCError: Ws must contain 2 comma separated floats: T_start,T_end");tick+=1
        self.v['Wathr'] = np.float32(self.v['Wathr']) # conversion
        self.v['Wap'] = np.float32(self.v['Wap']) # conversion
        if (self.v['Wap']<0) | (self.v['Wap']>1):
            print("QCError: Adaptive temperature parameter Wap must be 0-1");tick+=1
        self.v['que'] = int(self.v['que']) # conversion
        self.v['Wae'] = float(self.v['Wae']) # conversion
        
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
        self.v['numln'] = [int(m) for m in self.v['numln'].split(",")] # conversion
        if sum(np.argsort(self.v['numln'])==[0,1,2])!=3:
            print("QCError: Number line cutoffs must be in ascending order");tick+=1
        
        # Score settings
        self.v['fus'] = [np.float32(m) for m in self.v['fus'].split(',')] # conversion
        self.v['wts'] = [np.float32(m) for m in self.v['wts'].split(',')] # conversion
        self.v['om'] = np.float32(self.v['om']) # conversion
        if self.v['om']<1:
            self.v['om'] = [0, -1] # conversion
            print('ATTN: om<1. Gaussian factor has range from 0-1; scores should be negative.')
        else:
            self.v['om'] = [self.v['om'], self.v['om']-1] # conversion
        self.v['z'] = np.float32(self.v['z']) # conversion
        if self.v['z']<0:
            print("QCError: Score penalty coefficient z must be >= 0");tick+=1
        self.v['pwl'] = np.float32(self.v['pwl']) # conversion
        if (self.v['pwl']>=1) | (self.v['pwl']<=0):
            print("QCError: Peak width level pwl must be a fraction from 0-1");tick+=1
        self.v['ipent'] = np.float32(self.v['ipent']) # conversion
        self.v['ipene'] = np.float32(self.v['ipene']) # conversion
        
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
        
        # sym.inp
        if (sum(0==self.sym[4])!=0) | (sum(1==self.sym[4])!=0):
            print("QCError: timing variable(s) assigned Full symmetry (0)\n");tick+=1
        if (sum(0==self.sym[5])!=0) | (sum(1==self.sym[6])!=0):
            print("QCError: rf amplitude variable(s) assigned Full symmetry (0)\n");tick+=1
        
        if tick>0:
            sys.exit("Please fix %d input error(s)"%(tick))

class scobj():
    def __init__(self, L):
        self.pm = 0
        self.pmraw = 1E10
        self.wid = 1E10
        self.fac = 1E10
        self.phase = np.zeros((7, L))
    
    def copyfrom(self, src, phase=False):
        self.pm = src.pm
        self.pmraw = src.pmraw
        self.wid = src.wid
        self.fac = src.fac
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

def findrots(coords, coup, axis):
    low = [1E10, -1, ()]
    for m in np.linspace(0,180,1000):
        (a, nmax, amax) = INTAXMat(coords, rot=m, axis=axis)
        dif = abs(coup-amax/2/pi)
        if dif<low[0]:
            low[:2] = [dif, m]
            low[-1] = (a, nmax, amax)
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