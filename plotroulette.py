
import numpy as np
import scipy.signal as ss
import scipy.sparse as sp
import scipy.linalg as la
from scipy.optimize import fmin
from copy import copy
import cupy as cp
#from time import time
import threading
import queue
from math import factorial as fact
import matplotlib.pyplot as plt
import os
import sys
from PSLib import config,scobj
os.environ['CUDA_PATH']='C:/Users/joell/Anaconda3/pkgs/cudatoolkit-9.0-1/'
spgen = sp.csc_matrix
cos = np.cos
sin = np.sin
pi = np.pi
d2r = pi/180
i = np.complex64(np.complex(0,1))
hbar=1.055e-34
   
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

def genscf(obs, exp=[2500,5000,10000]):
    obs = np.sort(obs)
    exp = np.sort(exp)
    from scipy.optimize import curve_fit
    def f(x, m):
        return m*x
    val,cov = curve_fit(f, exp, obs)
    return val[0]

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
    
    with cp.cuda.Device(dev):
        gd     = cp.array(g, dtype=dtp)
        sxd    = cp.array(sx, dtype=dtp)
        rhod   = cp.array(rho, dtype=dtp)
        exD1d  = cp.array(exD1, dtype=dtp)
        exD1id = cp.array(exD1i, dtype=dtp)
        exD2d  = cp.array(exD2, dtype=dtp)
        exD2id = cp.array(exD2i, dtype=dtp)
        hold   = cp.array(np.empty((sz,sz), dtype=dtp))
    cp.cuda.Device(dev).use()
    
    flag = 0
    for n1 in range(steps):
        hold = cp.dot(sxd, rhod)
        gd[n1] = cp.trace(hold)
        if flag==0:
            hold = cp.dot(rhod, exD1id)
            rhod = cp.dot(exD1d , hold)
            flag = 1
        else:
            hold = cp.dot(rhod, exD2id)
            rhod = cp.dot(exD2d , hold)
            flag = 0
    
    g = cp.asnumpy(gd)
    return g

def propgeo(dev, steps, Det, rho, exD1, Q):
    g = np.zeros((steps), dtype=dtp)
    
    with cp.cuda.Device(dev):
#        mp = cp.get_default_memory_pool()
        
        Detd    = cp.array(Det, dtype=dtp)
        rhod   = cp.array(rho, dtype=dtp)
        exD1d  = cp.array(exD1, dtype=dtp)
        exD1id = cp.array(dagger(exD1), dtype=dtp)
        
#        print("Running on device %d (%3.1f%% full)"%(cp.cuda.device.get_device_id(), (mp.used_bytes()/mp.total_bytes())*100))
        for n1 in range(steps):
            g[n1] = cp.asnumpy( cp.trace( rhod.dot(Detd) ) )
            rhod = exD1d.dot(rhod).dot(exD1id)
        
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

    with cp.cuda.Device(dev):
#        mp = cp.get_default_memory_pool()
        
        Ad = cp.array(Apr, dtype=dtp)
        hold1 = cp.zeros((SZ,SZ), dtype=dtp)
#        print("Running on device %d (%3.1f%% full)"%(cp.cuda.device.get_device_id(), (mp.used_bytes()/mp.total_bytes())*100))
        for i in range(q+1):
            hold2 = cp.eye(SZ, dtype=dtp)
            if i>0:
                for j in range(i):
                    hold2 = hold2.dot(Ad)
            hold2 /= fact(i)
            hold1 += hold2
        
        if tick==1:
            for _ in range(int(n)):
                hold1 = hold1.dot(hold1)
        
        
    
    return cp.asnumpy(hold1)

def expmdev2(dev, A, q, Q):
    M_cycle, N_cycle = cur.phase[:2].shape#phase1.shape
    exD = np.eye(sz, dtype=dtp)
    with cp.cuda.Device(dev):
#        mp = cp.get_default_memory_pool()
        
        exDd = cp.eye(sz, dtype=dtp)
        
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
            
            Ad = cp.array(Apr, dtype=dtp)
            
            hold1 = cp.zeros((sz,sz), dtype=dtp)
            for i in range(q+1):
                hold2 = cp.eye(sz, dtype=dtp)
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
            
            exDd = hold1.dot(exDd)
    
    exD = cp.asnumpy(exDd)
    
    Q.put(exD)

def pretot():
    # Random chemical shifts of protons
    cs_distr = [0]
    for m in 10*np.random.uniform(-1,1,N-1):
        cs_distr.append(m)
    cs_distr = np.array(cs_distr)
    
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
                if gamma[m]==C.v['gamS']: # code for quality control -> set fac to 0
                    fac = 1
                else:
                    fac = 1
                Hdiplike += a[m,n]*Hjk
            else: # Heteronuclear
                fac = 1
                Hjk = operator2(Iz, Iz, m, n, N)
                Hdipunlike += a[m,n]*Hjk
            Hdiptot += fac*a[m,n]*Hjk
            
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

def mostprom(freq, ft, start, end, MINW=0, MAXW=700):
    stepsz = freq[1]-freq[0]
    MIN = min(freq)
    minw = int(np.floor(MINW/stepsz))
    maxw = int(np.ceil(MAXW/stepsz))
    
    # This could happen with large scf
    if MIN>start:
        return 0,0,1
    
    ind11 = int(np.floor((start-MIN)/stepsz))
    ind12 = int(np.ceil((end-MIN)/stepsz))
    ninds1,ndic1 = ss.find_peaks(ft[ind11:ind12], width=(minw, maxw)) # The "n" in front refers to new indices
    if len(ninds1)==0:
        pkh = pkrat = 1E-10
        pkw = 1E10
        pkf = abs(end-start)/2
    else:
        top2 = ndic1['prominences'].argsort()[-2:]
        ind1 = ninds1[top2]+ind11 # adding ind11 converts back to old indices
        
        pkh = ft[ind1]
        pkw = stepsz*ss.peak_widths(ft, [ind1[-1]], rel_height=C.v['pwl'])[0][0]
        if ndic1['prominences'][top2[-1]]<C.v['ipent']:
            pkw*=(C.v['ipent']/ndic1['prominences'][top2[-1]])**C.v['ipene']
        # pkh = ndic1['prominences'][top2]
        pkrat = pkh[-1]#/(1+0.0*pkh[-2])
        pkf = freq[ind1[-1]]
    
    return pkrat, pkw, pkf

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
    
    if sf=='max':
        sf = min(FREQ)-0.01*FREQ[0]
    p, pkw, pf = mostprom(FREQ, FT, sf, ef, MINW=150, MAXW=1500)
    
    return FT, FREQ, p, pkw, pf
    
dtp = 'complex64'

C = config()

Files = []
for m in os.listdir(C.v['fopath']):
    if m[-4:]=='.csv':
        Files.append(m)
if len(Files)==0:
    sys.exit('\n0 csv files found')
print('\n%d csv files found. Choose one by number, then press enter\n'%(len(Files)))
for m,n in enumerate(Files):
    print("%d: %s"%(m,n))
M = int(input(">>> "))

X = C.X
N = C.N
sz = 2**N

# Interaction parameters for NH system
gamma = C.gamma
############################

# PHASES, OFFSETS, AMPLITUDES, AND DURATIONS FOR THE PULSE SEQUENCE
scf = 1.0 # Scaling factor relative to the apparent dwell
L = C.v['L'] # number of subdwells

cur = scobj(L)
cur.phase = np.loadtxt(C.v['fopath']+'%s'%(Files[M]), skiprows=1, delimiter=',')
L = np.shape(cur.phase)[1]
dW = np.zeros((L), dtype='float32') # proton offsets

# rf amplitude
w_rf = 2*pi*C.v['wrf']
t90 = pi/(2*w_rf)

# Useful experimental parameters
Nm1 = C.v['Nm1']   # number of data points
W0 = C.v['W0']  # carrier frequency
SGM = C.v['sgmstd']    # sigma coefficient
dnu = np.float32(C.v['dnu'])     # linebroadening, Hz
z = C.v['z']
dcorr = 1   # dwell intensity correction

# Where (or in what ratios) to expect the signals
ntargs = len(C.v['Hhml'])
ratios=[]
for m in range(ntargs-1):
    ratios.append(C.v['Hhml'][m]/C.v['Hhml'][m+1])

# Reading in crystal rotations or generating new ones
new = 0 if len(sys.argv)==1 else int(sys.argv[1]) # user input
rots=[];axes=[];coords=[];Nmaxs=[];amaxs=[]
if (new==0) & ('report.txt' in os.listdir(C.v['fopath'])):
    with open(C.v['fopath']+'report.txt') as f:
        f.readline()
        for m,n in enumerate(C.v['Hhml']):
            parms = [float(n) for n in f.readline().split()[-4:]]
            
            rots.append(parms[0])
            axes.append(parms[1:])
            out = INTAXMat(X, rot=rots[-1], axis=axes[-1])
            coords.append(out[0])
            Nmaxs.append(out[1])
            amaxs.append(out[-1])
            print("Coupling Target %d: %5d Hz --> Read %5d Hz at %6.2f rotation about [%5.2f %5.2f %5.2f] axis"%(
                    m+1, n, amaxs[-1]/2/pi, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2]))
else:
    difs=[]
    for m,n in enumerate(C.v['Hhml']):
        out = findrots(X, n)
        
        difs.append(out[0])
        rots.append(out[1][0])
        axes.append(out[1][1:])
        coords.append(out[-1][0])
        Nmaxs.append(out[-1][1])
        amaxs.append(out[-1][-1])
        
        print("Coupling Target %d: %5d Hz --> Found %5d Hz at %6.2f rotation about [%5.2f %5.2f %5.2f] axis"%(
                m+1, n, amaxs[-1]/2/pi, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2]))

if C.v['sgmr']==0:
    sGm = C.v['sgmstd']*C.v['sigs']
else:
    if os.path.isfile(C.v['fopath']+'sgm.txt'):
        sgM = np.loadtxt(C.v['fopath']+'sgm.txt', delimiter=',').reshape(-1,N)
        sGm = sgM[M] if M<sgM.shape[0] else np.zeros((N,))
    else:
        print("\nsgm.txt not found -> defaulting to 0 for all sigmas")
        sGm = np.zeros((N,))

# Hdips and pretot
Hdips = []
for m in range(ntargs):
    a = coords[m]
    Hdip, Hsgm, Sx, Sy, Ixtot, Iytot, Iztot, Sd = pretot()
    Hdips.append(Hdip)
rho_det = Sd.todense()
rho00 = ((Sd-Ixtot)/(2**(N-1))).todense()

# Run allinone
FTs=[];FREQs=[];pms=[];pws=[];pws_real=[];pfs=[]
for m in range(ntargs):
    FT,FREQ,pm,pw,pf = allinone(Hdips[m], 'max', -1000)
    pm = (1E-10 if pm<0 else pm)
    pws_real.append(pw)
    pw *= abs(amaxs[m]/4/pi/pf)
    FTs.append(FT);FREQs.append(FREQ);pms.append(pm);pws.append(pw);pfs.append(pf)
# Factor
difs=[];facs=[]
fac=1#;C.v['z']=2.0
for m in range(ntargs-1):
    dif = (ratios[m]-(pfs[m]/pfs[m+1]))
    difs.append(dif)
    facs.append(C.v['om'][0]-C.v['om'][1]*np.exp(-( dif / C.v['z'] )**2 ))
    fac *= facs[-1]
fac = fac**(1/len(ratios)) if ntargs>1 else fac
rg = max(pfs)-min(pfs)
sd = np.std(pfs)
fac *= 2000/sd if rg<2000 else 1

width = np.prod(pws)**(1/ntargs) if ntargs>1 else np.prod(pws)
height = np.prod(pms)**(1/ntargs) if ntargs>1 else np.prod(pms)
scor = fac*(C.v['wts'][0]*width - C.v['wts'][1]*height)
scf = genscf([abs(m) for m in pfs], [m/4/pi for m in amaxs]) if ntargs>1 else pfs[0]/(amaxs[0]/4/pi)
pws_real=[m/scf for m in pws_real]
for m in FREQs:
    m/=scf

plt.close('all')
fig,ax = plt.subplots()
ax.set_xlabel(r'$\omega$ ($s^{-1}$)')
ax.set_ylabel('Amplitude')
for m in range(ntargs):
    ax.plot(FREQs[m], FTs[m], 'b', linewidth=0.5)
    ax.text(pfs[m]/scf, pms[m], '$\omega_{h}$=%d\n$I_{h}$=%.2f\n$w_{h}$=%d'%(pfs[m]/scf,pms[m],pws[m]),size=8)
ax.set_title('score=%.0f   width=%.0f   height=%.3f   fac=%.3f   scf=%.3f'%(scor,width,height,fac,scf), size=10)
ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]+2])
MIN = min([min(m) for m in FREQs]);MAX = max([max(m) for m in FREQs])
ax.set_xlim([MIN,MAX])
plt.show()
