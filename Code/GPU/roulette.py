"""
120319: Changed the starting rho and propagation method. Should be standardized going forward.
        rho_n+1 = exDtot * rho_n * exDtot_i
        G[n+1] = trace( rho*rho_det )
033120: findrots now uses a simplex to find the angle and axis in order to 
        achieve the coupling target.
        Code modified to accept as many frequency targets as are listed for 
        configuration option "Hhml". Be cognizant of unintended bugs
040120: Put in code to deactivate variables via RNG.txt
        - setting the floor and ceiling ranges equal will prevent the variable (ind1, ind2)
            from being chosen
        Put in warning after temperature increment for entire channel's power being off
040220: Changed the scoring factor to product of gaussians
040320: HS info is now printed to 1 file
040420: Configuration file is now tab separated, in order to allow filepaths with spaces
041320: Appending sgm vectors to file sgm.txt during the program instead of the end
041520: Added if statements to handle situation for 1 resonance target, even though this is unlikely
041720: Added stdout timing splits for increments
        Added a penalty for peak artifacts in mostprom()
042620: Replaced desum with collection set to collect set number of high steps
        Adaptive temperature updates using new method to ensure the average hs acceptance probability
050220: Added 'wts' option to configuration. These values multiply to the width,height in run3
050320: Stdout now printing score/width/height/factor
050520: Added code to read in previous crystal orientations if new=0.
        - This effectively lets you continue a simulation (except for sgm)
        - Now the code is in a subroutine called "orientations()"
        Changed temperature so that Ws works in concert with adaptive variables
        - No more temp, Wab options
        Added configuration option 'om' that sets a regular or inverse gaussian for fac
        In allinone() changed the addition to the starting frequency "sf"
050620: rel_height in mostprom() is now a configuration option
        Singularity penalty added to run3()
        The minimum peak prominence and exponent are now configuration options
"""
import numpy as np
import scipy.signal as ss
import scipy.sparse as sp
import scipy.linalg as la
from scipy.optimize import fmin
from copy import copy
import cupy as cp
from time import time
import threading
import queue
from math import factorial as fact
import collections
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

def orientations():
    prints = []
    rots=[];axes=[];coords=[];Nmaxs=[];amaxs=[]
    if (C.v['new']==0) & ('report.txt' in os.listdir(C.v['firefpath'])):
        with open(C.v['firefpath']+'report.txt') as f:
            f.readline()
            for m,n in enumerate(C.v['Hhml']):
                parms = [float(n) for n in f.readline().split()[-4:]]
                rots.append(parms[0]);axes.append(parms[1:]);out = INTAXMat(X, rot=rots[-1], axis=axes[-1])
                coords.append(out[0]);Nmaxs.append(out[1]);amaxs.append(out[-1])
                prints.append("Target %d: %f %f %f %f"%(m, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2])) # need good precision to reproduce results
                print("Coupling Target %d: %5d Hz --> Read %5d Hz at %6.2f rotation about [%5.2f %5.2f %5.2f] axis"%(
                        m+1, n, amaxs[-1]/2/pi, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2]))
    else:
        difs=[]
        for m,n in enumerate(C.v['Hhml']):
            out = findrots(X, n)
            difs.append(out[0]);rots.append(out[1][0]);axes.append(out[1][1:])
            coords.append(out[-1][0]);Nmaxs.append(out[-1][1]);amaxs.append(out[-1][-1])
            prints.append("Target %d: %f %f %f %f"%(m, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2])) # need good precision to reproduce results
            print("Coupling Target %d: %5d Hz --> Found %5d Hz at %6.2f rotation about [%5.2f %5.2f %5.2f] axis"%(
                    m+1, n, amaxs[-1]/2/pi, rots[-1], axes[-1][0], axes[-1][1], axes[-1][2]))
    return coords, amaxs, prints

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
            g[n1] = cp.asnumpy( cp.trace( Detd.dot(rhod) ) )
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
    
    return p, pkw, pf

def run3():
    
    pms=[];pws=[];pfs=[]
    for M in range(len(Hdips)):
        (pm, pw, pf) = allinone(Hdips[M], 'max', -1000)
        pm = (1E-10 if pm<0 else pm)
        pw *= abs(amaxs[M]/4/pi/pf)
        pms.append(pm)
        pws.append(pw)
        pfs.append(pf)
    # rg = max(pfs)-min(pfs)
    # Scoring function
    fac = 1
    for M in range(ntargs-1):
        dif = ratios[M] - pfs[M]/pfs[M+1]
        fac *= C.v['om'][0]-C.v['om'][1]*np.exp(-(dif / z)**2)
    fac =  fac**(1/(ntargs-1)) if ntargs>1 else fac
    height = np.prod(pms)**(1/ntargs) if ntargs>1 else np.prod(pms)
    width = np.prod(pws)**(1/ntargs) if ntargs>1 else np.prod(pws)
    
    # fac *= 5 if rg<2000 else 1
    sd = np.std(pfs)
    fac *= 2000/sd if np.std(pfs)<2000 else 1
    scor = fac*(C.v['wts'][0]*width - C.v['wts'][1]*height)
    
    return scor, height, width, fac
    
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
    
    email_sender = 'send.email.com' # insert sender email here
    email_receiver = 'recipient.email.com' # inserd recipient email here
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
    connection.login(email_sender, 'password') # insert password here
    connection.sendmail(email_sender, email_receiver, text)
    connection.quit()

rep = []
dtp = 'complex64'
C = config()

# Notices
if C.v['new']==False:
    print("ATTN: \"new\" is False. Will read in parameters from following files:")
    files = [m for m in os.listdir(C.v['firefpath']) if m[-3:]=='csv']
    if len(files)==0:
        sys.exit("No csv files found in 'firefpath'")
    for m in files:
        sys.stdout.write("         -> %s\n"%(m))
print("ATTN: File output path is: %s"%(C.v['fopath']))
if C.v['shutdown']==True:
    print("ATTN: This program will shutdown the computer upon completion")
print()
# Configuration file
if C.v['foconf']==True:
    with open(C.v['fopath']+"configfile",'w') as f:
        f.write(C.v['text'])

############################

# Critical system parameters
X = C.X # Initial coordinates
gamma = C.gamma # Interaction parameters for the system
N = C.N # Number of spins
sz = 2**N # Matrix dimension size
L = C.v['L'] # Number of subdwells
dW = np.zeros((L), dtype='float32') # proton offsets
rep.append("Total spins: %d"%(N))
print(rep[-1]+"\n")

# Score objects
cur = scobj(L)
hold = scobj(L)
low = scobj(L)
raw = scobj(L)
wSO = scobj(L)

# Miscellaneous simulation parameters
w_rf = 2*pi*C.v['wrf']
t90 = pi/(2*w_rf)
Nm1 = C.v['Nm1']   # number of data points
W0 = C.v['W0']  # carrier frequency
SGM = C.v['sgmstd']    # sigma coefficient
dnu = np.float32(C.v['dnu'])     # linebroadening, Hz
z = C.v['z'] # ratio penalty coefficient
fus = C.v['fus'] # uphill facilitator
dcorr = 1   # dwell intensity correction
scf = 1.0 # Scaling factor relative to the apparent dwell

# Where (or in what ratios) to expect the signals
ntargs = len(C.v['Hhml'])
ratios=[]
for m in range(ntargs-1):
    ratios.append(C.v['Hhml'][m]/C.v['Hhml'][m+1])

# Crystal rotations: Either read in previous orientations or generate new ones
coords, amaxs, prints = orientations() # subroutine's code can be copied and pasted diredctly into main program
for m in prints:
    rep.append(m)
if C.v['foreport']==True:
    rptxt = "report2.txt" if os.path.isfile(C.v['fopath']+'report.txt') else "report.txt" # to prevent overwriting by mistake
    with open(C.v['fopath']+rptxt, 'w') as f:
        f.write("\n".join(rep))
print()

# Simulation search restrictions
dim1 = L
sym = C.sym[:,:dim1]
RNG = np.zeros((7, dim1, 2)) # Sampling range
RNG[:,:,0] = C.RNG[:7,:dim1]
RNG[:,:,1] = C.RNG[7:,:dim1]
active = RNG[:,:,0]!=RNG[:,:,1] # Active variables for randomization 

# Refinement or new optimization
new = C.v['new']
if new==False:
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
            RNG[:,:,0] = cur.phase;RNG[:,:,1] = cur.phase;RNG[-2:,...] = [0,1] # still allow amplitudes to switch on/off
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
        
    # Sigmas
    if C.v['sgmr']==True:
        sGm = np.random.normal(0, SGM, (N,))
        mode = 'w' if l==0 else 'a'
        with open(C.v['fopath']+"sgm.txt", mode) as fi:
            fi.write(",".join([str(m) for m in sGm])+"\n")
    else:
        sGm = SGM*C.v['sigs'][:N]
    allsgm[l] = sGm
    
    # Preliminary Hamiltonian
    Hdips=[]
    for m in range(len(coords)):
        a = coords[m]
        Hdip, Hsgm, Sx, Sy, Ixtot, Iytot, Iztot, Sd = pretot()
        Hdips.append(Hdip)
    rho_det = Sd.todense()
    rho00 = ((Sd-Ixtot)/(2**(N-1))).todense()
    
    # Initialize score objects
    cur.pm,cur.pmraw,cur.wid,cur.fac = run3()
    for m in [hold,low,wSO,raw]:
        m.copyfrom(cur,phase=True)
    wSO.wid = 1E4-1
    
    # Initialize temperature variables
    incr  = C.v['incr'];steps = C.v['steps']
    WW = np.linspace(C.v['Ws'][0], C.v['Ws'][1], incr, dtype='float32');W=WW[0]
    que = collections.deque(maxlen=C.v['que']);que.append(0)
    hs = np.zeros((7,dim1));mags = np.zeros((2,7,dim1));mags[1]=1E-10
    for m in range(incr): # Temperature loop
        # Set temperature
        upav = np.mean(np.exp(-W*np.array(que)))#*1.15
        W = WW[m] if hold.pm>C.v['Wathr'] or m==0 else W*(upav/C.v['Wap'])**C.v['Wae']
        
        sys.stdout.write("m = %d; l = %d (%s)\nW = %.2e"%(m,l,files[l],W))
        accepth = 0
        acceptl = 0
        reject = 0
        
        split = time()
        for n in range(steps): # random steps loop
            
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
            
            # Generate random values rnd, rnd2 and impose symmetry
            fsps(sym[ind1, ind2]) # function automatically operates on cur
            # cur.phase[5:] = np.round(cur.phase[5:])
            
            # Score new sequence
            cur.pm,cur.pmraw,cur.wid,cur.fac = run3()
            dE = cur.pm - hold.pm
            
            # HS statistics
            if (C.v['foHS']==True) & (dE!=0):
                mags[0,ind1,ind2]+=abs(dE);mags[1,ind1,ind2]+=1
            # Facilitate uphill steps
            if dE>0:
                dE*=fus[vtyp]
            # Adaptive temperature variable
            que.append(dE) if dE>0 else None
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
        sys.stdout.write(" | %.1f s\n"%(time()-split))
        # print status
        print("   Steps accepted (+|=): %2d"%(accepth))
        print("   Steps accepted (-)  : %2d"%(acceptl))
        print("   Steps rejected      : %2d"%(reject))
        print("   Current score/w/h/f : %5.0f (%5.0f/%5.3f/%5.3f)"%(hold.pm, hold.wid, hold.pmraw, hold.fac))
        print("   Low score/w/h/f     : %5.0f (%5.0f/%5.3f/%5.3f)"%(low.pm, low.wid, low.pmraw, low.fac))
        print("   Global low w/h      :       (%5.0f/%5.3f)\n"%(wSO.wid, raw.pmraw))
        # Print warning if applicable
        for M in np.sum(hold.phase[-2:], axis=1):
            if M==0:
                print("WARNING: Entire channel's power is off\n")
        # Save Plow after every increment
        if C.v['foPlow']==True:
            np.savetxt(C.v['fopath']+"%s"%(files[l]), low.phase, delimiter=",", 
                   header='%s; sgmstd=%d; w_rf=%.2f; dnu=%.0f; pm/raw/wid=%.3f/%.3f/%d'%(
                           os.path.split(sys.argv[0])[1],SGM, w_rf/2000/pi, dnu, low.pm, low.pmraw, low.wid))
        
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
        MAG = np.divide(mags[0],mags[1]);MAG/=np.max(MAG)
        Hs = open(C.v['fopath']+"hs.txt", 'a')
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
    np.savetxt(C.v['fopath']+"RNG.txt", np.row_stack((RNG[:,:,0], RNG[:,:,1])), fmt='%.3f', delimiter=",")
if bool(int(C.v['fosgm']))==True: # save sigmas for reproduceability
    np.savetxt(C.v['fopath']+"sgm.txt", allsgm, fmt='%.3f', delimiter=",")
if bool(int(C.v['foreport']))==True: # save report with scores and misc. info
    rep.append("Total runtime: %.2f"%(time()-START))
    header="\n".join(rep)+"\n"
    for m in files:
        header+=m+" "
    np.savetxt(C.v['fopath']+"report.txt", pms, fmt='%5d %7.2f %7.2f %7.0f', delimiter=" ", header=header)
# sendmsg() # send email of report
print("\nTotal runtime: %.2f"%(time()-START))
