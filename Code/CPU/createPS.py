# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 08:15:06 2019

@author: joell
"""

import numpy as np
import os
import sys

w_rf = 2*np.pi*75E3
T90 = (1E6*np.pi)/(2*w_rf)

def genps2(fp, tbl, t90, r2d,fn='Pulse Sequence'):
    # tbl2 = np.ones((7,6))
    # tbl2[-1,2:4] = 0
    # tbl2[:5] = tbl
    # tbl = tbl2
    r,c = np.shape(tbl)
    dim1 = int(np.ceil(c/2))
    fp.write('%s\n\n'%(fn))
    rev = [];dcount = 1
    for m in range(dim1):
        if m==0:
            space = '4 '
        else:
            space = '  '
        if tbl[-2,m]==1 and tbl[-1,m]==0:
            rev.append('(p1%d ph1%d ipp1%d):f1\n'%(m+1,m+1,m+1))
        elif tbl[-2,m]==0 and tbl[-1,m]==1:
            rev.append('(p1%d ph2%d ipp2%d):f2\n'%(m+1,m+1,m+1))
        elif tbl[-2,m]==1 and tbl[-1,m]==1:
            rev.append('(p1%d ph1%d):f1 (p1%d ph2%d ipp1%d ipp2%d):f2\n'%(m+1,m+1,m+1,m+1,m+1,m+1))
        else:
            rev.append('d1%d\n'%(dcount))
            dcount+=1
        fp.write(space+rev[-1])
    start = (1 if ((c/2)%1)>0 else 0)
    for m in range(start,dim1,1):
        fp.write('  '+rev[-(m+1)])
    fp.write("\n")
    
    for m in range(dim1):
        if tbl[-2,m]==1:
            fp.write("ph1%d=(float, 180.0) "%(m+1))
            fp.write("%.3f %.3f %.3f %.3f\n"%(tbl[0,m]*r2d, tbl[0,-(m+1)]*r2d, tbl[2,m]*r2d, tbl[2,-(m+1)]*r2d))
    for m in range(dim1):
        if tbl[-1,m]==1:
            fp.write("\nph2%d=(float, 180.0) "%(m+1))
            fp.write("%.3f %.3f %.3f %.3f"%(tbl[1,m]*r2d, tbl[1,-(m+1)]*r2d, tbl[3,m]*r2d, tbl[3,-(m+1)]*r2d))
    
    fp.write("\n\nTimings: ")
    for m in range(dim1):
        fp.write("%5.2f "%(tbl[4, m]*t90))
    fp.write("\n\n")

def genps3(fp, tbl, t90, r2d,fn='Pulse Sequence'):
    tbl2 = np.ones((7,6))
    tbl2[:,[0,3]] = 0
    tbl2[:5] = tbl
    tbl = tbl2
    r,c = np.shape(tbl)
    fp.write('%s\n\n'%(fn))
    rev = [];rev2=[];dcount = 1
    for m in range(c):
        if m==0:
            space = '4 '
        else:
            space = '  '
        if tbl[-2,m]==1 and tbl[-1,m]==0:
            rev.append('(p1%d ph1%d ipp1%d):f1\n'%(m+1,m+1,m+1))
            rev2.append('p1%d'%(m+1))
        elif tbl[-2,m]==0 and tbl[-1,m]==1:
            rev.append('(p1%d ph2%d ipp2%d):f2\n'%(m+1,m+1,m+1))
            rev2.append('p1%d'%(m+1))
        elif tbl[-2,m]==1 and tbl[-1,m]==1:
            rev.append('(p1%d ph1%d):f1 (p1%d ph2%d ipp1%d ipp2%d):f2\n'%(m+1,m+1,m+1,m+1,m+1,m+1))
            rev2.append('p1%d'%(m+1))
        else:
            rev.append('d1%d\n'%(dcount))
            rev2.append('d1%d'%(m+1))
            dcount+=1
        fp.write(space+rev[-1])
    fp.write("\n")
    
    for m in range(c):
        if tbl[-2,m]==1:
            fp.write("ph1%d=(float, 180.0) "%(m+1))
            fp.write("%.3f %.3f\n"%(tbl[0,m]*r2d, tbl[2,m]*r2d))
    for m in range(c):
        if tbl[-1,m]==1:
            fp.write("\nph2%d=(float, 180.0) "%(m+1))
            fp.write("%.3f %.3f"%(tbl[1,m]*r2d, tbl[3,m]*r2d))
    
    fp.write("\n\nTimings:")
    for m in rev2:
        fp.write("%6s"%(m))
    fp.write("\n        ")
    for m in range(c):
        fp.write("%6.2f"%(tbl[4, m]*t90))
    fp.write("\n\n")

root = "C:/Users/joell/Documents/Python/Nevzorov/SPINCPU/Loop/"
on=0
fn = os.listdir(root)
f = open(root+'ps.txt', 'w')
for m in fn:
    if m[-4:]=='.csv':
        tbl = np.loadtxt(root+'%s'%(m), delimiter=',', skiprows=1)
        genps2(f, tbl, T90, 90, m)
        
        # Generating neural network data
        # dim1 = int(np.ceil(tbl.shape[1]/2))
        # if on==0:
        #     sys.stdout.write("%s"%(m))
        #     for n in range(2):
        #         if n==0:
        #             sd = dim1
        #         else:
        #             sd = dim1-1
        #         for o in range(sd):
        #             sys.stdout.write(",%.3f,%.3f"%(90*tbl[n,o],90*tbl[n+2,o]))
        #     for o in range(dim1):
        #         sys.stdout.write(",%.2f"%(T90*tbl[4,o]))
        #     sys.stdout.write("\n")
        # else:
        #     sys.stdout.write("%s"%(m))
        #     for n in range(2):
        #         for o in range(dim1):
        #             sys.stdout.write(",%.3f,%.3f"%(90*tbl[n,o],90*tbl[n+2,o]))
        #     for o in range(dim1):
        #         sys.stdout.write(",%.2f"%(T90*tbl[4,o]))
        #     sys.stdout.write("\n")
f.close()