import numpy as np
import pandas as pd
import sys
#===========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This code is traslated from Pzrikidis' fortran code
# to python by Seyed Amin Nabavizadeh
# for Nourhani's research lab
# All rights belongs to C. Pozrikidis
#
# This program is to be used only under the
# stipulations of the licensing agreement
#===========================================

#-----------------------------------------------
#  This program computes the following Stokes flows:
#
#  1) Flow due to the motion of
#     a three-dimensional rigid particle in free space
#     (Iflow=1)
#
#  2) Flow due to the motion of
#     a three-dimensional rigid particle in a semi-infinite
#     domain bounded by a plane wall located at y=wall
#    (Iflow=2)
#
#  The particle is an ellipsoid with:
#
#     x semi-axis equal to a
#     y semi-axis equal to b
#     z semi-axis equal to c

# SYMBOLS:
# --------
#
#  points(i,j)	coordinates of the ith node (j=0,1,2)
#
#  ne (k,j)     ne(k,0) is the number of elements adjacent to point k
#               ne(k,1), ... are the elements numbers, j = 1, ..., 6
#               for this triangulation, up to six
#
#  nn(k,i)       connectivity table: points for element k, i = 0,...,5
#
#  nbe(k,j)     the three neighboring elements of element k (j=0,1,2)
#
#  arel(i)	surface area of ith element
#
#  npts		total number of points
#
#  nelm		total number of elements
#
#  x0, y0, z0         coordinates of collocation points
#  vnx0, vny0, vnz0   unit normal vector at collocation points
#
#  alpha, beta, gamma: parameters for quadratic 
#                      xi-eta isoparametric mapping
#
#  Resist(i,j):   Grand resistance matrix

# number of particles per lattice row

Iflow = int(float(sys.argv[1])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
Ioctaicos = int(float(sys.argv[2])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
# Peclete Number
ndiv = int(sys.argv[3]) # level of discretization of starting octahedron
# number of particles per lattice row
cxp = float(float(sys.argv[4])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
# Peclete Number
wall = float(sys.argv[5]) # level of discretization of starting octahedron



int_ext = 2 # 1 for the interior; 2 for the exterior problem
boa,coa = 1.0, 1.0 # b/a and c/a The particle is a prolate spheroid" 
# with axes: a, b where b<a the axes ratio b/a less than 1
req = 1.0 # equivalent radius
#cxp = 0.0
cyp,czp = 0.0, 0.0 #particle centers
#rotation angles: phix, phiy,phiz
mint = 9 #mint number of Gauss_triangle base points for regular integrals 7 or 9
NGL = 4 #number of Gauss_Legendre base points for singular integrals
visc = 1.0    #fluid viscosity
Uinf = 1.0    #incident velocity          (for Iflow=4)
shrt = 1.0    #shear rate above the array (for Iflow=5)
#wall = -1.2    #wall:  y position of wall (for Iflow=2)
a11,a12,a13 = 1.0, 0.0, 0.0 #First base vector  (triply periodic flow)
a21,a22,a23 = 0.5, 0.866025, 0.0 #Second base vector (triply periodic flow)
a31,a32,a33 = 0.0, 0.0, 1.0 #Third base vector  (triply periodic flow)
Max1 = 5    #Max1 for summation of the periodic Green's function
Max2 = 5    #Max2 for summation of the periodic Green's function
a11,a12 = 1.0, 0.0 # First  base vector  (doubly periodic flow, Iflow=7)
a21,a22 = 0.0, 1.0 #Second base vector  (doubly flow, Iflow=7)
Ns,Np = 2, 2 #Ns, Np (sgf)
Method = 2 #Method for sgf_3d_2p (Enter 1 or 2) (Iflow = 3, 4)
eps = 0.001  #differentiation step for computing sgf_3d_2p for method=2
Iprec = 0 #Iprec: Index for preconditioning (0 for NO or 1 for YES)

# "Iflow=  1 for flow due to the motion of particle"
# "   in free space"
# 2 for flow due to the motion of particle"
# "   in a semi-infinite domain bounded"
# "   by a plane located at y=wall"

if (Ioctaicos == 1):
    if (ndiv == 0):
        num_points = int(18)
        num_elements = int(8)
        num_3elements = int(24)
    if (ndiv == 1):
        num_points = int(66)
        num_elements = int(32)
        num_3elements = int(96)
    if (ndiv == 2):
        num_points = int(258)
        num_elements = int(128)
        num_3elements = int(384)
    if (ndiv == 3):
        num_points = int(1026)
        num_elements = int(512)
        num_3elements = int(1536)
    if (ndiv == 4):
        num_points = int(4098)
        num_elements = int(2048)
        num_3elements = int(6144)


elif (Ioctaicos == 2):
    if (ndiv == 0):
        num_points = int(42)
        num_elements = int(20)
        num_3elements = int(60)
    if (ndiv == 1):
        num_points = int(162)
        num_elements = int(80)
        num_3elements = int(240)
    if (ndiv == 2):
        num_points = int(642)
        num_elements = int(320)
        num_3elements = int(960)
    if (ndiv == 3):
        num_points = int(2562)
        num_elements = int(1280)
        num_3elements = int(3840)


alpha = np.zeros(num_elements) #alpha
beta = np.zeros(num_elements) #beta
gamma = np.zeros(num_elements) #gamma

amat = np.zeros((num_3elements,num_3elements))
#  x0, y0, z0         coordinates of collocation points
#  vnx0, vny0, vnz0   unit normal vector at collocation points
vnx0 = np.zeros(num_elements)
vny0 = np.zeros(num_elements)
vnz0 = np.zeros(num_elements)
x0 = np.zeros(num_elements)
y0 =  np.zeros(num_elements)
z0 = np.zeros(num_elements)


# constant
piq = 0.25 * np.pi
pih = 0.50 * np.pi
pi2 = 2.00 * np.pi
pi4 = 4.00 * np.pi
pi6 = 6.00 * np.pi
pi8 = 8.00 * np.pi
Null = 0
Nfour = 4
Nseven = 7
oot  = 1.0/3.0
 


 

def sgf_3d_fs(x, y, z, x0, y0, z0):


#========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#========================================

#---------------------------------------
# Free-space Green's function: Stokeslet
#
# Pozrikidis (1992, p. 23)
#
#---------------------------------------

    # px, py, pz = 0.0, 0.0, 0.0
    # Txxx, Txxy, Txxz, \
    # Tyxy, Tyxz,Tzxz,  \
    # Txyx, Txyy, Txyz, \
    # Tyyy, Tyyz, Tzyz, \
    # Txzx, Txzy, Txzz, \
    # Tyzy, Tyzz, Tzzz = 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0, 0.0, 0.0,\
    # 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0
            
    dx = x - x0
    dy = y - y0
    dz = z - z0

    dxx = dx * dx
    dxy = dx * dy
    dxz = dx * dz
    dyy = dy * dy
    dyz = dy * dz
    dzz = dz * dz

    r = np.sqrt( dxx + dyy + dzz)
    r3  = r * r * r
    ri  = 1.0 / r
    ri3 = 1.0 / r3

    Gxx = ri + dxx * ri3
    Gxy = dxy * ri3
    Gxz = dxz * ri3
    Gyy = ri + dyy*ri3
    Gyz = dyz * ri3
    Gzz = ri + dzz * ri3

    Gyx = Gxy
    Gzx = Gxz
    Gzy = Gyz
    #---
    # compute the stress tensor and the pressure if needed
    #---

    # cf = -6.0 / (r3* r * r)

    # Txxx = dxx * dx * cf
    # Txxy = dxy * dx * cf
    # Txxz = dxz * dx * cf
    # Tyxy = dyy * dx * cf
    # Tyxz = dyz * dx * cf
    # Tzxz = dzz * dx * cf

    # Txyx = Txxy
    # Txyy = Tyxy
    # Txyz = Tyxz
    # Tyyy = dyy *dy * cf
    # Tyyz = dyz * dy * cf
    # Tzyz = dzz * dy * cf

    # Txzx = Txxz
    # Txzy = Tyxz
    # Txzz = Tzxz
    # Tyzy = dyy * dz * cf
    # Tyzz = dyz * dz * cf
    # Tzzz = dzz * dz * cf

    #---------
    #pressure
    #---------

    # cf = 2.0 * ri3
    # px = dx * cf
    # py = dy * cf
    # pz = dz * cf


    return Gxx, Gxy, Gxz, Gyx, Gyy, Gyz, Gzx, Gzy, Gzz


def sgf_3d_w(x, y, z, x0, y0, z0, wall):


#========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#========================================

#---------------------------------------
# Free-space Green's function: Stokeslet
#
# Pozrikidis (1992, p. 23)
#
#---------------------------------------

    # px, py, pz = 0.0, 0.0, 0.0
    # Txxx, Txxy, Txxz, \
    # Tyxy, Tyxz,Tzxz,  \
    # Txyx, Txyy, Txyz, \
    # Tyyy, Tyyz, Tzyz, \
    # Txzx, Txzy, Txzz, \
    # Tyzy, Tyzz, Tzzz = 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0, 0.0, 0.0,\
    # 0.0, 0.0, 0.0, 0.0, \
    # 0.0, 0.0
            
    dx = x - x0
    dy = y - y0
    dz = z - z0

    dxx = dx * dx
    dxy = dx * dy
    dxz = dx * dz
    dyy = dy * dy
    dyz = dy * dz
    dzz = dz * dz

    r = np.sqrt( dxx + dyy + dzz)
    r3  = r * r * r
    r5  = r3*r*r
    ri  = 1.0 / r
    ri3 = 1.0 / r3
    ri5 = 1.0/ r5
    
    Sxx = ri + dxx*ri3
    Sxy =      dxy*ri3
    Sxz =      dxz*ri3
    Syy = ri + dyy*ri3
    Syz =      dyz*ri3
    Szz = ri + dzz*ri3
    
    #-------------
    # image system
    #-------------

    x0im = 2.0*wall-x0
    dx   = x-x0im

    dxx = dx*dx
    dxy = dx*dy
    dxz = dx*dz

    r   = np.sqrt(dxx+dyy+dzz)
    r3  = r*r*r
    r5  = r3*r*r
    ri  = 1.0/r
    ri3 = 1.0/r3
    ri5 = 1.0/r5
    
    
    #----------------
    # image stokeslet
    #----------------    
    Sxx = Sxx - ri - dxx*ri3
    Sxy = Sxy      - dxy*ri3
    Sxz = Sxz      - dxz*ri3
    Syy = Syy - ri - dyy*ri3
    Syz = Syz      - dyz*ri3
    Szz = Szz - ri - dzz*ri3

    Syx = Sxy
    Szx = Sxz
    Szy = Syz
    #-----------------------
    # image potential dipole
    #-----------------------

    PDxx = - ri3 + 3.0*dxx*ri5
    PDyx =         3.0*dxy*ri5
    PDzx =         3.0*dxz*ri5

    PDxy = - PDyx
    PDyy =   ri3 - 3.0*dyy*ri5
    PDzy =       - 3.0*dyz*ri5

    PDxz = - PDzx
    PDyz =   PDzy
    PDzz =   ri3 - 3.0*dzz*ri5    
    
    
    #-----------------------
    # image stokeslet dipole
    #-----------------------

    SDxx = dx * PDxx
    SDyx = dx * PDyx - dy*ri3
    SDzx = dx * PDzx - dz*ri3

    SDxy = dx * PDxy - dy*ri3
    SDyy = dx * PDyy
    SDzy = dx * PDzy

    SDxz = dx * PDxz - dz*ri3
    SDyz = dx * PDyz
    SDzz = dx * PDzz    
    
    #---------
    # assemble
    #---------

    h0   = x0-wall
    h02  = 2.0*h0
    h0s2 = 2.0*h0*h0

    Gxx = Sxx + h0s2 * PDxx - h02 * SDxx
    Gxy = Sxy + h0s2 * PDxy - h02 * SDxy
    Gxz = Sxz + h0s2 * PDxz - h02 * SDxz

    Gyx = Syx + h0s2 * PDyx - h02 * SDyx
    Gyy = Syy + h0s2 * PDyy - h02 * SDyy
    Gyz = Syz + h0s2 * PDyz - h02 * SDyz

    Gzx = Szx + h0s2 * PDzx - h02 * SDzx
    Gzy = Szy + h0s2 * PDzy - h02 * SDzy
    Gzz = Szz + h0s2 * PDzz - h02 * SDzz    
    



    return Gxx, Gxy, Gxz, Gyx, Gyy, Gyz, Gzx, Gzy, Gzz

def slp_trgl3_sing (x1, y1, z1, x2, y2, z2, x3, y3, z3, zz, ww, \
GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow):

#===========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved.
#
# This program is to be used only under the
# stipulations of the licensing agreement
#===========================================

#---------------------------------------------
# Integrates the Green's function over a flat
# (three-node) triangle in local polar coordinates
# with origin at the singular point:
#
# (x1,y1,z1)
#
# SYMBOLS:
# -------
#
# asm: triangle area computed by numerical integration
#---------------------------------------------


    #---
    # compute surface metric: hs
    #---

    #dx = np.sqrt((x2 - x1)**2+(y2 - y1)**2+(z2 - z1)**2 )
    #dy = np.sqrt((x3 - x1)**2+(y3 - y1)**2+(z3 - z1)**2 )

    vnx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    vny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
    vnz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    area = 0.5*np.sqrt(vnx * vnx + vny * vny + vnz * vnz)

    hs = 2.0*area   # surface metric on a flat triangle

#---
# initialize
#---

    asm = 0.0    # triangle area

    uxx = 0.0
    uxy = 0.0
    uxz = 0.0

    uyx = 0.0
    uyy = 0.0
    uyz = 0.0

    uzx = 0.0
    uzy = 0.0
    uzz = 0.0

#---
# apply the double quadrature
#---

    for i in range(NGL):                  # integration wrt phi

        ph    = piq*(1.0+zz[i])
        cph   = np.cos(ph)
        sph   = np.sin(ph)
        rmax  = 1.0/(cph+sph)
        rmaxh = 0.5*rmax

        sxx = 0.0
        sxy = 0.0
        sxz = 0.0

        syx = 0.0
        syy = 0.0
        syz = 0.0

        szx = 0.0
        szy = 0.0
        szz = 0.0
        bsm = 0.0     # derivative of asm
        for j in range(NGL):
            r  = rmaxh * (1.0 + zz[j])
            xi = r * cph
            et = r * sph
            zt = 1.0 - xi - et

            x = x1 * zt + x2 * xi + x3 * et
            y = y1 * zt + y2 * xi + y3 * et
            z = z1 * zt + z2 * xi + z3 * et
            if (Iflow == 1):
                Gxx, Gxy, Gxz, \
                Gyx, Gyy, Gyz, \
                Gzx, Gzy, Gzz= sgf_3d_fs\
                (x, y, z, x1, y1, z1)
            elif (Iflow == 2):
                # wall at z=wall
                #Gzz, Gzx, Gzy, Gxz, Gxx, Gxy, Gyz, Gyx, Gyy = sgf_3d_w\
                #(z, x, y, z1, x1, y1, wall)


                # wall at y=wall
                #Gyy, Gyz, Gyx, Gzy, Gzz, Gzx, Gxy, Gxz, Gxx = sgf_3d_w\
                #(y, z, x, y1, z1, x1, wall)



                # wall at x=wall
                Gxx, Gxy, Gxz, Gyx, Gyy, Gyz, Gzx, Gzy, Gzz = sgf_3d_w\
                (x, y, z, x1, y1, z1, wall)
            
            
            
            cf = r * ww[j]

            bsm = bsm + cf

            sxx = sxx + Gxx*cf
            sxy = sxy + Gxy*cf
            sxz = sxz + Gxz*cf

            syx = syx + Gyx*cf
            syy = syy + Gyy*cf
            syz = syz + Gyz*cf

            szx = szx + Gzx*cf
            szy = szy + Gzy*cf
            szz = szz + Gzz*cf
        
        
        
        cf = ww[i]*rmaxh
        asm = asm + bsm * cf

        uxx = uxx + sxx * cf
        uxy = uxy + sxy * cf
        uxz = uxz + sxz * cf
        uyx = uyx + syx * cf
        uyy = uyy + syy * cf
        uyz = uyz + syz * cf
        uzx = uzx + szx * cf
        uzy = uzy + szy * cf
        uzz = uzz + szz * cf
        
        
#------------------------
# complete the quadrature
#------------------------

    cf = piq * hs

    asm  = asm * cf

    GExx = GExx + cf * uxx
    GExy = GExy + cf * uxy
    GExz = GExz + cf * uxz
    GEyx = GEyx + cf * uyx
    GEyy = GEyy + cf * uyy
    GEyz = GEyz + cf * uyz
    GEzx = GEzx + cf * uzx
    GEzy = GEzy + cf * uzy
    GEzz = GEzz + cf * uzz
            
    return GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz

def slp_trgl6(x0, y0, z0, k, mint, nn , points, alpha,beta, gamma, xiq, etq, Iflow):
#============================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#============================================

#----------------------------------------
# Integrates the Green's function over a
# non-singular triangle numbered: k
#
# SYMBOLS:
# -------
#
# mint: order of triangle quadrature
#
# GE_ij: integrated ij component over the elemen


    GExx, GExy, GExz, GEyx, GEyy = 0.0, 0.0, 0.0, 0.0, 0.0
    GEyz, GEzx, GEzy, GEzz = 0.0, 0.0, 0.0, 0.0
#---
# vertices of the kth triangle
#---
    i1 = nn[k,0]
    i2 = nn[k,1]
    i3 = nn[k,2]
    i4 = nn[k,3]
    i5 = nn[k,4]
    i6 = nn[k,5]
    for i in range(mint):
        xi  = xiq[i]
        eta = etq[i]
        x,y,z, DxDxi,DyDxi,DzDxi,DxDet,DyDet,DzDet\
        ,vnx,vny,vnz,hs = interp_p (points[i1,0],points[i1,1],points[i1,2]\
        ,points[i2,0],points[i2,1],points[i2,2],points[i3,0],points[i3,1],points[i3,2]\
        ,points[i4,0],points[i4,1],points[i4,2],points[i5,0],points[i5,1],points[i5,2]\
        ,points[i6,0],points[i6,1],points[i6,2],alpha[k],beta[k],gamma[k],xi,eta)
        if (Iflow == 1):
           Gxx, Gxy, Gxz, Gyx, Gyy, Gyz, Gzx, Gzy, Gzz = sgf_3d_fs(x, y, z, x0, y0, z0)
        elif (Iflow == 2):
        
            # wall at z=wall
            #Gzz, Gzx, Gzy, Gxz, Gxx, Gxy, Gyz, Gyx, Gyy = sgf_3d_w\
            #(z, x, y, z2, x2, y2, wall)
            
            
            # wall at y=wall
            #Gyy, Gyz, Gyx, Gzy, Gzz, Gzx, Gxy, Gxz, Gxx = sgf_3d_w\
            #(y, z, x, y2, z2, x2, wall)
            
            
            
            # wall at x=wall
            Gxx, Gxy, Gxz, Gyx, Gyy, Gyz, Gzx, Gzy, Gzz = sgf_3d_w(x, y, z, x0, y0, z0, wall)
        
        fc = 0.5 * hs * wq[i]

        GExx = GExx + Gxx * fc
        GExy = GExy + Gxy * fc
        GExz = GExz + Gxz * fc
        GEyx = GEyx + Gyx * fc
        GEyy = GEyy + Gyy * fc
        GEyz = GEyz + Gyz * fc
        GEzx = GEzx + Gzx * fc
        GEzy = GEzy + Gzy * fc
        GEzz = GEzz + Gzz * fc
        


    return GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz


def slp_trgl6_sing(x0, y0, z0, k, mint, nn , points, alpha,\
beta, gamma, NGL, zz, ww, xiq, etq, Iflow):
#========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved.
#
# This program is to be used only under the
# stipulations of the licensing agreement
#========================================

#----------------------------------------
# Integrate the Green's function over the kth
# singular quadratic triangle
#
# This is done by breaking up the singular
# triangle into six flat triangles, and
# then integrating individually over the
# flat triangles in local poral coordinates

    GExx, GExy, GExz, GEyx, GEyy = 0.0, 0.0, 0.0, 0.0, 0.0
    GEyz, GEzx, GEzy, GEzz = 0.0, 0.0, 0.0, 0.0

#---
    i1 = nn[k,0]
    i2 = nn[k,1]
    i3 = nn[k,2]
    i4 = nn[k,3]
    i5 = nn[k,4]
    i6 = nn[k,5]
    
#---
# Integrate over six flat triangles
#---

    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i1,0], points[i1,1], points[i1,2],\
    points[i4,0], points[i4,1], points[i4,2], zz, ww, \
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i4,0], points[i4,1], points[i4,2],\
    points[i2,0], points[i2,1], points[i2,2], zz, ww,\
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i2,0], points[i2,1], points[i2,2],\
    points[i5,0], points[i5,1], points[i5,2], zz, ww,\
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i5,0], points[i5,1], points[i5,2],\
    points[i3,0], points[i3,1], points[i3,2], zz, ww,\
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i3,0], points[i3,1], points[i3,2],\
    points[i6,0], points[i6,1], points[i6,2], zz, ww,\
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
    slp_trgl3_sing (x0, y0, z0,\
    points[i6,0], points[i6,1], points[i6,2],\
    points[i1,0], points[i1,1], points[i1,2], zz, ww,\
    GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz, NGL, Iflow)
    
    return GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz
    
    




def gauss_leg(NQ):
#=========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#=========================================

#------------------------------------------------
# This program accompanies the book:
#
#             C. Pozrikidis
# ``Numerical Computation in Science and Engineering''
#        Oxford University Press
#------------------------------------------------
    Z = np.zeros(20) #related to zw1
    W = np.zeros(20) #related to zw1 

    
    #--------------------
    if (NQ == 1):
    #--------------------
        
        Z[0] = 0.0
        
        W[0] = 2.0  
        
    elif (NQ == 2):
#-------------------------

        Z[0] = -0.57735026918962576450
        Z[1] = -Z[0]
    
        W[0] = 1.0
        W[1] = 1.0
    
    #-------------------------
    elif (NQ == 3):
    #-------------------------
    
          Z[0] = -0.77459666924148337703
          Z[1] =  0.0
          Z[2] = -Z[0]
        
          W[0] = 0.555555555555555555555
          W[1] = 0.888888888888888888888
          W[2] = 0.555555555555555555555
    
    #-------------------------
    elif (NQ == 4):
    #-------------------------
    
        Z[0] = -0.86113631159405257522
        Z[1] = -0.33998104358485626480 
        Z[2] = -Z[1]
        Z[3] = -Z[0]

        W[0] = 0.34785484513745385737
        W[1] = 0.65214515486254614262
        W[2] = W[1]
        W[3] = W[0]
    
    #-------------------------
    elif (NQ == 5):
    #-------------------------
    
        Z[0] = -0.90617984593866399279
        Z[1] = -0.53846931010568309103
        Z[2] =  0.0
        Z[3] = -Z[1]
        Z[4] = -Z[0]

        W[0] = 0.23692688505618908751
        W[1] = 0.47862867049936646804
        W[2] = 0.56888888888888888889
        W[3] = W[1]
        W[4] = W[0]
    
    #-------------------------
    elif (NQ == 6):
    #-------------------------
    
        Z[0] = -0.932469514203152
        Z[1] = -0.661209386466265
        Z[2] = -0.238619186083197

        Z[3] = -Z[2]
        Z[4] = -Z[1]
        Z[5] = -Z[0]

        W[0] = 0.171324492379170
        W[1] = 0.360761573048139
        W[2] = 0.467913934572691

        W[3] = W[2]
        W[4] = W[1]
        W[5] = W[0]
    
    #-------------------------
    elif(NQ == 8):
    #-------------------------
    
        Z[0] = -0.96028985649753623168
        Z[1] = -0.79666647741362673959
        Z[2] = -0.52553240991632898581
        Z[3] = -0.18343464249564980493

        Z[4] = -Z[3]
        Z[5] = -Z[2]
        Z[6] = -Z[1]
        Z[7] = -Z[0]

        W[0] = 0.10122853629037625915
        W[1] = 0.22238103445337447054
        W[2] = 0.31370664587788728733
        W[3] = 0.36268378337836198296

        W[4] = W[3]
        W[5] = W[2]
        W[6] = W[1]
        W[7] = W[0]
    
    #--------------------------
    elif(NQ == 12):
    #--------------------------
    
        Z[0] = -0.981560634246719
        Z[1] = -0.904117256370475
        Z[2] = -0.769902674194305
        Z[3] = -0.587317954286617
        Z[4] = -0.367831498998180
        Z[5] = -0.125233408511469

        Z[6] = -Z[5]
        Z[7] = -Z[4]
        Z[8] = -Z[3]
        Z[9]= -Z[2]
        Z[10]= -Z[1]
        Z[11]= -Z[0]

        W[0] = 0.047175336386511
        W[1] = 0.106939325995318
        W[2] = 0.160078328543346
        W[3] = 0.203167426723066
        W[4] = 0.233492536538355
        W[5] = 0.249147045813403

        W[6] = W[5]
        W[7] = W[4]
        W[8] = W[3]
        W[9]= W[2]
        W[10]= W[1]
        W[11]= W[0]
    
    #---------------------------
    elif (NQ == 20):
    #---------------------------
    
        Z[0] = -0.993128599185094924786
        Z[1] = -0.963971927277913791268
        Z[2] = -0.912234428251325905868
        Z[3] = -0.839116971822218823395
        Z[4] = -0.746331906460150792614
        Z[5] = -0.636053680726515025453
        Z[6] = -0.510867001950827098004
        Z[7] = -0.373706088715419560673
        Z[8] = -0.227785851141645078080
        Z[9]= -0.076526521133497333755

        Z[10] = -Z[9]
        Z[11] = -Z[8]
        Z[12] = -Z[7]
        Z[13] = -Z[6]
        Z[14] = -Z[5]
        Z[15] = -Z[4]
        Z[16] = -Z[3]
        Z[17] = -Z[2]
        Z[18] = -Z[1]
        Z[19] = -Z[0]

        W[0] = 0.017614007139152118312
        W[1] = 0.040601429800386941331
        W[2] = 0.062672048334109063570
        W[3] = 0.083276741576704748725
        W[4] = 0.101930119817240435037
        W[5] = 0.118194531961518417312
        W[6] = 0.131688638449176626898
        W[7] = 0.142096109318382051329
        W[8] = 0.149172986472603746788
        W[9]= 0.152753387130725850698

        W[10] = W[9]
        W[11] = W[8]
        W[12] = W[7]
        W[13] = W[6]
        W[14] = W[5]
        W[15] = W[4]
        W[16] = W[3]
        W[17] = W[2]
        W[18] = W[1]
        W[19] = W[0]
    return Z, W       
    
def gauss_trgl(N):
    #=========================================
    # FDLIB
    #
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licencing agreement
    #========================================

    #------------------------------------
    # Abscissas and weights for 
    # Gaussian integration over a triangle
    #
    # Integration is performed with respect to the
    # triangle barycentric coordinates
    #
    # SYMBOLS:
    # -------
    #
    # N: order of the quadrature
    #    choose from 1,3,4,5,7,9,12,13
    #
    #    Default value is 7
    #------------------------------------    
    xi = np.zeros(20) #related to tra quadratures
    eta = np.zeros(20) #related to tra quadratures
    w = np.zeros(20) #related to traq    
    if (N != 1 and N != 3 and N !=4 \
        and N != 6 and N != 7 and N != 9\
            and N != 12 and N != 13):
        print('gauss_trgl:')
        print('Number of Guass triangle quadrature points not available;')
        print('Will take N=7.')
        N = 7
    if (N ==1):
        xi[0] = 1.0/3.0
        eta[0] = 1.0/3.0
        w[0] = 1.0
    elif (N ==3):
        xi[0] = 1.0/6.0
        eta[0] = 1.0/6.0
        w[0] = 1.0/3.0

        xi[1] = 2.0/3.0
        eta[1] = 1.0/6.0
        w[1] = w[0]

        xi[2] = 1.0/6.0
        eta[2] = 2.0/3.0
        w[2] = w[0]
    elif (N ==4):
        xi[0] = 1.0/3.0
        eta[0] = 1.0/3.0
        w[0] = -27.0/48.0

        xi[1] = 1.0/5.0
        eta[1] = 1.0/5.0
        w[1] = 25.0/48.0

        xi[2] = 3.0/5.0
        eta[2] = 1.0/5.0
        w[2] = 25.0/48.0

        xi[3] = 1.0/5.0
        eta[3] = 3.0/5.0
        w[3] = 25.0/48.0
    elif (N ==6):
        al = 0.816847572980459
        be = 0.445948490915965
        ga = 0.108103018168070
        de = 0.091576213509771
        o1 = 0.109951743655322
        o2 = 0.223381589678011

        xi[0] = de
        xi[1] = al
        xi[2] = de
        xi[3] = be
        xi[4] = ga
        xi[5] = be

        eta[0] = de
        eta[1] = de
        eta[2] = al
        eta[3] = be
        eta[4] = be
        eta[5] = ga

        w[0] = o1
        w[1] = o1
        w[2] = o1
        w[3] = o2
        w[4] = o2
        w[5] = o2
    elif (N ==7):
        al = 0.797426958353087
        be = 0.470142064105115        
        ga = 0.059715871789770
        de = 0.101286507323456
        o1 = 0.125939180544827
        o2 = 0.132394152788506

        xi[0] = de
        xi[1] = al
        xi[2] = de
        xi[3] = be
        xi[4] = ga
        xi[5] = be
        xi[6] = 1.0/3.0

        eta[0] = de
        eta[1] = de
        eta[2] = al
        eta[3] = be
        eta[4] = be
        eta[5] = ga
        eta[6] = 1.0/3.0

        w[0] = o1
        w[1] = o1
        w[2] = o1
        w[3] = o2
        w[4] = o2
        w[5] = o2
        w[6] = 0.225
    elif (N == 9):
    #-------------------------

        al = 0.124949503233232
        qa = 0.165409927389841
        rh = 0.797112651860071
        de = 0.437525248383384
        ru = 0.037477420750088
        o1 = 0.205950504760887
        o2 = 0.063691414286223

        xi[0] = de
        xi[1] = al
        xi[2] = de
        xi[3] = qa
        xi[4] = ru
        xi[5] = rh
        xi[6] = qa
        xi[7] = ru
        xi[8] = rh

        eta[0] = de
        eta[1] = de
        eta[2] = al
        eta[3] = ru
        eta[4] = qa
        eta[5] = qa
        eta[6] = rh
        eta[7] = rh
        eta[8] = ru

        w[0] = o1
        w[1] = o1
        w[2] = o1
        w[3] = o2
        w[4] = o2
        w[5] = o2
        w[6] = o2
        w[7] = o2
        w[8] = o2

    #--------------------------
    elif (N == 12):
    #--------------------------

        al = 0.873821971016996
        be = 0.249286745170910
        ga = 0.501426509658179
        de = 0.063089014491502
        rh = 0.636502499121399
        qa = 0.310352451033785
        ru = 0.053145049844816
        o1 = 0.050844906370207
        o2 = 0.116786275726379
        o3 = 0.082851075618374

        xi[0]  = de
        xi[1]  = al
        xi[2]  = de
        xi[3]  = be
        xi[4]  = ga
        xi[5]  = be
        xi[6]  = qa
        xi[7]  = ru
        xi[8]  = rh
        xi[9] = qa
        xi[10] = ru
        xi[11] = rh

        eta[0]  = de
        eta[1]  = de
        eta[2]  = al
        eta[3]  = be
        eta[4]  = be
        eta[5]  = ga
        eta[6]  = ru
        eta[7]  = qa
        eta[8]  = qa
        eta[9] = rh
        eta[10] = rh
        eta[11] = ru

        w[0]  = o1
        w[1]  = o1
        w[2]  = o1
        w[3]  = o2
        w[4]  = o2
        w[5]  = o2
        w[6]  = o3
        w[7]  = o3
        w[8]  = o3
        w[9] = o3
        w[10] = o3
        w[11] = o3

    #---------------------------
    elif (N == 13):
    #---------------------------

        al = 0.479308067841923
        be = 0.065130102902216
        ga = 0.869739794195568
        de = 0.260345966079038
        rh = 0.638444188569809
        qa = 0.312865496004875
        ru = 0.048690315425316
        o1 = 0.175615257433204
        o2 = 0.053347235608839
        o3 = 0.077113760890257
        o4 =-0.149570044467670

        xi[0]  = de
        xi[1]  = al
        xi[2]  = de
        xi[3]  = be
        xi[4]  = ga
        xi[5]  = be
        xi[6]  = qa
        xi[7]  = ru
        xi[8]  = rh
        xi[9] = qa
        xi[10] = ru
        xi[11] = rh
        xi[12] = 1.0/3.0

        eta[0]  = de
        eta[1]  = de
        eta[2]  = al
        eta[3]  = be
        eta[4]  = be
        eta[5]  = ga
        eta[6]  = ru
        eta[7]  = qa
        eta[8]  = qa
        eta[9] = rh
        eta[10] = rh
        eta[11] = ru
        eta[12] = 1.0/3.0

        w[0]  = o1
        w[1]  = o1
        w[2]  = o1
        w[3]  = o2
        w[4]  = o2
        w[5]  = o2
        w[6]  = o3
        w[7]  = o3
        w[8]  = o3
        w[9] = o3
        w[10] = o3
        w[11] = o3
        w[12] = o4
    return N,xi,eta,w    
 
def elm_geom (nelm,npts,mint,points, nn,alpha, beta, gamma, xiq,etq,wq):
#========================================
# FDLIB, BEMLIB
#
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#========================================

#----------------------------------------
# Compute:
#
# (a) the surface area of the individual elements
# (b) the x, y, and z surface moments over each element
# (c) the total surface area and volume
# (d) the mean curvature of each element
# (e) the average value of the normal vector at each node;
#     this is done by computing the normal vector at the 6 nodes
#     of each triangle, and then averaging the contributions.
#
#  SYMBOLS:
#  --------
#
#  area:     total surface area
#  vlm:      total volume enclosed by the surface area
#  cx,cy,cz: surface centroid
#
#  itali[i]: number of elements sharing a node
#----------------------------------------
#---
# initialize
#---

    arel = np.zeros(num_elements) #related to geom element
    xmom = np.zeros(num_elements) #x moment over each elemnt
    ymom = np.zeros(num_elements) #y moment over each elemnt
    zmom = np.zeros(num_elements) #z moment over each elemnt
    vna = np.zeros((num_points,3)) #related to geometry    
    crvmel = np.zeros(num_elements) #related to geom element    

    itally = np.zeros(num_points)
    xxi = np.zeros(6)
    eet = np.zeros(6)

    DxDx = np.zeros(6)
    DyDx = np.zeros(6)
    DzDx = np.zeros(6)
    
    DxDe = np.zeros(6)
    DyDe = np.zeros(6)
    DzDe = np.zeros(6)
    
    vx = np.zeros(6)
    vy = np.zeros(6)
    vz = np.zeros(6)


    area = 0.0
    vlm  = 0.0


    for i in range(npts):
        vna[i,0] = 0.0
        vna[i,1] = 0.0
        vna[i,2] = 0.0

        itally[i] = 0    # for averaging at a node
    for k in range(nelm):
        i1 = nn[k,0]    # global node labels
        i2 = nn[k,1]
        i3 = nn[k,2]
        i4 = nn[k,3]
        i5 = nn[k,4]
        i6 = nn[k,5]

        al = alpha[k]
        be = beta [k]
        ga = gamma[k]

        alc = 1.0-al
        bec = 1.0-be
        gac = 1.0-ga        
#---
# initialize
# element variables
#---
        arel[k] = 0.0   # element area

        xmom[k] = 0.0   # element moments
        ymom[k] = 0.0
        zmom[k] = 0.0
        
#-----------------------------------------
# compute:
#
#  surface area of the individual elements
#  x, y, and z surface moments
#  total surface area and enclosed volume
#----------------------------------------
        for i in range(mint):
            xi  = xiq[i]
            eta = etq[i]
            x,y,z, DxDxi,DyDxi,DzDxi,DxDet,DyDet,DzDet\
            ,vnx,vny,vnz,hs = interp_p (points[i1,0],points[i1,1],points[i1,2]\
            ,points[i2,0],points[i2,1],points[i2,2],points[i3,0],points[i3,1],points[i3,2]\
            ,points[i4,0],points[i4,1],points[i4,2],points[i5,0],points[i5,1],points[i5,2]\
            ,points[i6,0],points[i6,1],points[i6,2],al,be,ga,xi,eta)
            cf = hs*wq[i]

            arel[k] = arel[k] + cf 

            xmom[k] = xmom[k] + cf*x
            ymom[k] = ymom[k] + cf*y
            zmom[k] = zmom[k] + cf*z

            vlm = vlm + (x*vnx+y*vny+z*vnz)*cf
        arel[k] = 0.5*arel[k]
        xmom[k] = 0.5*xmom[k]
        ymom[k] = 0.5*ymom[k]
        zmom[k] = 0.5*zmom[k]

        area = area +arel[k]

#------------------------------------------------------
# compute:
#
#   (a) the average value of the normal vector
#   (b) the mean curvature as a contour integral
#       using the nifty formula (4.2.10)
#       of Pozrikidis (1997)
#------------------------------------------------------

#---
# node triangle coordinates
#---
        xxi[0] = 0.0
        eet[0] = 0.0

        xxi[1] = 1.0
        eet[1] = 0.0

        xxi[2] = 0.0
        eet[2] = 1.0

        xxi[3] = al
        eet[3] = 0.0

        xxi[4] = ga
        eet[4] = gac

        xxi[5] = 0.0
        eet[5] = be
#---
# loop over triangle coordinates
# of the nodes
#---
        for i in range(6):
            xi  = xxi[i]
            eta = eet[i]
            x,y,z,DxDx[i],DyDx[i],DzDx[i],DxDe[i],DyDe[i],DzDe[i],vx[i],vy[i],vz[i],hs = interp_p (\
            points[i1,0],points[i1,1],points[i1,2],points[i2,0],points[i2,1],points[i2,2],\
            points[i3,0],points[i3,1],points[i3,2],points[i4,0],points[i4,1],points[i4,2],\
            points[i5,0],points[i5,1],points[i5,2],points[i6,0],points[i6,1],points[i6,2],al,be,ga,xi,eta) 
            m = nn[k,i]  #global index of local node i on element k

            vna[m,0] = vna[m,0] + vx[i]
            vna[m,1] = vna[m,1] + vy[i]
            vna[m,2] = vna[m,2] + vz[i]

            itally[m] = itally[m]+1
        crvmel[k] = 0.0
        Ido = 1
        Ido = 0  # skip the computation of the element mean curvature 
        if(Ido == 1):
#---
# line integral along segment 1-4-2
#---

            bvx1 = vy[0]*DzDx[0]-vz[0]*DyDx[0]
            bvy1 = vz[0]*DxDx[0]-vx[0]*DzDx[0]
            bvz1 = vx[0]*DyDx[0]-vy[0]*DxDx[0]

            bvx2 = vy[3]*DzDx[3]-vz[3]*DyDx[3]
            bvy2 = vz[3]*DxDx[3]-vx[3]*DzDx[3]
            bvz2 = vx[3]*DyDx[3]-vy[3]*DxDx[3]

            bvx3 = vy[1]*DzDx[1]-vz[1]*DyDx[1]
            bvy3 = vz[1]*DxDx[1]-vx[1]*DzDx[1]
            bvz3 = vx[1]*DyDx[1]-vy[1]*DxDx[1]

            crvx = al*bvx1 + bvx2 + alc*bvx3
            crvy = al*bvy1 + bvy2 + alc*bvy3
            crvz = al*bvz1 + bvz2 + alc*bvz3 

#---
# computation of mean curvature:
# line integral along segment 2-5-3
#---

            bvx1 = vy[1]*DzDx[1]-vz[1]*DyDx[1]
            bvy1 = vz[1]*DxDx[1]-vx[1]*DzDx[1]
            bvz1 = vx[1]*DyDx[1]-vy[1]*DxDx[1]

            bvx2 = vy[4]*DzDx[4]-vz[4]*DyDx[4]
            bvy2 = vz[4]*DxDx[4]-vx[4]*DzDx[4]
            bvz2 = vx[4]*DyDx[4]-vy[4]*DxDx[4]

            bvx3 = vy[2]*DzDx[2]-vz[2]*DyDx[2]
            bvy3 = vz[2]*DxDx[2]-vx[2]*DzDx[2]
            bvz3 = vx[2]*DyDx[2]-vy[2]*DxDx[2]

            #---
            crvx = crvx - gac*bvx1 - bvx2 - ga*bvx3
            crvy = crvy - gac*bvy1 - bvy2 - ga*bvy3
            crvz = crvz - gac*bvz1 - bvz2 - ga*bvz3
            #---

            bvx1 = vy[1]*DzDe[1]-vz[1]*DyDe[1]
            bvy1 = vz[1]*DxDe[1]-vx[1]*DzDe[1]
            bvz1 = vx[1]*DyDe[1]-vy[1]*DxDe[1]

            bvx2 = vy[4]*DzDe[4]-vz[4]*DyDe[4]
            bvy2 = vz[4]*DxDe[4]-vx[4]*DzDe[4]
            bvz2 = vx[4]*DyDe[4]-vy[4]*DxDe[4]

            bvx3 = vy[2]*DzDe[2]-vz[2]*DyDe[2]
            bvy3 = vz[2]*DxDe[2]-vx[2]*DzDe[2]
            bvz3 = vx[2]*DyDe[2]-vy[2]*DxDe[2]

            crvx = crvx + gac*bvx1 + bvx2 + ga*bvx3
            crvy = crvy + gac*bvy1 + bvy2 + ga*bvy3
            crvz = crvz + gac*bvz1 + bvz2 + ga*bvz3

            #--
            # computation of curvature
            # line integral along segment 3-6-1
            #---

            bvx1 = vy[0]*DzDe[0]-vz[0]*DyDe[0]
            bvy1 = vz[0]*DxDe[0]-vx[0]*DzDe[0]
            bvz1 = vx[0]*DyDe[0]-vy[0]*DxDe[0]

            bvx2 = vy[5]*DzDe[5]-vz[5]*DyDe[5]
            bvy2 = vz[5]*DxDe[5]-vx[5]*DzDe[5]
            bvz2 = vx[5]*DyDe[5]-vy[5]*DxDe[5]

            bvx3 = vy[2]*DzDe[2]-vz[2]*DyDe[2]
            bvy3 = vz[2]*DxDe[2]-vx[2]*DzDe[2]
            bvz3 = vx[2]*DyDe[2]-vy[2]*DxDe[2]

            crvx = crvx - be*bvx1 - bvx2 - bec*bvx3
            crvy = crvy - be*bvy1 - bvy2 - bec*bvy3
            crvz = crvz - be*bvz1 - bvz2 - bec*bvz3

            cf = 0.25/arel(k)

            #--------------------------------------
            # one way to compute the mean curvature
            # is to consider the norm of the contour
            # integral around the edges:
            #
            #     crvx      = cf*crvx
            #     crvy      = cf*crvy
            #     crvz      = cf*crvz
            #     crvmel(k) = sqrt(crvx**2+crvy**2+crvz**2]
            #--------------------------------------

            #---------------------------------------------
            # another way to compute the mean curvature is to
            # project the curvature vector onto the normal
            # vector at the element centroid
            #---------------------------------------------

            xi  = 1.0/3.0
            eta = 1.0/3.0
            x,y,z,DxDxi,DyDxi,DzDxi,DxDet,\
            DyDet,DzDet,vnx,vny,vnz,hs = interp_p (points[i1,0],points[i1,1],\
            points[i1,2],points[i2,0],points[i2,1],points[i2,2],\
            points[i3,0],points[i3,1],points[i3,2]\
            ,points[i5,0],points[i5,1],points[i5,2],\
            points[i6,0],points[i6,1],points[i6,2]\
            ,alpha[k],beta[k],gamma[k],xi,eta)
            crvmel[k] = cf*(crvx*vnx+crvy*vny+crvz*vnz)
    #---------------------------------------
    # Average the normal vector at the nodes
    # and then normalize to make its
    # length equal to unity
    #---------------------------------------    
    for i in range(npts):
        par = float(itally[i])

        vna[i,0] = vna[i,0]/par
        vna[i,1] = vna[i,1]/par
        vna[i,2] = vna[i,2]/par

        par = np.sqrt(vna[i,0]**2+vna[i,1]**2+vna[i,2]**2)

        vna[i,0] = vna[i,0]/par
        vna[i,1] = vna[i,1]/par
        vna[i,2] = vna[i,2]/par
#---
# final computation of the surface-centroid
# and volume
#---
    cx = np.sum(xmom)/area
    cy = np.sum(ymom)/area
    cz = np.sum(zmom)/area

    vlm = 0.5*vlm/3.0
            
    return xmom,ymom,zmom,area,vlm,cx,cy,cz, arel, vna, crvmel  




    
    
def interp_p (x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,\
z4,x5,y5,z5,x6,y6,z6,al,be,ga,xi,eta):


#============================================
# FDLIB, BEMLIB
# 
# Copyright by C. Pozrikidis, 1999
# All rights reserved
#
# This program is to be used only under the
# stipulations of the licensing agreement
#============================================

#----------------------------------------------------
#  Interpolate over an element to compute geometrical 
#  variables including the following:
#
#  0] position vector
#  1] eangential vectors in the xi and eta directions
#  2] unit normal vector
#  4) line and surface metrics
#
#-----------------------------------------------------------

#--------
# prepare
#--------

    alc = 1.0-al
    bec = 1.0-be
    gac = 1.0-ga

    alalc = al*alc
    bebec = be*bec
    gagac = ga*gac

#-------------------------
# evaluate basis functions
#-------------------------

    ph2 = xi *(xi -al+eta*(al-ga)/gac)/alc
    ph3 = eta*(eta-be+xi *(be+ga-1.0)/ga)/bec
    ph4 = xi *(1.0-xi-eta)/alalc
    ph5 = xi*eta/gagac
    ph6 = eta*(1.0-xi-eta)/bebec
    ph1 = 1.0-ph2-ph3-ph4-ph5-ph6

#------------------------------------------
# interpolate the position vector (x, y, z)
#------------------------------------------

    x = x1*ph1 + x2*ph2 + x3*ph3 + x4*ph4 + x5*ph5 + x6*ph6
    y = y1*ph1 + y2*ph2 + y3*ph3 + y4*ph4 + y5*ph5 + y6*ph6
    z = z1*ph1 + z2*ph2 + z3*ph3 + z4*ph4 + z5*ph5 + z6*ph6

#---------------------------
#-----------
#---
# evaluate xi derivatives of basis functions
#---

    dph2 =  (2.0*xi-al +eta*(al-ga)/gac)/alc
    dph3 =  eta*(be+ga-1.0)/(ga*bec)
    dph4 =  (1.0-2.0*xi-eta)/alalc
    dph5 =  eta/gagac
    dph6 = -eta/bebec
    dph1 = -dph2-dph3-dph4-dph5-dph6

#---
# compute dx/dxi from xi derivatives of phi
#---

    DxDxi = x1*dph1 + x2*dph2 + x3*dph3 + x4*dph4+ x5*dph5 + x6*dph6
    DyDxi = y1*dph1 + y2*dph2 + y3*dph3 + y4*dph4+ y5*dph5 + y6*dph6
    DzDxi = z1*dph1 + z2*dph2 + z3*dph3 + z4*dph4+ z5*dph5 + z6*dph6

#---
# evaluate eta derivatives of basis functions
#---

    pph2 =  xi*(al-ga)/(alc*gac)
    pph3 =  (2.0*eta-be+xi*(be+ga-1.0)/ga)/bec
    pph4 =  -xi/alalc
    pph5 =   xi/gagac
    pph6 =  (1.0-xi-2.0*eta)/bebec
    pph1 = -pph2-pph3-pph4-pph5-pph6

#---
# compute dx/deta from eta derivatives of phi 
#---

    DxDet = x1*pph1 + x2*pph2 + x3*pph3 + x4*pph4+ x5*pph5 + x6*pph6

    DyDet = y1*pph1 + y2*pph2 + y3*pph3 + y4*pph4+ y5*pph5 + y6*pph6

    DzDet = z1*pph1 + z2*pph2 + z3*pph3 + z4*pph4+ z5*pph5 + z6*pph6

#--
#  normal vector    vn = (DxDxi)x(DxDeta) 
#  surface metric   hs = norm(vn) 
#---

    vnx = DyDxi * DzDet - DyDet * DzDxi
    vny = DzDxi * DxDet - DzDet * DxDxi
    vnz = DxDxi * DyDet - DxDet * DyDxi

    hs = np.sqrt(vnx*vnx + vny*vny + vnz*vnz)

#---
#  normalization
#---

    vnx = vnx/hs
    vny = vny/hs
    vnz = vnz/hs

    return x,y,z,DxDxi,DyDxi,DzDxi,DxDet,DyDet,DzDet,vnx,vny,vnz,hs
    
    
def abcd(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6):
    #========================================
    # FDLIB, BEMLIB
    #
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement
    #========================================

    #--------------------------------------
    # compute the parametric representation
    # coefficients alpha, beta, gamma
    #--------------------------------------
    d42 = np.sqrt( (x4-x2)**2 + (y4-y2)**2 + (z4-z2)**2 )
    d41 = np.sqrt( (x4-x1)**2 + (y4-y1)**2 + (z4-z1)**2 )
    d63 = np.sqrt( (x6-x3)**2 + (y6-y3)**2 + (z6-z3)**2 )
    d61 = np.sqrt( (x6-x1)**2 + (y6-y1)**2 + (z6-z1)**2 )
    d52 = np.sqrt( (x5-x2)**2 + (y5-y2)**2 + (z5-z2)**2 )
    d53 = np.sqrt( (x5-x3)**2 + (y5-y3)**2 + (z5-z3)**2 )

    al = 1.0/(1.0+d42/d41)
    be = 1.0/(1.0+d63/d61)
    ga = 1.0/(1.0+d52/d53)
    return al,be,ga

   
    




    
    

    
    



   

################################################################################
################
################
################
################
################          #  #   #   # # #
################         ## ###  ###  # ###
################         #   # #   # # # #
################
###############################################################################


# Abscissas and weights for the Gauss--Legendre
# quadrature with NQ points
#
# This table contains values for
#
#  NGL = 1, 2, 3, 4, 5, 6, 8, 12, 20
#
#  Default value is 20
# Triangulation of the unit sphere
# by subdividing a regular octahedron
# into six-node quadratic triangles.


if(NGL != 1 and NGL != 2 and NGL != 3\
   and NGL != 4 and NGL != 5 and NGL != 6 \
   and NGL != 8 and NGL != 12 and NGL != 20) :
    
    
    print( 'gauss_leg:')
    print( ' chosen number of Gaussian points')
    print( ' is not available; Will take NQ=20')
    NGL = 20



# reading csv file
points = pd.read_csv('geometry.csv', header=None, index_col=None).values
ne = pd.read_csv('ne.csv', header=None, index_col=None).values
nn = pd.read_csv('nn.csv', header=None, index_col=None).values
nbe = pd.read_csv('nbe.csv', header=None, index_col=None).values

npts = len(points)
nelm = len(nn)
print('number of points:', npts)
print('number of elements:', nelm)

for j in range(npts):
    tem = ne[j,0] +1
    ne[j,1:tem] = ne[j,1:tem] - 1
nbe = nbe -1
nn = nn -1

points[:,0] = points[:,0] +cxp



#---------------
# prepare to run
#---------------

nelm2 = 2 * nelm


#---------------------
# read the quadratures
#---------------------
zz,ww = gauss_leg (NGL)

#xiq = np.zeros(20) #related to tra quadratures
#etq = np.zeros(20) #related to tra quadratures
#wq = np.zeros(20) #related to traq 
mint,xiq,etq,wq = gauss_trgl(mint)


#---------------------------------------------
# compute the coefficients alpha, beta, gamma,
# for the quadratic xi-eta mapping
# of each element
#---------------------------------------------

for k in range(nelm):
    i1 = nn[k,0]
    i2 = nn[k,1]
    i3 = nn[k,2]
    i4 = nn[k,3]
    i5 = nn[k,4]
    i6 = nn[k,5]
    alpha[k], beta[k], gamma[k] = abcd(points[i1,0],points[i1,1],\
    points[i1,2],points[i2,0],points[i2,1],\
    points[i2,2],points[i3,0],points[i3,1],\
    points[i3,2],points[i4,0],points[i4,1],\
    points[i4,2],points[i5,0],points[i5,1],\
    points[i5,2],points[i6,0],points[i6,1],points[i6,2])
#------------------------------------------------
# compute:
#
#  vlm:  surface area of the individual elements
#  xmom,yomo,zmom:  x, y, and z surface moments
#                      over each element
#    xmom = np.zeros(512) #x moment over each elemnt
#    ymom = np.zeros(512) #y moment over each elemnt
#    zmom = np.zeros(512) #z moment over each elemnt
#  area:   total surface area and volume
#  crvmel: mean curvature over each element
#  vna:    average normal vector at the nodes
#  are1 #related to geom element
#  cx,cy,cz: surface centroid
#------------------------------------------------
xmom,ymom,zmom,area,vlm,cx,cy,cz, arel, vna, crvmel = \
elm_geom (nelm, npts, mint, points, nn, alpha, beta, gamma, xiq, etq, wq)

dict1 = {'x':xmom, 'y': ymom, 'z':zmom, 'arel': arel} 
df = pd.DataFrame(dict1)
df.to_csv("xyz_mom_area1.csv", header =False , index=False)

#---
# normalize surface area and volume
#---

area = area/(pi4*req**2)          # normalize
vlm  = 3.0*vlm /(pi4*req**3)    # normalize

print('area:',area)
print('vlm:',vlm)
print("cx={}, cy={}, cz={}".format(cx,cy,cz))


#----------------------------------
# Collocation points will be placed
# at the element centroids
#
# Compute:
#
#  1) coordinates of collocation points
#  2) normal vector at collocation points
#----------------------------------
xi  = 1.0/3.0
eta = 1.0/3.0

for i in range(nelm):
    i1 = nn[i,0]    # global node labels
    i2 = nn[i,1]
    i3 = nn[i,2]
    i4 = nn[i,3]
    i5 = nn[i,4]
    i6 = nn[i,5]


#  x0, y0, z0         coordinates of collocation points
#  vnx0, vny0, vnz0   unit normal vector at collocation points

    x0[i],y0[i],z0[i], DxDxi,DyDxi,DzDxi,DxDet,DyDet,DzDet\
    ,vnx0[i],vny0[i],vnz0[i],hs = interp_p (points[i1,0],points[i1,1],points[i1,2]\
    ,points[i2,0],points[i2,1],points[i2,2],points[i3,0],points[i3,1],points[i3,2]\
    ,points[i4,0],points[i4,1],points[i4,2],points[i5,0],points[i5,1],points[i5,2]\
    ,points[i6,0],points[i6,1],points[i6,2],alpha[i],beta[i],gamma[i],xi,eta)


# writing csv file
dict1 = {'x':x0[0:nelm], 'y': y0[0:nelm], 'z':z0[0:nelm]} 
df = pd.DataFrame(dict1)
df.to_csv("element_collocation_points.out"+'.csv', index=False)


#-------------------------------
# Generate the influence matrix,
# three rows at a time,
# corresponding to the x, y, z 
# components of the integral equation
#------------------------------------

cf = -1.0/(pi8*visc)


RM  = np.zeros((nelm*3,nelm*3))
rhs  = np.zeros((nelm*3,10))
for i in range(nelm):           # run over the collocation points
    

    print('prtcl_3d: collocating at Node: ',i)

    inelm  = i+nelm
    inelm2 = i+nelm+nelm

    for j in range(nelm):           # run over the elements

        jnelm  = j+nelm
        jnelm2 = j+nelm+nelm

#-----------------------
        if (i != j):     # regular element
#-----------------------
            GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
            slp_trgl6(x0[i], y0[i], z0[i], j, mint, nn , \
            points, alpha, beta, gamma, \
            xiq, etq, Iflow)
        else:
            GExx, GExy, GExz, GEyx, GEyy, GEyz, GEzx, GEzy, GEzz = \
            slp_trgl6_sing(x0[i], y0[i], z0[i], j, mint, nn , \
            points, alpha, beta, gamma, NGL, zz, ww, \
            xiq, etq, Iflow)           


        RM[i,j] = cf * GExx
        RM[i,jnelm] = cf * GEyx
        RM[i,jnelm2] = cf * GEzx

        RM[inelm,j] = cf * GExy
        RM[inelm,jnelm]  = cf * GEyy
        RM[inelm,jnelm2] = cf * GEzy

        RM[inelm2,j] = cf * GExz
        RM[inelm2,jnelm] = cf * GEyz
        RM[inelm2,jnelm2] = cf * GEzz


        aj = arel[j]

        RM[i,j] = RM[i,j] + vnx0[i] * vnx0[j] * aj
        RM[i,jnelm] = RM[i,jnelm] + vnx0[i] * vny0[j] * aj
        RM[i,jnelm2] = RM[i,jnelm2] + vnx0[i] * vnz0[j] * aj

        RM[inelm,j] = RM[inelm,j] + vny0[i] * vnx0[j] * aj
        RM[inelm,jnelm] = RM[inelm,jnelm] + vny0[i] * vny0[j] * aj
        RM[inelm,jnelm2] = RM[inelm,jnelm2] + vny0[i] * vnz0[j] * aj

        RM[inelm2,j] = RM[inelm2,j] + vnz0[i] * vnx0[j] * aj
        RM[inelm2,jnelm] = RM[inelm2,jnelm] + vnz0[i] * vny0[j] * aj
        RM[inelm2,jnelm2] = RM[inelm2,jnelm2]+ vnz0[i] * vnz0[j] * aj

#------------
# system size
#------------

Mdim = 3*nelm

Mls = Mdim  # size of the linear system emerging from collocation

#-------------------------------------------------
# test for uniform flow past a solitary sphere
# in free space.
#
# In this case, the exact solution of the integral
# equation for translation
# is known to be: 
#
# traction = 1.5 visc*U/req
#------------------------------------------------


#---------------------------
# set the right-hand side(s)
#---------------------------

#----------------------------------------------
# Flow due to particle rigid-body motion
#
# The six right-hand sides
# express the six modes of translation and rotation
#----------------------------------------------


nrhs = 6           # number of right-hand sides

for i in range(Mdim):       # initialize
    for k in range(nrhs):
        rhs[i,k] = 0.0

for i in range(nelm):
    inelm  = i+nelm
    inelm2 = i+nelm+nelm

    rhs[i,0] = 1.0 # translation along the x axis
    rhs[inelm,1] = 1.0 # translation along the y axis
    rhs[inelm2,2] = 1.0 # translation along the z axis  
    xhat = x0[i]-cxp
    yhat = y0[i]-cyp
    zhat = z0[i]-czp
    #        xhat = x0(i) ! reset
    #        yhat = y0(i)
    #        zhat = z0(i)

    rhs[inelm, 3] = -zhat      # rigid-body rotation
    rhs[inelm2, 3] = yhat
    rhs[i, 4] = zhat
    rhs[inelm2, 4] = -xhat
    rhs[i, 5] = -yhat
    rhs[inelm, 5] = xhat

#------------------------
# solve the linear system
#------------------------

#------------
#     pause " Print the linear system ?"
#   for i in range(Mdim):
#       print((RM(i,j),j=1,Mdim),(rhs(i,j),j=1,nrhs))
#     
#------------
Isymg  = 0    # system is not symmetric
Iwlpvt = 1    # pivoting enabled
print(' prtcl_3d: system size: ',Mls)
ff = np.zeros((nelm*3,Mls))
aindex = np.arange(1,nelm + 1) #index for later using in input
for i in range(nrhs):
    ff[:,i] = np.dot(np. linalg. inv(RM),rhs[:,i])


# writing csv file

print(' prtcl_3d: tractions for translation modes:')
dict1 = {\
'Tx1': ff[:nelm,0], 'Ty1':ff[nelm:2*nelm,0], 'Tz1':ff[2*nelm:3*nelm,0]\
, 'Tx2': ff[:nelm,1], 'Ty2':ff[nelm:2*nelm,1], 'Tz2':ff[2*nelm:3*nelm,1]\
, 'Tx3': ff[:nelm,2], 'Ty3':ff[nelm:2*nelm,2], 'Tz3':ff[2*nelm:3*nelm,2]} 
df = pd.DataFrame(dict1)
df.to_csv("tractions_for_translation_modes"+'.csv', index=False)

print(' prtcl_3d: tractions for rotation modes:')
dict1 = { \
'Tx1': ff[:nelm,3], 'Ty1':ff[nelm:2*nelm,3], 'Tz1':ff[2*nelm:3*nelm,3]\
, 'Tx2': ff[:nelm,4], 'Ty2':ff[nelm:2*nelm,4], 'Tz2':ff[2*nelm:3*nelm,4]\
, 'Tx3': ff[:nelm,5], 'Ty3':ff[nelm:2*nelm,5], 'Tz3':ff[2*nelm:3*nelm,5]} 
df = pd.DataFrame(dict1)
df.to_csv("tractions_for_rotational_modes"+'.csv', index=False)

#------------------------------------
# Assign traction to elements
#
# Compute force, torque, 
#------------------------------------


cf = -1.0/(pi6*req*visc)

Resist = np.zeros((6,6))
fx = np.zeros(nelm)
fy = np.zeros(nelm)
fz = np.zeros(nelm)

for k in range(nrhs):

    Frcx = 0.0
    Frcy = 0.0
    Frcz = 0.0

    Trqx = 0.0
    Trqy = 0.0
    Trqz = 0.0

    for i in range(nelm):

        inelm  = i + nelm
        inelm2 = inelm + nelm

        fx[i] = ff[i,k]
        fy[i] = ff[inelm,k]
        fz[i] = ff[inelm2,k]

        Frcx = Frcx + fx[i] * arel[i]
        Frcy = Frcy + fy[i] * arel[i]
        Frcz = Frcz + fz[i] * arel[i]

        Trqx = Trqx + fz[i] * ymom[i]- fy[i] * zmom[i]
        Trqy = Trqy + fx[i] * zmom[i]- fz[i] * xmom[i]
        Trqz = Trqz + fy[i] * xmom[i]- fx[i] * ymom[i]
        
    Trcx = Trqx - Frcz * cy + Frcy * cz
    Trcy = Trqy - Frcx * cz + Frcz * cx
    Trcz = Trqz - Frcy * cx + Frcx * cy

    Resist[0,k] = np.round(cf * Frcx,5)
    Resist[1,k] = np.round(cf * Frcy,5)
    Resist[2,k] = np.round(cf * Frcz,5)

    Resist[3,k] = np.round(cf * Trcx ,5)
    Resist[4,k] = np.round(cf * Trcy ,5)
    Resist[5,k] = np.round(cf * Trcz,5)
pd.DataFrame(Resist).to_csv("resistance.csv", header =False , index=False)

print('The program is finished.')










