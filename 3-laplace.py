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


iflow = int(float(sys.argv[1])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
Ioctaicos = int(float(sys.argv[2])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
#Peclete Number
ndiv = int(sys.argv[3]) # level of discretization of starting octahedron
#number of particles per lattice row
cxp = float(float(sys.argv[4])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
#Peclete Number
wall = float(sys.argv[5]) # level of discretization of starting octahedron






# with axes: a, b where b<a the axes ratio b/a less than 1
req = 1.0 # equivalent radius
#cxp = 0.0
cyp,czp = 0.0, 0.0 #particle centers
#rotation angles: phix, phiy,phiz
mint = 7 #mint number of Gauss_triangle base points for regular integrals 7 or 9
NGL = 6 #number of Gauss_Legendre base points for singular integrals
cf = -0.5

# "iflow=  1 for flow due to the motion of particle"
# "   in free space"
# 2 for flow due to the motion of particle"
# "   in a semi-infinite domain bounded"
# "   by a plane located at y=wall"




# reading csv file
unit_p = pd.read_csv('unit_sphere_points.csv', header=None, index_col=None).values
ne = pd.read_csv('ne.csv', header=None, index_col=None).values
nn = pd.read_csv('nn.csv', header=None, index_col=None).values
nbe = pd.read_csv('nbe.csv', header=None, index_col=None).values

npts = len(unit_p)
nelm = len(nn)
print('number of points:', npts)
print('number of elements:', nelm)

for j in range(npts):
    tem = ne[j,0] +1
    ne[j,1:tem] = ne[j,1:tem] - 1
nbe = nbe - 1
nn = nn - 1
num_points = len(unit_p)
num_elements = len(nn)


q = np.zeros(num_points)
dlp = np.zeros(num_points)
dfdn  = np.zeros(num_points)
dfdn[unit_p[:,2]>0.0] = -1.0
dfdn[unit_p[:,2]<0.0] = 1.0
dfdn[unit_p[:,2]==0.0] = 0.0
amat = np.zeros((npts,npts))

points = pd.read_csv('geometry.csv', header=None, index_col=None).values
points[:,0] = points[:,0] + cxp
#---------------
# prepare to run
#---------------
alpha = np.zeros(num_elements) #alpha
beta = np.zeros(num_elements) #beta
gamma = np.zeros(num_elements) #gamma

#  x0, y0, z0         coordinates of collocation points
#  vnx0, vny0, vnz0   unit normal vector at collocation points
vnx0 = np.zeros(num_elements)
vny0 = np.zeros(num_elements)
vnz0 = np.zeros(num_elements)
x0 = np.zeros(num_elements)
y0 =  np.zeros(num_elements)
z0 = np.zeros(num_elements)
nelm2 = 2 * nelm

def ldlp_3d_interp(            
        x1,y1,z1,                
        x2,y2,z2,                
        x3,y3,z3,                
        x4,y4,z4,                
        x5,y5,z5,                
        x6,y6,z6,                
        vx1,vy1,vz1,             
        vx2,vy2,vz2,             
        vx3,vy3,vz3,             
        vx4,vy4,vz4,             
        vx5,vy5,vz5,             
        vx6,vy6,vz6,             
        q1,q2,q3,q4,q5,q6,       
        al,be,ga,                
        xi,eta,                  
        ):

    #-----------------------------------------
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #-------------------------------------------
    #   Utility of the Biot-Savart integrator:
    #
    #   Interpolates over an element for the:
    #
    #   position vector
    #   normal vector
    #   the scalar density q
    #   the surface metric
    #
    #   Iopt_int = 1 only the position vector
    #              2 position vector and rest of variables
    #-------------------------------------------



    #--------
    # prepare
    #--------

    alc = 1.0-al
    bec = 1.0-be
    gac = 1.0-ga

    alalc = al*alc
    bebec = be*bec
    gagac = ga*gac

    #----------------------------
    # compute the basis functions
    #----------------------------

    ph2 = xi *(xi -al+eta*(al-ga)/gac)/alc
    ph3 = eta*(eta-be+xi *(be+ga-1.0)/ga)/bec
    ph4 = xi *(1.0-xi-eta)/alalc
    ph5 = xi*eta          /gagac
    ph6 = eta*(1.0-xi-eta)/bebec
    ph1 = 1.0-ph2-ph3-ph4-ph5-ph6

    #--------------------------------
    # interpolate the position vector
    #--------------------------------

    x = x1*ph1 + x2*ph2 + x3*ph3 + x4*ph4 + x5*ph5 + x6*ph6
    y = y1*ph1 + y2*ph2 + y3*ph3 + y4*ph4+ y5*ph5 + y6*ph6
    z = z1*ph1 + z2*ph2 + z3*ph3 + z4*ph4+ z5*ph5 + z6*ph6

    #-------------------------------
    # interpolate the scalar density
    #-------------------------------

    qint = q1*ph1 + q2*ph2 + q3*ph3+ q4*ph4 + q5*ph5 + q6*ph6



    #----------------------------------
    # interpolate for the normal vector
    #----------------------------------

    vx = vx1*ph1 +vx2*ph2 +vx3*ph3 +vx4*ph4+ vx5*ph5 +vx6*ph6
    vy = vy1*ph1 +vy2*ph2 +vy3*ph3 +vy4*ph4 + vy5*ph5 +vy6*ph6
    vz = vz1*ph1 +vz2*ph2 +vz3*ph3 +vz4*ph4+ vz5*ph5 +vz6*ph6

    #------------------------------------------
    # compute xi derivatives of basis functions
    #------------------------------------------

    dph2 =  (2.0*xi-al+eta*(al-ga)/gac)/alc
    dph3 =  eta*(be+ga-1.0)/(ga*bec)
    dph4 =  (1.0-2.0*xi-eta)/alalc
    dph5 =  eta/gagac
    dph6 = -eta/bebec
    dph1 = -dph2-dph3-dph4-dph5-dph6

    #----------------------------------------------
    # compute xi derivatives of the position vector
    #----------------------------------------------

    DxDxi = x1*dph1 + x2*dph2 + x3*dph3 + x4*dph4 + x5*dph5 + x6*dph6
    DyDxi = y1*dph1 + y2*dph2 + y3*dph3 + y4*dph4+ y5*dph5 + y6*dph6
    DzDxi = z1*dph1 + z2*dph2 + z3*dph3 + z4*dph4+ z5*dph5 + z6*dph6

    #-------------------------------------------
    # compute eta derivatives of basis functions
    #-------------------------------------------

    pph2 =  xi*(al-ga)/(alc*gac)
    pph3 =  (2.0*eta-be+xi*(be+ga-1.0)/ga)/bec
    pph4 = -xi/alalc
    pph5 =  xi/gagac
    pph6 =  (1.0-xi-2.0*eta)/bebec
    pph1 = -pph2-pph3-pph4-pph5-pph6

    #-----------------------------------------------
    # compute eta derivatives of the position vector
    #-----------------------------------------------

    DxDet = x1*pph1 + x2*pph2 + x3*pph3 + x4*pph4+ x5*pph5 + x6*pph6
    DyDet = y1*pph1 + y2*pph2 + y3*pph3 + y4*pph4 + y5*pph5 + y6*pph6
    DzDet = z1*pph1 + z2*pph2 + z3*pph3 + z4*pph4+ z5*pph5 + z6*pph6

    #---------------------------------------
    #  compute the raw normal vector and the
    #  surface metric hs
    #---------------------------------------

    vnxr = DyDxi * DzDet - DyDet * DzDxi
    vnyr = DzDxi * DxDet - DzDet * DxDxi
    vnzr = DxDxi * DyDet - DxDet * DyDxi

    hs = np.sqrt(vnxr**2+vnyr**2+vnzr**2 )



    return x,y,z,vx,vy,vz,hs,qint
    







def ldlp_3d_integral(npts, x0, y0, z0,ipoint, k, wall, mint,Iflow,q,q0):

    #-----------------------------------------
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #--------------------------------------
    # Compute the double-layer laplace potential
    # over a non-singular triangle numbered k
    #--------------------------------------



    #---
    # COMMON blocks
    #---


    #-----------
    # initialize
    #-----------



    area = 0.0
    ptl  = 0.0

    #---
    # define triangle nodes
    # and mapping parameters
    #---

    i1 = nn[k,0]
    i2 = nn[k,1]
    i3 = nn[k,2]
    i4 = nn[k,3]
    i5 = nn[k,4]
    i6 = nn[k,5]

    al = alpha[k]
    be = beta [k]
    ga = gamma[k]

    #---
    # loop over integration points
    #---


    for i in range(mint):

        xi  = xiq[i]
        eta = etq[i]

        x, y, z,vnx, vny, vnz,hs,qint = ldlp_3d_interp(
            points[i1,0], points[i1,1], points[i1,2],
            points[i2,0], points[i2,1], points[i2,2],
            points[i3,0], points[i3,1], points[i3,2],
            points[i4,0], points[i4,1], points[i4,2],
            points[i5,0], points[i5,1], points[i5,2],
            points[i6,0], points[i6,1], points[i6,2],
            vna[i1,0], vna[i1,1], vna[i1,2],
            vna[i2,0], vna[i2,1], vna[i2,2],
            vna[i3,0], vna[i3,1], vna[i3,2],
            vna[i4,0], vna[i4,1], vna[i4,2],
            vna[i5,0], vna[i5,1], vna[i5,2],
            vna[i6,0], vna[i6,1], vna[i6,2],
            q[i1], q[i2], q[i3], q[i4], q[i5], q[i6],
            al, be, ga,
            xi, eta,
            )

        #---
        # compute the Green's function
        #---

        if (Iflow == 1):

            G,Gx,Gy,Gz = lgf_3d_fs (x,y,z,x0,y0,z0)

        elif (Iflow == 2):


			#If the wall is moving in x direction
            G,Gx,Gy,Gz = lgf_3d_w_x(x,y,z,x0,y0,z0,wall)
			
			#If the wall is moving in y direction
			#G,Gx,Gy,Gz = lgf_3d_w_y(x,y,z,x0,y0,z0,wall)
			
			#If the wall is moving in z direction
			#G,Gx,Gy,Gz = lgf_3d_w_z(x,y,z,x0,y0,z0,wall)

        #---
        # apply the triangle quadrature
        #---

        cf = 0.5*hs*wq[i]
        area = area + cf

        ptl = ptl + (qint-q0)*(vnx*Gx+vny*Gy+vnz*Gz)*cf



    return ptl, area

    
    
    
    
    

def ldlp_3d(npts,nelm,mint,q,Iflow):

    #-----------------------------------------
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #----------------------------------
    # Computes the principal value of the
    # double-layer potential
    # of a scalar function q
    # at the nodes of a triangular grid
    # on a closed surface
    #----------------------------------
    dlp = np.zeros(npts)


    tol=0.00000001

    #--------------
    # COMMON blocks
    #--------------

    #----------------------
    # launch the quadrature
    #----------------------

    for i in range(npts):


        x0 = points[i,0]
        y0 = points[i,1]
        z0 = points[i,2]

        q0 = q[i]

        #-----------
        # initialize
        #-----------

        srf_area = 0.0
        ptl      = 0.0

        #----------------------
        # Compile the integrals
        # over the triangles
        #---------------------

        for k in range(nelm):



            i1 = nn[k,0]
            i2 = nn[k,1]
            i3 = nn[k,2]
            i4 = nn[k,3]
            i5 = nn[k,4]
            i6 = nn[k,5]

            test = np.abs(q[i1]) + np.abs(q[i2]) + np.abs(q[i3])+ np.abs(q[i4])+np.abs(q[i5])+np.abs(q[i6])+ np.abs(q0)

            if(test>tol):

            #---
            # apply the quadrature
            #---

                pptl,arelm = ldlp_3d_integral (npts,x0,y0,z0,i,k,wall,mint,Iflow, q, q0)

                ptl = ptl+pptl

                srf_area = srf_area+arelm


        dlp[i] = ptl - 0.50 * q0





    return dlp





def lslp_3d_interp(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4,x5,y5,z5,x6,y6,z6,f1,f2,f3,f4,f5,f6,al,be,ga,xi_c,eta_c):

    #-----------------------------------------
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #-------------------------------------------
    # Utility of the Laplace single-layer integrator:
    #
    # Interpolates over an element for:
    #
    #   the position vector
    #   the surface metric
    #   the scalar density f
    #
    #
    #   Iopt_int = 1 only the position vector
    #              2 position vector and rest of variables
    #-------------------------------------------



    #--------
    # prepare
    #--------

    alc = 1.0-al
    bec = 1.0-be
    gac = 1.0-ga

    alalc = al*alc
    bebec = be*bec
    gagac = ga*gac

    #----------------------------
    # compute the basis functions
    #----------------------------

    ph2 = xi_c *(xi_c -al+eta_c*(al-ga)/gac)   /alc
    ph3 = eta_c*(eta_c-be+xi_c *(be+ga-1.0)/ga)/bec
    ph4 = xi_c *(1.0-xi_c-eta_c)/alalc
    ph5 = xi_c*eta_c          /gagac
    ph6 = eta_c*(1.0-xi_c-eta_c)/bebec
    ph1 = 1.0-ph2-ph3-ph4-ph5-ph6

    #--------------------------------
    # interpolate the position vector
    #--------------------------------

    x = x1*ph1 + x2*ph2 + x3*ph3 + x4*ph4 + x5*ph5 + x6*ph6
    y = y1*ph1 + y2*ph2 + y3*ph3 + y4*ph4 + y5*ph5 + y6*ph6
    z = z1*ph1 + z2*ph2 + z3*ph3+ z4*ph4 + z5*ph5 + z6*ph6

    #--------------------------
    # interpolate the density f
    #--------------------------

    f = f1*ph1 + f2*ph2 + f3*ph3 + f4*ph4 + f5*ph5 + f6*ph6

    #------------------------------------------
    # compute xi_c derivatives of basis functions
    #------------------------------------------

    dph2 =  (2.0*xi_c-al+eta_c*(al-ga)/gac)/alc
    dph3 =  eta_c*(be+ga-1.0)/(ga*bec)
    dph4 =  (1.0-2.0*xi_c-eta_c)/alalc
    dph5 =  eta_c/gagac
    dph6 = -eta_c/bebec
    dph1 = -dph2-dph3-dph4-dph5-dph6

    #----------------------------------------------
    # compute xi_c derivatives of the position vector
    #----------------------------------------------

    DxDxi = x1*dph1 + x2*dph2 + x3*dph3 + x4*dph4+ x5*dph5 + x6*dph6
    DyDxi = y1*dph1 + y2*dph2 + y3*dph3 + y4*dph4 + y5*dph5 + y6*dph6
    DzDxi = z1*dph1 + z2*dph2 + z3*dph3 + z4*dph4 + z5*dph5 + z6*dph6

    #-------------------------------------------
    # compute eta_c derivatives of basis functions
    #-------------------------------------------

    pph2 =  xi_c*(al-ga)/(alc*gac)
    pph3 =  (2.0*eta_c-be+xi_c*(be+ga-1.0)/ga)/bec
    pph4 = -xi_c/alalc
    pph5 =  xi_c/gagac
    pph6 =  (1.0-xi_c-2.0*eta_c)/bebec
    pph1 = -pph2-pph3-pph4-pph5-pph6

    #-----------------------------------------------
    # compute eta_c derivatives of the position vector
    #-----------------------------------------------

    DxDet = x1*pph1 + x2*pph2 + x3*pph3 + x4*pph4+ x5*pph5 + x6*pph6
    DyDet = y1*pph1 + y2*pph2 + y3*pph3 + y4*pph4+ y5*pph5 + y6*pph6
    DzDet = z1*pph1 + z2*pph2 + z3*pph3 + z4*pph4+ z5*pph5 + z6*pph6

    #--------------------------------------
    # compute the raw normal vector and the
    # surface metric hs
    #--------------------------------------

    vnxr = DyDxi * DzDet - DyDet * DzDxi
    vnyr = DzDxi * DxDet - DzDet * DxDxi
    vnzr = DxDxi * DyDet - DxDet * DyDxi

    hs = np.sqrt(vnxr**2+vnyr**2+vnzr**2 )

    #-----
    # Done
    #-----



    return x,y,z,hs,f


def lslp_3d_integral(npts, x0,y0,z0, wall, k,mint,Iflow,f):

    #-----------------------------------------
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #--------------------------------------
    # Compute the Laplace single-layer potential
    # over a non-singular triangle numbered k
    #--------------------------------------




    #---
    # initialize
    #---


    area = 0.0
    slp  = 0.0

    #---
    # define triangle nodes
    # and mapping parameters
    #---

    i1 = nn[k,0]
    i2 = nn[k,1]
    i3 = nn[k,2]
    i4 = nn[k,3]
    i5 = nn[k,4]
    i6 = nn[k,5]

    al = alpha[k]
    be = beta [k]
    ga = gamma[k]

    #---
    # loop over integration points
    #---

    for i in range(mint):

        xi_c  = xiq[i]
        eta_c = etq[i]

        x,y,z,hs,fint = lslp_3d_interp(points[i1,0],points[i1,1],points[i1,2], \
                       points[i2,0],points[i2,1],points[i2,2],\
                       points[i3,0],points[i3,1],points[i3,2],\
                       points[i4,0],points[i4,1],points[i4,2],\
                       points[i5,0],points[i5,1],points[i5,2],\
                       points[i6,0],points[i6,1],points[i6,2],\
                       f[i1],f[i2],f[i3],f[i4],f[i5],f[i6],al,be,ga,xi_c,eta_c)

        #-----------------------------
        # compute the Green's function
        #-----------------------------

        if (Iflow == 1):

            G,Gx,Gy,Gz = lgf_3d_fs (x,y,z,x0,y0,z0)

        elif (Iflow == 2):
            #If the wall is moving in x direction
            G,Gx,Gy,Gz = lgf_3d_w_x(x,y,z,x0,y0,z0,wall)

            #If the wall is moving in y direction
            #G,Gx,Gy,Gz = lgf_3d_w_y(,x,y,z,x0,y0,z0,wall)

            #If the wall is moving in z direction
            #G,Gx,Gy,Gz = lgf_3d_w_z(x,y,z,x0,y0,z0,wall)


        #------------------------------
        # apply the triangle quadrature
        #------------------------------

        cf = 0.5 *hs*wq[i]

        area = area+ cf

        slp = slp + fint*G*cf



    return slp,area



def lgf_3d_w_z(x,y,z,x0,y0,z0,wall):

    #-----------------------------------------
    # FDLIB, BEMLIB
    #
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #---------------------------------------
    # Green's function of Laplace's equation
    # in a semi-infinite domain
    # bounded by a plane wall located at z = wall
    #
    #  G = 1/(4*pi*r) (+-)  1/(4*pi*r_im)
    #
    #----------------------------------------



    #--------
    # prepare
    #--------

    #sign = 1.0 #for Green function
    sign = -1.0 #for Neumann function

    #--------------------
    # primary singularity
    #--------------------

    dx = x-x0
    dy = y-y0
    dz = z-z0

    r = np.sqrt(dx**2+dy**2+dz**2)

    G = 1.0/(np.pi * 4 *r)

    #------------------
    # image singularity
    #------------------

    #x0i = x0
    #y0i = y0
    z0i = 2.0*wall - z0

    dxi = dx
    dyi = dy
    dzi = z-z0i

    ri = np.sqrt(dxi**2+dyi**2+dzi**2)

    G = G - sign/(np.pi * 4*ri)


    den  = r**3  *np.pi * 4
    deni = ri**3 *np.pi * 4

    Gx = - dx/den + sign * dxi/deni
    Gy = - dy/den + sign * dyi/deni
    Gz = - dz/den + sign * dzi/deni


    return G,Gx,Gy,Gz
    
    
    

def lgf_3d_w_y(x,y,z,x0,y0,z0,wall):

    #-----------------------------------------
    # FDLIB, BEMLIB
    #
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #---------------------------------------
    # Green's function of Laplace's equation
    # in a semi-infinite domain
    # bounded by a plane wall located at y = wall
    #
    #  G = 1/(4*pi*r) (+-)  1/(4*pi*r_im)
    #
    #----------------------------------------



    #--------
    # prepare
    #--------

    #sign = 1.0 #for Green function
    sign = -1.0 #for Neumann function

    #--------------------
    # primary singularity
    #--------------------

    dx = x-x0
    dy = y-y0
    dz = z-z0

    r = np.sqrt(dx**2+dy**2+dz**2)

    G = 1.0/(np.pi * 4 *r)

    #------------------
    # image singularity
    #------------------

    #x0i = x0
    y0i = 2.0*wall - y0
    #z0i = z0

    dxi = dx
    dyi = y - y0i
    dzi = dz

    ri = np.sqrt(dxi**2+dyi**2+dzi**2)

    G = G - sign/(np.pi * 4*ri)


    den  = r**3  *np.pi * 4
    deni = ri**3 *np.pi * 4

    Gx = - dx/den + sign * dxi/deni
    Gy = - dy/den + sign * dyi/deni
    Gz = - dz/den + sign * dzi/deni


    return G,Gx,Gy,Gz
    
    
    
def lgf_3d_w_x(x,y,z,x0,y0,z0,wall):

    #-----------------------------------------
    # FDLIB, BEMLIB
    #
    # Copyright by C. Pozrikidis, 1999
    # All rights reserved.
    #
    # This program is to be used only under the
    # stipulations of the licensing agreement.
    #----------------------------------------

    #---------------------------------------
    # Green's function of Laplace's equation
    # in a semi-infinite domain
    # bounded by a plane wall located at x = wall
    #
    #  G = 1/(4*pi*r) (+-)  1/(4*pi*r_im)
    #
    #
    #----------------------------------------



    #--------
    # prepare
    #--------

    #sign = 1.0 #for Green function
    sign = -1.0 #for Neumann function

    #--------------------
    # primary singularity
    #--------------------

    dx = x-x0
    dy = y-y0
    dz = z-z0

    r = np.sqrt(dx**2+dy**2+dz**2)

    G = 1.0/(np.pi * 4 *r)

    #------------------
    # image singularity
    #------------------

    x0i = 2.0*wall - x0
    #y0i = y0
    #z0i = z0

    dxi = x-x0i
    dyi = dy
    dzi = dz

    ri = np.sqrt(dxi**2+dyi**2+dzi**2)

    G = G - sign/(np.pi * 4*ri)


    den  = r**3  *np.pi * 4
    deni = ri**3 *np.pi * 4

    Gx = - dx/den + sign * dxi/deni
    Gy = - dy/den + sign * dyi/deni
    Gz = - dz/den + sign * dzi/deni


    return G,Gx,Gy,Gz



def lgf_3d_fs (x,y,z,x0,y0,z0):

    #------

    dx = x-x0
    dy = y-y0
    dz = z-z0

    r = np.sqrt(dx**2+dy**2+dz**2)

    G = 1.0/(np.pi * 4.0 *r)

    den = np.pi * 4.0 * r**3

    Gx = - dx/den
    Gy = - dy/den
    Gz = - dz/den
    return G,Gx,Gy,Gz

def lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall):






    #------
    # flags
    #------



    #---
    # compute triangle area and surface metric
    #---

    #dx = np.sqrt( (x2-x1)**2+(y2-y1)**2+(z2-z1)**2 )
    #dy = np.sqrt( (x3-x1)**2+(y3-y1)**2+(z3-z1)**2 )

    vnx = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
    vny = (z2-z1)*(x3-x1) - (x2-x1)*(z3-z1)
    vnz = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

    area = 0.5 * np.sqrt( vnx**2+vny**2+vnz**2 )
    hs   = 2.0 * area

    vnx = vnx/hs
    vny = vny/hs
    vnz = vnz/hs

    #-----------
    # initialize
    #-----------

    asm = 0.0
    slp = 0.0

    #---------------------------
    # Double Gaussian quadrature
    #---------------------------

    for i in range(NGL):


        ph    = np.pi/4.0*(1.0+zz[i])
        cph   = np.cos(ph)
        sph   = np.sin(ph)
        rmax  = 1.0/(cph+sph)
        rmaxh = 0.5*rmax

        bsm = 0.0
        btl = 0.0

        for j in range(NGL):

            r= rmaxh*(1.0+zz[j])

            xi = r*cph
            et = r*sph
            zt = 1.0-xi-et

            x = x1*zt + x2*xi + x3*et
            y = y1*zt + y2*xi + y3*et
            z = z1*zt + z2*xi + z3*et
            f = f1*zt + f2*xi + f3*et


            if (iflow == 1) :

                G,Gx,Gy,Gz =  lgf_3d_fs (x,y,z,x1,y1,z1)

            elif (iflow == 2) :

                #if the wall is moving in x direction
                G,Gx,Gy,Gz = lgf_3d_w_x(x,y,z,x1,y1,z1,wall)
                
                #if the wall is moving in y direction
                #G,Gx,Gy,Gz =lgf_3d_w_y(x,y,z,x1,y1,z1,wall)
                
                #if the wall is moving in z direction
                #G,Gx,Gy,Gz = lgf_3d_w_z(x,y,z,x1,y1,z1,wall)				


            cf = r*ww[j]

            bsm = bsm + cf
            btl = btl + f*G*cf


        cf = ww[i]*rmaxh

        asm = asm + bsm*cf
        slp = slp + btl*cf


    #---
    # finish up the quadrature
    #---

    cf = np.pi/4.0*hs

    asm = asm*cf
    slp = slp*cf


    return slp,area






def lslp_3d(npts,nelm,mint,NGL,wall,f,iflow):
    rhs = np.zeros(npts)
    tol = 0.0000001


    for i in range(npts):


        x0 = points[i,0]
        y0 = points[i,1]
        z0 = points[i,2]

        #-----------
        # initialize
        #-----------

        srf_area = 0.0
        ptl      = 0.0

        #----------------------
        # Compile the integrals
        # over the triangles
        #---------------------

        for k in range(nelm): # run over elements

            #      WRITE (6,*) " Integrating over element",k

            i1 = nn[k,0]
            i2 = nn[k,1]
            i3 = nn[k,2]
            i4 = nn[k,3]
            i5 = nn[k,4]
            i6 = nn[k,5]

            #---
            # integration will be performed only
            # if f is nonzero at least at one node
            #---

            test = np.abs(f[i1])+np.abs(f[i2])+np.abs(f[i3])+ np.abs(f[i4])+np.abs(f[i5])+np.abs(f[i6])

            if (test>tol):   # if test= 0 skip the integration over this triangle

			#-------------------------------------------
			# singular integration:
			# --------------------
			#
			# if the point i is a vertex node,
			# integrate over the flat triangle
			# defined by the node
			# using the polar integration rule
			#
			# if the point i is a mid-node,
			# breakup the curved triangle into four
			# flat triangles and integrate over the flat triangles
			# using the polar integration rule
			#  
			#   Iopt_int = 1 only the position vector
			#              2 position vector and rest of variables
			#
			# non-singular integration:
			# ------------------------
			#
			# use a regular triangle quadrature
			#--------------------------------------------

			#--------------------------------------------
			# singular element with singularity at node 1
			# this is a vertex node
			#--------------------------------------------


                if(i == i1):

                    x1 = points[i1,0]
                    y1 = points[i1,1]
                    z1 = points[i1,2]
                    f1 = f[i1]

                    x2 = points[i2,0]
                    y2 = points[i2,1]
                    z2 = points[i2,2]
                    f2 = f[i2]

                    x3 = points[i3,0]
                    y3 = points[i3,1]
                    z3 = points[i3,2]
                    f3 = f[i3]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

				#--------------------------------------------
				# singular element with singularity at node 2
				# vertex node
				#--------------------------------------------

                elif(i == i2):

                    x1 = points[i2,0]
                    y1 = points[i2,1]
                    z1 = points[i2,2]
                    f1 = f[i2]

                    x2 = points[i3,0]
                    y2 = points[i3,1]
                    z2 = points[i3,2]
                    f2 = f[i3]

                    x3 = points[i1,0]
                    y3 = points[i1,1]
                    z3 = points[i1,2]
                    f3 = f[i1]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

				#--------------------------------------------
				# singular element with singularity at node 3
				# vertex node
				#--------------------------------------------

                elif(i == i3):

                    x1 = points[i3,0]
                    y1 = points[i3,1]
                    z1 = points[i3,2]
                    f1 = f[i3]

                    x2 = points[i1,0]
                    y2 = points[i1,1]
                    z2 = points[i1,2]
                    f2 = f[i1]

                    x3 = points[i2,0]
                    y3 = points[i2,1]
                    z3 = points[i2,2]
                    f3 = f[i2]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

				#--------------------------------------------
				# singular element with singularity at node 4
				# mid-node
				# Will integrate over 4 flat triangles
				#--------------------------------------------

                elif(i == i4):

                    x1 = points[i4,0]
                    y1 = points[i4,1]
                    z1 = points[i4,2]
                    f1 = f[i4]

                    x2 = points[i6,0]
                    y2 = points[i6,1]
                    z2 = points[i6,2]
                    f2 = f[i6]

                    x3 = points[i1,0]
                    y3 = points[i1,1]
                    z3 = points[i1,2]
                    f3 = f[i1]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i4,0]
                    y1 = points[i4,1]
                    z1 = points[i4,2]
                    f1 = f[i4]

                    x2 = points[i3,0]
                    y2 = points[i3,1]
                    z2 = points[i3,2]
                    f2 = f[i3]

                    x3 = points[i6,0]
                    y3 = points[i6,1]
                    z3 = points[i6,2]
                    f3 = f[i6]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i4,0]
                    y1 = points[i4,1]
                    z1 = points[i4,2]
                    f1 = f[i4]

                    x2 = points[i5,0]
                    y2 = points[i5,1]
                    z2 = points[i5,2]
                    f2 = f[i5]

                    x3 = points[i3,0]
                    y3 = points[i3,1]
                    z3 = points[i3,2]
                    f3 = f[i3]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i4,0]
                    y1 = points[i4,1]
                    z1 = points[i4,2]
                    f1 = f[i4]

                    x2 = points[i2,0]
                    y2 = points[i2,1]
                    z2 = points[i2,2]
                    f2 = f[i2]

                    x3 = points[i5,0]
                    y3 = points[i5,1]
                    z3 = points[i5,2]
                    f3 = f[i5]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

				#--------------------------------------------
				# singular element with singularity at node 5
				# mid-node
				# Will integrate over 4 flat triangles
				#--------------------------------------------

                elif(i == i5):

                    x1 = points[i5,0]
                    y1 = points[i5,1]
                    z1 = points[i5,2]
                    f1 = f[i5]

                    x2 = points[i4,0]
                    y2 = points[i4,1]
                    z2 = points[i4,2]
                    f2 = f[i4]

                    x3 = points[i2,0]
                    y3 = points[i2,1]
                    z3 = points[i2,2]
                    f3 = f[i2]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i5,0]
                    y1 = points[i5,1]
                    z1 = points[i5,2]
                    f1 = f[i5]

                    x2 = points[i1,0]
                    y2 = points[i1,1]
                    z2 = points[i1,2]
                    f2 = f[i1]

                    x3 = points[i4,0]
                    y3 = points[i4,1]
                    z3 = points[i4,2]
                    f3 = f[i4]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i5,0]
                    y1 = points[i5,1]
                    z1 = points[i5,2]
                    f1 = f[i5]

                    x2 = points[i6,0]
                    y2 = points[i6,1]
                    z2 = points[i6,2]
                    f2 = f[i6]

                    x3 = points[i1,0]
                    y3 = points[i1,1]
                    z3 = points[i1,2]
                    f3 = f[i1]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i5,0]
                    y1 = points[i5,1]
                    z1 = points[i5,2]
                    f1 = f[i5]

                    x2 = points[i3,0]
                    y2 = points[i3,1]
                    z2 = points[i3,2]
                    f2 = f[i3]

                    x3 = points[i6,0]
                    y3 = points[i6,1]
                    z3 = points[i6,2]
                    f3 = f[i6]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

				#--------------------------------------------
				# singular element with singularity at node 6
				# mid-node
				# Will integrate over 4 flat triangles
				#--------------------------------------------

                elif(i == i6):

                    x1 = points[i6,0]
                    y1 = points[i6,1]
                    z1 = points[i6,2]
                    f1 = f[i6]

                    x2 = points[i1,0]
                    y2 = points[i1,1]
                    z2 = points[i1,2]
                    f2 = f[i1]

                    x3 = points[i4,0]
                    y3 = points[i4,1]
                    z3 = points[i4,2]
                    f3 = f[i4]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i6,0]
                    y1 = points[i6,1]
                    z1 = points[i6,2]
                    f1 = f[i6]

                    x2 = points[i4,0]
                    y2 = points[i4,1]
                    z2 = points[i4,2]
                    f2 = f[i4]

                    x3 = points[i2,0]
                    y3 = points[i2,1]
                    z3 = points[i2,2]
                    f3 = f[i2]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i6,0]
                    y1 = points[i6,1]
                    z1 = points[i6,2]
                    f1 = f[i6]

                    x2 = points[i2,0]
                    y2 = points[i2,1]
                    z2 = points[i2,2]
                    f2 = f[i2]

                    x3 = points[i5,0]
                    y3 = points[i5,1]
                    z3 = points[i5,2]
                    f3 = f[i5]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm

                    x1 = points[i6,0]
                    y1 = points[i6,1]
                    z1 = points[i6,2]
                    f1 = f[i6]

                    x2 = points[i5,0]
                    y2 = points[i5,1]
                    z2 = points[i5,2]
                    f2 = f[i5]

                    x3 = points[i3,0]
                    y3 = points[i3,1]
                    z3 = points[i3,2]
                    f3 = f[i3]

                    pptl,arelm = lslp_3d_integral_sing(NGL,iflow,x1,y1,z1,x2,y2,z2,x3,y3,z3,f1,f2,f3,wall)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm



				#--------------------
				# regular integration
				#---------------------

                else:

                    pptl,arelm = lslp_3d_integral(npts, x0,y0,z0,wall, k,mint,iflow,f)

                    ptl = ptl + pptl

                    srf_area = srf_area+arelm


		
        rhs[i] = ptl



    

    return rhs
 




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
#  3] line and surface metrics
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

area = area/(np.pi*4*req**2)          # normalize
vlm  = 3.0*vlm /(np.pi*4*req**2)    # normalize



#---
# proceed with the impulses
#---
print('before rhs')
rhs = lslp_3d(npts,nelm,mint,NGL,wall,dfdn,iflow)
print('after rhs')
for j in range(npts):

    print('j',j)
    q[j] = 1.0   #impulse

    
    dlp = ldlp_3d(npts, nelm, mint, q, iflow)
    amat[:,j] = dlp 

    q[j] = 0.0    # reset

amat = amat +cf * np.eye(npts)



ff = np.dot(np. linalg. inv(amat),rhs)


dict1 = {\
'x': points[:,0], 'y':points[:,1], 'z':points[:,2]\
, 'C': ff, 'dfdn':dfdn} 
df = pd.DataFrame(dict1)
df.to_csv("concentration"+'.csv', index=False)







































