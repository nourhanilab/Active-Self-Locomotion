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
#  3) Uniform flow past
#     a triply periodic lattice of rigid particles
#     (Iflow=3)
#
#  4) Uniform flow normal to a doubly periodic array of particles 
#     representing a screen
#     (Iflow=4)
#
#  5) Shear flow over a doubly periodic array of particles 
#     representing a screen.
#     (Iflow=5)
#
#  6) Shear flow past a spherical bump on a plane wall
#     located at y=wall
#     (Iflow=6)
#
#  7) Shear flow past a doubly periodic array of particles 
#     above a wall located at y=wall
#     (Iflow=7)
#
#  In cases 1--5, 7 the particle is an ellipsoid with:
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

Ioctaicos = int(float(sys.argv[1])) #Ioctaicos: 1 for octahadron, 2 for icosahedron
# Peclete Number
ndiv = int(sys.argv[2]) # level of discretization of starting octahedron
int_ext = 2 # 1 for the interior; 2 for the exterior problem

boa,coa = 1.0, 1.0 # b/a and c/a The particle is a prolate spheroid" 
# with axes: a, b where b<a the axes ratio b/a less than 1
req = 1.0 # equivalent radius
cxp,cyp,czp = 0.0, 0.0, 0.0 #particle centers
phi1,phi2,phi3 = 0.0, 0.0, 0.0 #rotation angles: phix, phiy,phiz 0.20 0.10 -0.50

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
    if (ndiv == 4):
        num_points = int(10242)
        num_elements = int(5120)
        num_3elements = int(15360)
        
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
 


 
    

def trgl6_octa(ndiv):
#----------------------------------------
#  0'th level discretization (8 elements)
#  nodes are set manually
#----------------------------------------
    nn= np.zeros((num_elements,6), dtype=int) #related to elements: n(i,j) .... node number of point j on element i, where j=1,...,6
    nbe = np.zeros((num_elements,3), dtype=int) #related to elements nbe(i,j): label of element sharing side j of element i where j = 1, 2, 3
    ne = np.zeros((num_points,7), dtype=int) # ne(i,j) ... ne(i,1) is the number of elements touching node i. ne(i,2:ne(i,1)) 
    # are the corresponding element
    points = np.zeros((num_points,3)) #points p(i,j): (x,y,z) coords. of surface node labeled i  (j=1,2,3) x = p(i,1) y = p(i,2) z = p(i,3)


    # local variable used in function
    x = np.zeros((num_points,6)) #x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    y = np.zeros((num_points,6)) #x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    z = np.zeros((num_points,6)) ##x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    xn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    yn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    zn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    eps=0.0000001
#----------------------------------------
#  0'th level discretization (8 elements)
#  nodes are set manually
#----------------------------------------
    nelm = 8
    
#---
#  corner points .... upper half of xz plane
#---

    x[0,0]= 0.0
    y[0,0]= 0.0
    z[0,0]= 1.0

    x[0,1]= 1.0
    y[0,1]= 0.0
    z[0,1]= 0.0

    x[0,2]= 0.0
    y[0,2]= 1.0
    z[0,2]= 0.0
    #---
    x[4,0]= 1.0
    y[4,0]= 0.0
    z[4,0]= 0.0

    x[4,1]= 0.0
    y[4,1]= 0.0
    z[4,1]=-1.0

    x[4,2]= 0.0
    y[4,2]= 1.0
    z[4,2]= 0.0
    #---
    x[5,0]= 0.0
    y[5,0]= 0.0
    z[5,0]=-1.0 

    x[5,1]=-1.0
    y[5,1]= 0.0
    z[5,1]= 0.0

    x[5,2]= 0.0
    y[5,2]= 1.0
    z[5,2]= 0.0
    #---
    x[1,0]=-1.0
    y[1,0]= 0.0
    z[1,0]= 0.0

    x[1,1]= 0.0
    y[1,1]= 0.0
    z[1,1]= 1.0

    x[1,2]= 0.0
    y[1,2]= 1.0
    z[1,2]= 0.0

    #---
    #  corner points .... lower half xz plane
    #---

    x[3,0]= 0.0
    y[3,0]= 0.0
    z[3,0]= 1.0

    x[3,1]= 0.0
    y[3,1]=-1.0
    z[3,1]= 0.0

    x[3,2]= 1.0
    y[3,2]= 0.0
    z[3,2]= 0.0
    #---
    x[7,0]= 1.0
    y[7,0]= 0.0
    z[7,0]= 0.0

    x[7,1]= 0.0
    y[7,1]=-1.0
    z[7,1]= 0.0

    x[7,2]= 0.0
    y[7,2]= 0.0
    z[7,2]=-1.0
    #---
    x[6,0]= 0.0
    y[6,0]= 0.0
    z[6,0]=-1.0 

    x[6,1]= 0.0
    y[6,1]=-1.0
    z[6,1]= 0.0

    x[6,2]=-1.0
    y[6,2]= 0.0
    z[6,2]= 0.0

    #---
    x[2,0]=-1.0
    y[2,0]= 0.0
    z[2,0]= 0.0

    x[2,1]= 0.0
    y[2,1]=-1.0
    z[2,1]= 0.0

    x[2,2]= 0.0
    y[2,2]= 0.0
    z[2,2]= 1.0    
    
    
#---
# compute the mid-points of the sides
# numbered 4, 5, 6
#---

    for i in range(nelm):
        x[i,3]= 0.5*(x[i,0]+x[i,1])
        y[i,3]= 0.5*(y[i,0]+y[i,1])
        z[i,3]= 0.5*(z[i,0]+z[i,1])

        x[i,4]= 0.5*(x[i,1]+x[i,2])
        y[i,4]= 0.5*(y[i,1]+y[i,2])
        z[i,4]= 0.5*(z[i,1]+z[i,2])

        x[i,5]= 0.5*(x[i,2]+x[i,0])
        y[i,5]= 0.5*(y[i,2]+y[i,0])
        z[i,5]= 0.5*(z[i,2]+z[i,0])    
    
    
    
#----------------------------------------
# compute node coordinates on each element 
# for discretization levels
# 1 through ndiv
#----------------------------------------

    for  i in range(ndiv):

        num = 0

        for j in range(nelm):                        #  loop over old elements

#---
#  assign corner points to sub-elements
#---    
    
    
            xn[num,0]= x[j,0]                  #  first sub-element
            yn[num,0]= y[j,0]
            zn[num,0]= z[j,0]

            xn[num,1]= x[j,3]
            yn[num,1]= y[j,3] 
            zn[num,1]= z[j,3]

            xn[num,2]= x[j,5]
            yn[num,2]= y[j,5]
            zn[num,2]= z[j,5]

            xn[num,3]= 0.5*(xn[num,0]+xn[num,1])
            yn[num,3]= 0.5*(yn[num,0]+yn[num,1])
            zn[num,3]= 0.5*(zn[num,0]+zn[num,1])

            xn[num,4]= 0.5*(xn[num,1]+xn[num,2])
            yn[num,4]= 0.5*(yn[num,1]+yn[num,2])
            zn[num,4]= 0.5*(zn[num,1]+zn[num,2])

            xn[num,5]= 0.5*(xn[num,2]+xn[num,0])
            yn[num,5]= 0.5*(yn[num,2]+yn[num,0])
            zn[num,5]= 0.5*(zn[num,2]+zn[num,0])

            xn[num+1,0]= x[j,3]                #  second sub-element
            yn[num+1,0]= y[j,3]
            zn[num+1,0]= z[j,3]

            xn[num+1,1]= x[j,1]
            yn[num+1,1]= y[j,1]
            zn[num+1,1]= z[j,1]

            xn[num+1,2]= x[j,4]
            yn[num+1,2]= y[j,4]
            zn[num+1,2]= z[j,4]

            xn[num+1,3]= 0.5*(xn[num+1,0]+xn[num+1,1])
            yn[num+1,3]= 0.5*(yn[num+1,0]+yn[num+1,1])
            zn[num+1,3]= 0.5*(zn[num+1,0]+zn[num+1,1])

            xn[num+1,4]= 0.5*(xn[num+1,1]+xn[num+1,2])
            yn[num+1,4]= 0.5*(yn[num+1,1]+yn[num+1,2])
            zn[num+1,4]= 0.5*(zn[num+1,1]+zn[num+1,2])

            xn[num+1,5]= 0.5*(xn[num+1,2]+xn[num+1,0])
            yn[num+1,5]= 0.5*(yn[num+1,2]+yn[num+1,0])
            zn[num+1,5]= 0.5*(zn[num+1,2]+zn[num+1,0])

            xn[num+2,0]= x[j,5]                #  third sub-element
            yn[num+2,0]= y[j,5]
            zn[num+2,0]= z[j,5]

            xn[num+2,1]= x[j,4]
            yn[num+2,1]= y[j,4]
            zn[num+2,1]= z[j,4]

            xn[num+2,2]= x[j,2]
            yn[num+2,2]= y[j,2]
            zn[num+2,2]= z[j,2]

            xn[num+2,3]= 0.5*(xn[num+2,0]+xn[num+2,1])
            yn[num+2,3]= 0.5*(yn[num+2,0]+yn[num+2,1])
            zn[num+2,3]= 0.5*(zn[num+2,0]+zn[num+2,1])

            xn[num+2,4]= 0.5*(xn[num+2,1]+xn[num+2,2])
            yn[num+2,4]= 0.5*(yn[num+2,1]+yn[num+2,2])
            zn[num+2,4]= 0.5*(zn[num+2,1]+zn[num+2,2])

            xn[num+2,5]= 0.5*(xn[num+2,2]+xn[num+2,0])
            yn[num+2,5]= 0.5*(yn[num+2,2]+yn[num+2,0])
            zn[num+2,5]= 0.5*(zn[num+2,2]+zn[num+2,0])

            xn[num+3,0]= x[j,3]                #  fourth sub-element
            yn[num+3,0]= y[j,3]
            zn[num+3,0]= z[j,3]

            xn[num+3,1]= x[j,4]
            yn[num+3,1]= y[j,4]
            zn[num+3,1]= z[j,4]

            xn[num+3,2]= x[j,5]
            yn[num+3,2]= y[j,5]
            zn[num+3,2]= z[j,5]

            xn[num+3,3]= 0.5*(xn[num+3,0]+xn[num+3,1])    #mid points
            yn[num+3,3]= 0.5*(yn[num+3,0]+yn[num+3,1])
            zn[num+3,3]= 0.5*(zn[num+3,0]+zn[num+3,1])

            xn[num+3,4]= 0.5*(xn[num+3,1]+xn[num+3,2])
            yn[num+3,4]= 0.5*(yn[num+3,1]+yn[num+3,2])
            zn[num+3,4]= 0.5*(zn[num+3,1]+zn[num+3,2])

            xn[num+3,5]= 0.5*(xn[num+3,2]+xn[num+3,0])
            yn[num+3,5]= 0.5*(yn[num+3,2]+yn[num+3,0])
            zn[num+3,5]= 0.5*(zn[num+3,2]+zn[num+3,0])

            num = num+4                # four elements were generated    
    
        nelm=nelm*4
    
#---
# rename the new points
# and place them in the master list
#---
    
        for k in range(nelm):
            for l in range(6):
                x[k,l] = xn[k,l]
                y[k,l] = yn[k,l]
                z[k,l] = zn[k,l]

                xn[k,l] = 0.0   # zero just in case
                yn[k,l] = 0.0
                zn[k,l] = 0.0

    
    
#-----------------------------------
# Create a list of surface nodes by looping 
# over all elements
# and adding nodes not already found in the list.
#
# Fill the connectivity table n(i,j) 
# node numbers of element points 1-6
#-----------------------------------

#---
# first element is set mannualy 

    points[0,0] = x[0,0]
    points[0,1] = y[0,0]
    points[0,2] = z[0,0]

    points[1,0] = x[0,1]
    points[1,1] = y[0,1]
    points[1,2] = z[0,1]

    points[2,0] = x[0,2]
    points[2,1] = y[0,2]
    points[2,2] = z[0,2]

    points[3,0] = x[0,3]
    points[3,1] = y[0,3]
    points[3,2] = z[0,3]

    points[4,0] = x[0,4]
    points[4,1] = y[0,4]
    points[4,2] = z[0,4]

    points[5,0] = x[0,5]
    points[5,1] = y[0,5]
    points[5,2] = z[0,5]

    nn[0,0] = 0
    nn[0,1] = 1
    nn[0,2] = 2
    nn[0,3] = 3
    nn[0,4] = 4
    nn[0,5] = 5
    npts=6
    for i in np.arange(1,nelm): #loop over elements
        for j in range(6): # loop over element nodes
            Iflag=0
            for k in range(npts):
                if (np.abs(x[i,j]-points[k,0]) <= eps):
                    if (np.abs(y[i,j]-points[k,1]) <= eps):
                        if (np.abs(z[i,j]-points[k,2]) <= eps):
                            Iflag = 1      # the node has been previously recorded 
                            nn[i,j] = k     # the jth local node of element i is the kth global node 
            
            if (Iflag == 0 ):         # record the node
                npts = npts+1         
                points[npts - 1,0] = x[i,j]
                points[npts -1,1] = y[i,j]
                points[npts -1,2] = z[i,j]

                nn[i,j] = npts -1


# ----------------------------------
# Generate connectivity table ne(i,j)
# for elements touching node i
#----------------------------------       
    for i in range(npts):
        for j in range(7):
            ne[i,j] = 0
    for i in range(npts): #  loop over global nodes
        ne[i,1] = 0
        icount = 0
        for j in range(nelm):
            for k in range(6):
                if (np.abs(points[i,0]-x[j,k]) <= eps):
                    if (np.abs(points[i,1]-y[j,k]) <=eps):
                        if (np.abs(points[i,2]-z[j,k]) <= eps):

                            icount=icount+1
                            ne[i,0]=ne[i,0]+1
                            ne[i,icount]=j
#------------------------------------------
#  Create connectivity table nbe(i,j) for 
#  neighboring elements j of element i
#
#  Testing is done with respect to the mid-points
#
# (for boundary elements with only 2 neighbors,
#  the array entry will be zero)
#------------------------------------------
    for i in range(nelm):
        for j in range(3):

            nbe[i,j] = 0

            
    for i in range (nelm) :
        jcount=0
        for j in np.arange(3,6): #  loop over mid-points
            for k in range(nelm): #  test element  
                 if (k != i):
                    for l in np.arange(3,6):
                        if (abs(x[i,j]-x[k,l]) <= eps):
                            if (abs(y[i,j]-y[k,l]) <= eps):
                                if (abs(z[i,j]-z[k,l]) <= eps):

                                    nbe[i,jcount]=k
            if (nbe[i,jcount] != 0):
                jcount = jcount + 1 
#---
# project points p(i,j) onto the unit sphere
#---
    for i in range(npts):
        r = np.sqrt(points[i,0]**2+points[i,1]**2+points[i,2]**2)

        points[i,0] = points[i,0]/r
        points[i,1] = points[i,1]/r
        points[i,2] = points[i,2]/r
    return npts, nelm, points , ne, nn, nbe 



 
def trgl6_icos(ndiv):
#----------------------------------------
#  0'th level discretization (8 elements)
#  nodes are set manually
#----------------------------------------
    nn= np.zeros((num_elements,6), dtype=int) #related to elements: n(i,j) .... node number of point j on element i, where j=1,...,6
    nbe = np.zeros((num_elements,3), dtype=int) #related to elements nbe(i,j): label of element sharing side j of element i where j = 1, 2, 3
    ne = np.zeros((num_points,7), dtype=int) # ne(i,j) ... ne(i,1) is the number of elements touching node i. ne(i,2:ne(i,1)) 
    # are the corresponding element
    points = np.zeros((num_points,3)) #points p(i,j): (x,y,z) coords. of surface node labeled i  (j=1,2,3) x = p(i,1) y = p(i,2) z = p(i,3)


    # local variable used in function
    x = np.zeros((num_points,6)) #x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    y = np.zeros((num_points,6)) #x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    z = np.zeros((num_points,6)) ##x(i,j), y(i,j), z(i,j) .... Cartesian coords of point j on element i
    xn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    yn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    zn = np.zeros((num_points,6)) #used for assign corner points to sub-elements
    VX = np.zeros(12)
    VY = np.zeros(12)
    VZ = np.zeros(12)
    eps=0.00000001
    #----------------------------------------
    # Begin with the zeroth-level
    # discretization (20 elements)
    #
    # Nodes are set manually on the unit sphere
    #----------------------------------------

    #---
    # the icosahedron has 12 vertices
    #---
    
    ru = 0.25*np.sqrt(10.0+2.0*np.sqrt(5.0))
    rm = 0.25*(1.0+np.sqrt(5.0))   # twice the golden ratio

    c0 = 0.0
    #     c1 = 0.9512
    c1 = ru
    #     c2 = 0.8507 D0
    c2 = 2.0*ru/np.sqrt(5.0)
    #     c3 = 0.8090 D0
    c3 = rm
    #     c4 = 0.4253 D0
    c4 = ru/np.sqrt(5.0)
    #     c5 = 0.2629
    c5 = np.sqrt( ru**2-c3**2-c4**2)
    c6 = 0.5
    #     c7 = 0.6882 D0
    c7 = np.sqrt( ru**2-c4**2-c6**2)

    #     write (6,*) c5,c7

    VX[0] =  c0
    VY[0] =  c0
    VZ[0] =  c1

    VX[1] =  c0
    VY[1] =  c2
    VZ[1] =  c4

    VX[2] =  c3
    VY[2] =  c5
    VZ[2] =  c4

    VX[3] =  c6
    VY[3] = -c7
    VZ[3] =  c4

    VX[4] = -c6
    VY[4] = -c7
    VZ[4] =  c4

    VX[5] = -c3
    VY[5] =  c5
    VZ[5] =  c4

    VX[6] = -c6
    VY[6] =  c7
    VZ[6] = -c4

    VX[7] =  c6
    VY[7] =  c7
    VZ[7] = -c4

    VX[8] =  c3
    VY[8] = -c5
    VZ[8] = -c4

    VX[9] =  c0
    VY[9] = -c2
    VZ[9] = -c4

    VX[10] = -c3
    VY[10] = -c5
    VZ[10] = -c4

    VX[11] =  c0
    VY[11] =  c0
    VZ[11] = -c1

      

#------------------------
# define the corner nodes
#------------------------
       
    x[0,0] = VX[0]   # first element
    y[0,0] = VY[0]
    z[0,0] = VZ[0]
    x[0,1] = VX[2]
    y[0,1] = VY[2]
    z[0,1] = VZ[2]
    x[0,2] = VX[1]
    y[0,2] = VY[1]
    z[0,2] = VZ[1]
    #---
    x[1,0] = VX[0]
    y[1,0] = VY[0]
    z[1,0] = VZ[0]
    x[1,1] = VX[3]
    y[1,1] = VY[3]
    z[1,1] = VZ[3]
    x[1,2] = VX[2]
    y[1,2] = VY[2]
    z[1,2] = VZ[2]
    #---
    x[2,0] = VX[0]
    y[2,0] = VY[0]
    z[2,0] = VZ[0]
    x[2,1] = VX[4]
    y[2,1] = VY[4]
    z[2,1] = VZ[4]
    x[2,2] = VX[3]
    y[2,2] = VY[3]
    z[2,2] = VZ[3]
    #---
    x[3,0] = VX[0]
    y[3,0] = VY[0]
    z[3,0] = VZ[0]
    x[3,1] = VX[5]
    y[3,1] = VY[5]
    z[3,1] = VZ[5]
    x[3,2] = VX[4]
    y[3,2] = VY[4]
    z[3,2] = VZ[4]
    #---
    x[4,0] = VX[0]
    y[4,0] = VY[0]
    z[4,0] = VZ[0]
    x[4,1] = VX[1]
    y[4,1] = VY[1]
    z[4,1] = VZ[1]
    x[4,2] = VX[5]
    y[4,2] = VY[5]
    z[4,2] = VZ[5]
    #---
    x[5,0] = VX[1]
    y[5,0] = VY[1]
    z[5,0] = VZ[1]
    x[5,1] = VX[2]
    y[5,1] = VY[2]
    z[5,1] = VZ[2]
    x[5,2] = VX[7]
    y[5,2] = VY[7]
    z[5,2] = VZ[7]
    #---
    x[6,0] = VX[2]
    y[6,0] = VY[2]
    z[6,0] = VZ[2]
    x[6,1] = VX[3]
    y[6,1] = VY[3]
    z[6,1] = VZ[3]
    x[6,2] = VX[8]
    y[6,2] = VY[8]
    z[6,2] = VZ[8]
    #---
    x[7,0] = VX[3]
    y[7,0] = VY[3]
    z[7,0] = VZ[3]
    x[7,1] = VX[4]
    y[7,1] = VY[4]
    z[7,1] = VZ[4]
    x[7,2] = VX[9]
    y[7,2] = VY[9]
    z[7,2] = VZ[9]
    #---
    x[8,0] = VX[4]
    y[8,0] = VY[4]
    z[8,0] = VZ[4]
    x[8,1] = VX[5]
    y[8,1] = VY[5]
    z[8,1] = VZ[5]
    x[8,2] = VX[10]
    y[8,2] = VY[10]
    z[8,2] = VZ[10]
    #---
    x[9,0] = VX[5]
    y[9,0] = VY[5]
    z[9,0] = VZ[5]
    x[9,1] = VX[1]
    y[9,1] = VY[1]
    z[9,1] = VZ[1]
    x[9,2] = VX[6]
    y[9,2] = VY[6]
    z[9,2] = VZ[6]
    #---
    x[10,0] = VX[1]
    y[10,0] = VY[1]
    z[10,0] = VZ[1]
    x[10,1] = VX[7]
    y[10,1] = VY[7]
    z[10,1] = VZ[7]
    x[10,2] = VX[6]
    y[10,2] = VY[6]
    z[10,2] = VZ[6]
    #---
    x[11,0] = VX[2]
    y[11,0] = VY[2]
    z[11,0] = VZ[2]
    x[11,1] = VX[8]
    y[11,1] = VY[8]
    z[11,1] = VZ[8]
    x[11,2] = VX[7]
    y[11,2] = VY[7]
    z[11,2] = VZ[7]
    #---
    x[12,0] = VX[3]
    y[12,0] = VY[3]
    z[12,0] = VZ[3]
    x[12,1] = VX[9]
    y[12,1] = VY[9]
    z[12,1] = VZ[9]
    x[12,2] = VX[8]
    y[12,2] = VY[8]
    z[12,2] = VZ[8]
    #---
    x[13,0] = VX[4]
    y[13,0] = VY[4]
    z[13,0] = VZ[4]
    x[13,1] = VX[10]
    y[13,1] = VY[10]
    z[13,1] = VZ[10]
    x[13,2] = VX[9]
    y[13,2] = VY[9]
    z[13,2] = VZ[9]
    #---
    x[14,0] = VX[5]
    y[14,0] = VY[5]
    z[14,0] = VZ[5]
    x[14,1] = VX[6]
    y[14,1] = VY[6]
    z[14,1] = VZ[6]
    x[14,2] = VX[10]
    y[14,2] = VY[10]
    z[14,2] = VZ[10]
    #---
    x[15,0] = VX[6]
    y[15,0] = VY[6]
    z[15,0] = VZ[6]
    x[15,1] = VX[7]
    y[15,1] = VY[7]
    z[15,1] = VZ[7]
    x[15,2] = VX[11]
    y[15,2] = VY[11]
    z[15,2] = VZ[11]
    #---
    x[16,0] = VX[7]
    y[16,0] = VY[7]
    z[16,0] = VZ[7]
    x[16,1] = VX[8]
    y[16,1] = VY[8]
    z[16,1] = VZ[8]
    x[16,2] = VX[11]
    y[16,2] = VY[11]
    z[16,2] = VZ[11]
    #--- 
    x[17,0] = VX[8]
    y[17,0] = VY[8]
    z[17,0] = VZ[8]
    x[17,1] = VX[9]
    y[17,1] = VY[9]
    z[17,1] = VZ[9]
    x[17,2] = VX[11]
    y[17,2] = VY[11]
    z[17,2] = VZ[11]
    #---
    x[18,0] = VX[9]
    y[18,0] = VY[9]
    z[18,0] = VZ[9]
    x[18,1] = VX[10]
    y[18,1] = VY[10]
    z[18,1] = VZ[10]
    x[18,2] = VX[11]
    y[18,2] = VY[11]
    z[18,2] = VZ[11]
    #---
    x[19,0] = VX[10]
    y[19,0] = VY[10]
    z[19,0] = VZ[10]
    x[19,1] = VX[6]
    y[19,1] = VY[6]
    z[19,1] = VZ[6]
    x[19,2] = VX[11]
    y[19,2] = VY[11]
    z[19,2] = VZ[11]
       
#------------------------------------------
# compute the mid-points of the three sides
# of the 20 first-generation elements
#
# midpoints are numbered 4, 5, 6
#------------------------------------------
    nelm = 20
    for i in range(nelm):
        x[i,3] = 0.5*(x[i,0]+x[i,1])
        y[i,3] = 0.5*(y[i,0]+y[i,1])
        z[i,3] = 0.5*(z[i,0]+z[i,1])

        x[i,4] = 0.5*(x[i,1]+x[i,2])
        y[i,4] = 0.5*(y[i,1]+y[i,2])
        z[i,4] = 0.5*(z[i,1]+z[i,2])

        x[i,5] = 0.5*(x[i,2]+x[i,0])
        y[i,5] = 0.5*(y[i,2]+y[i,0])
        z[i,5] = 0.5*(z[i,2]+z[i,0])
#---
# project the nodes onto the unit sphere
#---
    for k in range(nelm):
        for l in range (6):
            rad = np.sqrt(x[k,l]**2+y[k,l]**2+z[k,l]**2)
            x[k,l] = x[k,l]/rad
            y[k,l] = y[k,l]/rad
            z[k,l] = z[k,l]/rad
    if (ndiv !=0):
#-------------------------------------------
# compute the local element node coordinates
# for discretization levels 1 through ndiv
#-------------------------------------------
        for  i in range(ndiv):
            nm = 0      # count the new elements arising by sub-division four element will be generated during each pass # over old elements
            for j in range(nelm):
        #---
        # assign corner points to sub-elements
        # these will become the "new" elements
        #--- 
                if (nm != 0):
                    nm = nm + 1
                xn[nm,0] = x[j,0]                  #  first sub-element
                yn[nm,0] = y[j,0]
                zn[nm,0] = z[j,0]

                xn[nm,1] = x[j,3]
                yn[nm,1] = y[j,3] 
                zn[nm,1] = z[j,3]

                xn[nm,2] = x[j,5]
                yn[nm,2] = y[j,5]
                zn[nm,2] = z[j,5]

                xn[nm,3] = 0.50*(xn[nm,0]+xn[nm,1])
                yn[nm,3] = 0.50*(yn[nm,0]+yn[nm,1])
                zn[nm,3] = 0.50*(zn[nm,0]+zn[nm,1])

                xn[nm,4] = 0.50*(xn[nm,1]+xn[nm,2])
                yn[nm,4] = 0.50*(yn[nm,1]+yn[nm,2])
                zn[nm,4] = 0.50*(zn[nm,1]+zn[nm,2])

                xn[nm,5] = 0.50*(xn[nm,2]+xn[nm,0])
                yn[nm,5] = 0.50*(yn[nm,2]+yn[nm,0])
                zn[nm,5] = 0.50*(zn[nm,2]+zn[nm,0])

                nm = nm+1
                
                xn[nm,0] = x[j,3]                #  second sub-element
                yn[nm,0] = y[j,3]
                zn[nm,0] = z[j,3]

                xn[nm,1] = x[j,1]
                yn[nm,1] = y[j,1]
                zn[nm,1] = z[j,1]

                xn[nm,2] = x[j,4]
                yn[nm,2] = y[j,4]
                zn[nm,2] = z[j,4]

                xn[nm,3] = 0.5*(xn[nm,0]+xn[nm,1])
                yn[nm,3] = 0.5*(yn[nm,0]+yn[nm,1])
                zn[nm,3] = 0.5*(zn[nm,0]+zn[nm,1])

                xn[nm,4] = 0.5*(xn[nm,1]+xn[nm,2])
                yn[nm,4] = 0.5*(yn[nm,1]+yn[nm,2])
                zn[nm,4] = 0.5*(zn[nm,1]+zn[nm,2])

                xn[nm,5] = 0.5*(xn[nm,2]+xn[nm,0])
                yn[nm,5] = 0.5*(yn[nm,2]+yn[nm,0])
                zn[nm,5] = 0.5*(zn[nm,2]+zn[nm,0])

                nm = nm+1
                
                xn[nm,0] = x[j,5]                #  third sub-element
                yn[nm,0] = y[j,5]
                zn[nm,0] = z[j,5]

                xn[nm,1] = x[j,4]
                yn[nm,1] = y[j,4]
                zn[nm,1] = z[j,4]

                xn[nm,2] = x[j,2]
                yn[nm,2] = y[j,2]
                zn[nm,2] = z[j,2]

                xn[nm,3] = 0.5*(xn[nm,0]+xn[nm,1])
                yn[nm,3] = 0.5*(yn[nm,0]+yn[nm,1])
                zn[nm,3] = 0.5*(zn[nm,0]+zn[nm,1])

                xn[nm,4] = 0.5*(xn[nm,1]+xn[nm,2])
                yn[nm,4] = 0.5*(yn[nm,1]+yn[nm,2])
                zn[nm,4] = 0.5*(zn[nm,1]+zn[nm,2])

                xn[nm,5] = 0.5*(xn[nm,2]+xn[nm,0])
                yn[nm,5] = 0.5*(yn[nm,2]+yn[nm,0])
                zn[nm,5] = 0.5*(zn[nm,2]+zn[nm,0])

                nm = nm+1

                xn[nm,0] = x[j,3]                #  fourth sub-element
                yn[nm,0] = y[j,3]
                zn[nm,0] = z[j,3]

                xn[nm,1] = x[j,4]
                yn[nm,1] = y[j,4]
                zn[nm,1] = z[j,4]

                xn[nm,2] = x[j,5]
                yn[nm,2] = y[j,5]
                zn[nm,2] = z[j,5]

                xn[nm,3] = 0.5*(xn[nm,0]+xn[nm,1])    # mid points
                yn[nm,3] = 0.5*(yn[nm,0]+yn[nm,1])
                zn[nm,3] = 0.5*(zn[nm,0]+zn[nm,1])

                xn[nm,4] = 0.5*(xn[nm,1]+xn[nm,2])
                yn[nm,4] = 0.5*(yn[nm,1]+yn[nm,2])
                zn[nm,4] = 0.5*(zn[nm,1]+zn[nm,2])

                xn[nm,5] = 0.5*(xn[nm,2]+xn[nm,0])
                yn[nm,5] = 0.5*(yn[nm,2]+yn[nm,0])
                zn[nm,5] = 0.5*(zn[nm,2]+zn[nm,0])
#--------------------------------------
# number of elements has been increased
# by a factor of four
#--------------------------------------

            nelm = 4*nelm
#---
# relabel the new points
# and place them in the master list
#---
            for k in range(nelm):
                for l in range(6):
                    x[k,l] = xn[k,l]
                    y[k,l] = yn[k,l]
                    z[k,l] = zn[k,l]

                    #--- project onto the unit sphere

                    rad = np.sqrt(x[k,l]**2+y[k,l]**2+z[k,l]**2)
                    x[k,l] = x[k,l]/rad
                    y[k,l] = y[k,l]/rad
                    z[k,l] = z[k,l]/rad

                    xn[k,l] = 0.0   # zero just in case
                    yn[k,l] = 0.0
                    zn[k,l] = 0.0 
    



#-----------------------------------
# Create a list of surface nodes by looping 
# over all elements
# and adding nodes not already found in the list.
#
# Fill the connectivity table n(i,j) 
# node numbers of element points 1-6
#-----------------------------------

#---
# first element is set mannualy 
    points[0,0] = x[0,0]
    points[0,1] = y[0,0]
    points[0,2] = z[0,0]

    points[1,0] = x[0,1]
    points[1,1] = y[0,1]
    points[1,2] = z[0,1]

    points[2,0] = x[0,2]
    points[2,1] = y[0,2]
    points[2,2] = z[0,2]

    points[3,0] = x[0,3]
    points[3,1] = y[0,3]
    points[3,2] = z[0,3]

    points[4,0] = x[0,4]
    points[4,1] = y[0,4]
    points[4,2] = z[0,4]

    points[5,0] = x[0,5]
    points[5,1] = y[0,5]
    points[5,2] = z[0,5]

    nn[0,0] = 0
    nn[0,1] = 1
    nn[0,2] = 2
    nn[0,3] = 3
    nn[0,4] = 4
    nn[0,5] = 5

    npts=6
    for i in np.arange(1,nelm): #loop over elements
        for j in range(6): # loop over element nodes
            Iflag=0
            for k in range(npts):
                if (np.abs(x[i,j]-points[k,0]) <= eps):
                    if (np.abs(y[i,j]-points[k,1]) <= eps):
                        if (np.abs(z[i,j]-points[k,2]) <= eps):
                            Iflag = 1      # the node has been previously recorded 
                            nn[i,j] = k     # the jth local node of element i is the kth global node 
            
            if (Iflag == 0 ):         # record the node
                npts = npts+1         
                points[npts - 1,0] = x[i,j]
                points[npts -1,1] = y[i,j]
                points[npts -1,2] = z[i,j]

                nn[i,j] = npts -1

# ----------------------------------
# Generate connectivity table ne(i,j)
# for elements touching node i
#----------------------------------       
    for i in range(npts):
        for j in range(7):
            ne[i,j] = 0
    for i in range(npts): #  loop over global nodes
        ne[i,1] = 0
        icount = 0
        for j in range(nelm):
            for k in range(6):
                if (np.abs(points[i,0]-x[j,k]) <= eps):
                    if (np.abs(points[i,1]-y[j,k]) <=eps):
                        if (np.abs(points[i,2]-z[j,k]) <= eps):

                            icount=icount+1
                            ne[i,0]=ne[i,0]+1
                            ne[i,icount]=j
                            
#------------------------------------------
#  Create connectivity table nbe(i,j) for 
#  neighboring elements j of element i
#
#  Testing is done with respect to the mid-points
#
# (for boundary elements with only 2 neighbors,
#  the array entry will be zero)
#------------------------------------------
    for i in range(nelm):
        for j in range(3):

            nbe[i,j] = 0

            
    for i in range (nelm) :
        jcount=0
        for j in np.arange(3,6): #  loop over mid-points
            for k in range(nelm): #  test element  
                 if (k != i):
                    for l in np.arange(3,6):
                        if (abs(x[i,j]-x[k,l]) <= eps):
                            if (abs(y[i,j]-y[k,l]) <= eps):
                                if (abs(z[i,j]-z[k,l]) <= eps):

                                    nbe[i,jcount]=k
            if (nbe[i,jcount] != 0):
                jcount = jcount + 1 
#---
# project points p(i,j) onto the unit sphere
#---
    for i in range(npts):
        r = np.sqrt(points[i,0]**2+points[i,1]**2+points[i,2]**2)

        points[i,0] = points[i,0]/r
        points[i,1] = points[i,1]/r
        points[i,2] = points[i,2]/r
    return npts, nelm, points , ne, nn, nbe    



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





if(Ioctaicos == 1):
    npts, nelm, points , ne, nn, nbe = trgl6_octa(ndiv)
else:
    npts, nelm, points , ne, nn, nbe = trgl6_icos(ndiv)
print('lnm_3d: number of points:', npts)
print('lnm_3d: number of elements:', nelm)





#---
# expand to specified radius;
# deform to an ellipsoid;
# translate center
# to specified position;
#---
# scale = req/(boa*coa)**oot


# for i in range(npts):
    # points[i,0] = scale * points[i,0]
    # points[i,1] = scale * boa * points[i,1]
    # points[i,2] = scale * coa * points[i,2]

#-------------------------
#rotate by phi1,phi2,phi3
#around the x,y,z axes
#-------------------------

# phi1 = phi1* np.pi   # scale
# phi2 = phi2 * np.pi   # scale
# phi3 = phi3 * np.pi   # scale  

# cs = np.cos(phi1)
# sn = np.sin(phi1)

#rotate about the x axis
# for i in range(npts):                  # rotate about the x axis
    # tmpx = points[i,0]
    # tmpy = cs * points[i,1] + sn * points[i,2]
    # tmpz =-sn * points[i,1] + cs * points[i,2]
    # points[i,0] = tmpx
    # points[i,1] = tmpy
    # points[i,2] = tmpz

# cs = np.cos(phi2)
# sn = np.sin(phi2)

# for i in range(npts):                  # rotate about the y axis
    # tmpx = cs * points[i,0] - sn * points[i,2]
    # tmpy = points[i,1]
    # tmpz = sn * points[i,0] + cs * points[i,2]
    # points[i,0] = tmpx
    # points[i,1] = tmpy
    # points[i,2] = tmpz


# cs = np.cos(phi3)
# sn = np.sin(phi3)

# for i in range(npts):               # rotate about the z axis
    # tmpx = cs*points[i,0]+sn*points[i,1]
    # tmpy =-sn*points[i,0]+cs*points[i,1]
    # tmpz = points[i,2]
    # points[i,0] = tmpx
    # points[i,1] = tmpy
    # points[i,2] = tmpz


# phi1 = phi1/np.pi   # unscale
# phi2 = phi2/np.pi
# phi3 = phi3/np.pi

#---------------------
#translate center to
#specified position
#---------------------
# for i in range(npts):
    # points[i,0] = points[i,0] + cxp
    # points[i,1] = points[i,1] + cyp
    # points[i,2] = points[i,2] + czp




# writing csv file
pd.DataFrame(points).to_csv("unit_sphere_points.csv", header=None, index = False)
for j in range(npts):
    tem = ne[j,0] + 1
    ne[j,1:tem] = ne[j,1:tem] + 1
pd.DataFrame(ne).to_csv("ne.csv", header=None, index = False)
nbe = nbe +1
pd.DataFrame(nbe).to_csv("nbe.csv", header=None, index = False)
nn = nn +1
pd.DataFrame(nn).to_csv("nn.csv", header=None, index = False)










