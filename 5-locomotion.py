#
# This code is for self-phoresis
# of active particles, and developed
# by Amir Nourhani and Seyed Amin Nabavizadeh
# for Nourhani's research lab
# 
# The interpolation function is from 
# C. Pozrikidis's library
#

import pandas as pd
import numpy as np
import os
import sys

b_source = float(sys.argv[1])
b_sink = float(sys.argv[2])

# reading csv file
unit_p = pd.read_csv('unit_sphere_points.csv', header=None, index_col=None).values
df = pd.read_csv('concentration.csv', index_col=None).values
p = np.copy(df[:,:3])
conc = np.copy(df[:,3])


nn = pd.read_csv('nn.csv', header=None, index_col=None).values
# python convention related to first element of an array started from 0
nn = nn - 1

df = pd.read_csv('xyz_mom_area1.csv', header=None, index_col=None).values
xmom = df[:,0]
ymom = df[:,1]
zmom = df[:,2]
arel = df[:,3]


Hr = pd.read_csv('tractions_for_rotational_modes.csv', index_col=None).values
Ht = pd.read_csv('tractions_for_translation_modes.csv', index_col=None).values
Resist = pd.read_csv('resistance.csv', header=None, index_col=None).values
ResistInv = np.linalg.inv(Resist)

num_points = len(unit_p)
num_elements = len(nn)
b_ph  = np.zeros(num_points)
b_ph[unit_p[:,2]>0.0] = b_source
b_ph[unit_p[:,2]<0.0] = b_sink
b_ph[unit_p[:,2]==0.0] = (b_source + b_sink)/2.

ForceTorque = np.zeros(6)
phor_vel = np.zeros(6)
slip_vel = np.zeros((num_elements,3))


def phiFunc (al, be, ga, xi, eta):

   phi2 = xi *(xi - al+eta*(al-ga)/(1.-ga))/(1.-al)
   phi3 = eta*(eta-be+xi *(be+ga-1.)/ga)/(1.-be)
   phi4 = xi *(1.-xi-eta)/(al*(1.-al))
   phi5 = xi*eta/(ga*(1.-ga))
   phi6 = eta*(1.-xi-eta)/(be*(1.-be))
   phi1 = 1.-phi2-phi3-phi4-phi5-phi6

   return np.array([phi1, phi2, phi3, phi4, phi5, phi6])


def DphiDxiFunc (al, be, ga, xi, eta):

   Dphi2_Dxi =  (2.0*xi-al +eta*(al-ga)/(1.-ga))/(1.0-al)
   Dphi3_Dxi =  eta*(be+ga-1.0)/(ga*(1.0-be))
   Dphi4_Dxi =  (1.0-2.0*xi-eta)/(al*(1.0-al))
   Dphi5_Dxi =  eta/(ga*(1.0-ga))
   Dphi6_Dxi = -eta/(be*(1.0-be))
   Dphi1_Dxi = -Dphi2_Dxi-Dphi3_Dxi-Dphi4_Dxi-Dphi5_Dxi-Dphi6_Dxi

   return np.array([Dphi1_Dxi, Dphi2_Dxi, Dphi3_Dxi, Dphi4_Dxi, Dphi5_Dxi, Dphi6_Dxi])


def DphiDetaFunc (al, be, ga, xi, eta):

    Dphi2_Deta =  xi*(al-ga)/((1.-al)*(1.-ga))
    Dphi3_Deta =  (2.0*eta-be+xi*(be+ga-1.0)/ga)/(1.-be)
    Dphi4_Deta =  -xi/(al*(1.-al))
    Dphi5_Deta =   xi/(ga*(1.-ga))
    Dphi6_Deta =  (1.0-xi-2.0*eta)/(be*(1.-be))
    Dphi1_Deta = -Dphi2_Deta-Dphi3_Deta-Dphi4_Deta-Dphi5_Deta-Dphi6_Deta

    return np.array([Dphi1_Deta, Dphi2_Deta, Dphi3_Deta, Dphi4_Deta, Dphi5_Deta, Dphi6_Deta])


def interpolate (r1, r2, r3, r4, r5, r6,
                 c1, c2, c3, c4, c5, c6,
                 b1, b2, b3, b4, b5, b6,
                 Ht, Hr, elm_area,
                 al, be, ga, xi, eta):

    # interpolation coefficients
    phi = phiFunc (al, be, ga, xi, eta)
    Dphi_Dxi = DphiDxiFunc(al, be, ga, xi, eta)
    Dphi_Deta = DphiDetaFunc (al, be, ga, xi, eta)

    # inverse of the metric tensor: g_inv
    E_xi = np.dot(Dphi_Dxi, np.array([r1,r2,r3,r4,r5,r6]))
    E_eta = np.dot(Dphi_Deta, np.array([r1,r2,r3,r4,r5,r6]))
    ExiEeta = np.array([E_xi, E_eta])
    g_inv = np.linalg.inv(np.dot(ExiEeta, ExiEeta.T))

    # surface gradient of concentration
    Dc_Dxi = np.dot(Dphi_Dxi, np.array([c1,c2,c3,c4,c5,c6]))
    Dc_Deta = np.dot(Dphi_Deta, np.array([c1,c2,c3,c4,c5,c6]))
    coeff = np.dot(g_inv, np.array([Dc_Dxi, Dc_Deta]))
    grad_c = coeff[0]*E_xi + coeff[1]*E_eta

    # slip velocity
    mu = np.dot(phi, np.array([b1,b2,b3,b4,b5,b6]))
    v_slip = - mu * grad_c

    # Force and torque from the element
    elm_F = np.dot(Ht, v_slip) * elm_area
    elm_L = np.dot(Hr, v_slip) * elm_area

    return v_slip, np.append(elm_F, elm_L)


def abc(r1, r2, r3, r4, r5, r6):

    al = 1.0/(1.0 + np.linalg.norm(r4-r2)/np.linalg.norm(r4-r1))
    be = 1.0/(1.0 + np.linalg.norm(r6-r3)/np.linalg.norm(r6-r1))
    ga = 1.0/(1.0 + np.linalg.norm(r5-r2)/np.linalg.norm(r5-r3))

    return al,be,ga


# Main Program


#---------------------------------------------------
# Calculating the gradient of the concentration at
# the point reprenenting the center of the isoparametric
# triangle xi_c = eta_c = 1./3.


xi_c = 1.0/3.0
eta_c = 1.0/3.0
for elm_index in range(num_elements):

    i1, i2, i3, i4, i5, i6 = nn[elm_index]

    alpha, betta, gamma = abc(p[i1], p[i2], p[i3], p[i4], p[i5], p[i6])
    v_slip, elm_FL = interpolate(p[i1], p[i2], p[i3], p[i4], p[i5], p[i6],
                           conc[i1], conc[i2], conc[i3], conc[i4], conc[i5], conc[i6],
                           b_ph[i1], b_ph[i2], b_ph[i3], b_ph[i4], b_ph[i5], b_ph[i6],
                           Ht[elm_index].reshape((3,3)),
                           Hr[elm_index].reshape((3,3)),
                           arel[elm_index],
                           alpha, betta, gamma, xi_c, eta_c)

    slip_vel[elm_index,:] = v_slip

    ForceTorque = ForceTorque + elm_FL

phor_vel = np.dot(ResistInv, ForceTorque)



dict_temp = {'element_No': np.arange(num_elements) + 1, \
        'Ux': slip_vel[:,0], 'Uy':slip_vel[:,1], \
        'Uz':slip_vel[:,2]}

pd.DataFrame(dict_temp).to_csv("slip_velocity_element.csv", index=False)

#with open('phor_vel_omega.csv', 'w') as abc:
#    np.savetxt(abc, phor_vel, delimiter=",")

dict_temp = {'0': [phor_vel[0]], \
        '1': [phor_vel[1]], \
        '2': [phor_vel[2]], \
        '3': [phor_vel[3]], \
        '4': [phor_vel[4]], \
        '5': [phor_vel[5]]}
pd.DataFrame(dict_temp).to_csv("phor_vel_omega.csv",header=False, index=False)


