#
# This code is developed by Seyed Amin Nabavizadeh
# for Nourhani's research lab
# 


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
import os 
import sys


Ioctaicos = int(float(sys.argv[1]))
ndiv = int(float(sys.argv[2]))
coa = float(sys.argv[3])
rot_y = float(sys.argv[4])

fileName_input =  'unit_sphere_points.csv'

temp_geometry = pd.read_csv(fileName_input, header = None, index_col = None).values

# scale the z-coordinate bu coa
temp_geometry[:,2] = coa * temp_geometry[:,2]
geometry = np.copy(temp_geometry)

# rotation about y axis by angle rot_y
cosine = np.cos(np.pi /180.0 * rot_y)
sinus = np.sin(np.pi /180.0 * rot_y)
geometry[:,0] = cosine * temp_geometry[:,0] + sinus * temp_geometry[:,2]
geometry[:,2] = - sinus * temp_geometry[:,0] + cosine * temp_geometry[:,2]



# save the geometry to geometry.csv
pd.DataFrame(geometry).to_csv("geometry"+ ".csv",header =False , index=False)    

# save the geometry to geometry_paraview.csv for paraview
dict1 = {'x':geometry[:,0], 'y': geometry[:,1], 'z':geometry[:,2]}
pd.DataFrame(dict1).to_csv('geometry_paraview'+'.csv', index=False)



