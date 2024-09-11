#
#  2D Burgers' simulation
#
#  Version 0.1.0
#  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
#  All rights reserved under BSD 3-clause license.
#

import numpy as np
import adios2
from sys import argv

filename1 = argv[1]
filename2 = argv[2]
varlist = ["u_velocity","v_velocity","T_temperature"]

nx = 500
ny = 500
lx = 5
ly = 5
Nx = nx + 2*lx
Ny = ny + 2*ly
data1 = np.zeros([Ny,Ny], dtype=np.double)
data2 = np.zeros([Ny,Ny], dtype=np.double)

# -------------------

adios = adios2.ADIOS()
ioRead1 = adios.DeclareIO("ioReader1")
ioRead2 = adios.DeclareIO("ioReader2")

engineRead1 = ioRead1.Open(filename1, adios2.Mode.Read)
engineRead2 = ioRead2.Open(filename2, adios2.Mode.Read)
for var_name in varlist :
  var1 = ioRead1.InquireVariable(var_name)
  engineRead1.Get(var1, data1)
  engineRead1.PerformGets()

  var2 = ioRead2.InquireVariable(var_name)
  engineRead2.Get(var2, data2)
  engineRead2.PerformGets()

  print("Mean absolute error of", var_name,"=",np.mean(np.abs(data1-data2)))
  print("Max  absolute error of", var_name,"=",np.max(np.abs(data1-data2)))

engineRead1.Close()
engineRead2.Close()

