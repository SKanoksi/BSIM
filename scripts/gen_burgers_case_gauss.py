#
#  2D Burgers' simulation
#
#  Version 0.1.0
#  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
#  All rights reserved under BSD 3-clause license.
#

import numpy as np
import adios2
import matplotlib.pyplot as plt

filename = './BSIM_input/BSIM_checkpoint_2023-01-01_00-00-00'
varlist = ["u_velocity","v_velocity","T_temperature"]

nx = 201
ny = 201
lx = 8
ly = 8
Nx = nx + 2*lx
Ny = ny + 2*ly
data_u = np.zeros(Nx*Ny, dtype=np.double)
data_v = np.zeros(Nx*Ny, dtype=np.double)
data_T = np.zeros(Nx*Ny, dtype=np.double)

dx = 2.5
dy = 2.5
pos0_x = - dx*Nx*2/5
pos0_y = - dy*Ny*2/5

func = lambda x, y : 2.0*np.exp( - (x**2+y**2)/1000. , dtype=np.double)

pos_x = np.linspace(pos0_x, pos0_x+ Nx*dx, Nx)
pos_y = np.linspace(pos0_y, pos0_y+ Ny*dy, Ny)
for j in range(0, Ny):
  for i in range(0, Nx):
    data_u[j*Nx + i] = +0.2
    data_v[j*Nx + i] = +0.1
    data_T[j*Nx + i] = func(pos_x[i] , pos_y[j])


adios = adios2.ADIOS()
ioWrite = adios.DeclareIO("ioWriter")
ioWrite.SetEngine('bp4')

engineWrite = ioWrite.Open(filename, adios2.Mode.Write)

var_write = ioWrite.DefineVariable(varlist[0], data_u, [Ny, Nx], [0,0], [Ny,Nx], True)
engineWrite.Put(var_write, data_u[:])
var_write = ioWrite.DefineVariable(varlist[1], data_v, [Ny, Nx], [0,0], [Ny,Nx], True)
engineWrite.Put(var_write, data_v[:])
var_write = ioWrite.DefineVariable(varlist[2], data_T, [Ny, Nx], [0,0], [Ny,Nx], True)
engineWrite.Put(var_write, data_T[:])

engineWrite.PerformPuts()
#engineWrite.PerformDataWrite() # To disk (Only supported in BP5)
engineWrite.Close()

# -------------------------

plt.figure(0)
plt.clf()

xx, yy = np.meshgrid(pos_x, pos_y)
plt.contourf(xx, yy, np.reshape(data_T, [Ny,Nx]))
plt.colorbar()

plt.savefig('Figure_gen_case.png', dpi=300)

