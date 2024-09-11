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
#from matplotlib import ticker

from sys import argv

var_name = argv[1]
filename = argv[2]

adios = adios2.ADIOS()
ioRead = adios.DeclareIO("ioReader")
engineRead = ioRead.Open(filename, adios2.Mode.Read)

var = ioRead.InquireVariable(var_name)

shape = var.Shape() # 0=y, 1=x
data = np.zeros(shape, dtype=np.double)
engineRead.Get(var, data)

engineRead.PerformGets()
engineRead.Close()

# -------------------------
data = np.abs(data)

'''
size = 12
plt.rcParams.update({'font.size': size})

plt.figure(0, figsize=(9.6, 3.6))
plt.plot(np.arange(len(data[:,20])), data[:,20], 'k')
plt.ylim([-1.,1.])
plt.savefig(var_name+'.png', dpi=300)
'''


# -------------------------

#'''
plt.figure(0)
plt.clf()

xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
zz = np.reshape(data, shape)
plt.contourf(xx, yy, zz)
#zz = np.ma.masked_where(zz<=0, zz)
#plt.contourf(xx, yy, zz, locator=ticker.LogLocator(numticks=10))   # log color
plt.colorbar()
plt.clim(1e-3, 0.1)

plt.savefig(var_name+'.png', dpi=300)
#'''


