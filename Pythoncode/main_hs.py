import numpy as np
from functions_hardsphere import *



# ### INITIALIZE MPI ######################################################################################
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nr_proc = comm.Get_size()
#########################################################################################################
# N=10 #number of particles in each processor
# 
l=10 #size of the box
# Each processor handle a subbox of coordinates:
# [top left, top right, bottom right, bottom left]
subbox1=[[-l/2,l/2],[l/2,l/2],[l/2,0],[-l/2,0]]
subbox2=[[-l/2,0],[l/2,0],[l/2,-l/2],[-l/2,-l/2]]

if rank==0:
    IS = initial_state(2, [np.array([0,0]), np.array([0,1])], [np.array([-1,0]), np.array([0,0])], 0.1*np.ones(2),0.1*np.ones(2))
    heap = create_heap(IS)

elif rank==1:
    IS = initial_state(2, [np.array([1,0]), np.array([1,1])], [ np.array([0,-1]), np.array([0,0])], 0.1*np.ones(2),0.1*np.ones(2))
    heap = create_heap(IS)
else:
    heap=[]

print(rank,heap)




## Our code 
### Denote end of mpi calls.
MPI.Finalize()
###  