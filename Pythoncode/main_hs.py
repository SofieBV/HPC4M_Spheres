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
subbox1=[[-l/2,l/2],[l/2,l/2],[l/2,0],[-l/2,0],"bottom"]
subbox2=[[-l/2,0],[l/2,0],[l/2,-l/2],[-l/2,-l/2],"top"]

if rank==0:
    inputs_nr = 2
    inputs_pos =  [np.array([0.3,0.1]), np.array([0.5,0.1])]
    inputs_vel =  [np.array([-1,0]), np.array([0,0])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=subbox1
    special_wall = "bottom"
    IS = initial_state(2,inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)

elif rank==1:
    inputs_nr = 2
    inputs_pos =  [np.array([-1,-0.1]), np.array([1,-1])]
    inputs_vel =  [np.array([-1,0]), np.array([0,0])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=subbox2
    IS = initial_state(2,inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)
else:
    heap=[]

############### TRY TO RUN THE COLLISION ##########################################################################################
#### initialise state and run collisions up to time T ####
#### throws up error due to lack of wall ####

T = 1
L = -np.ones(len(IS)) # last collision time for each atom
simulation = [] # any collisions that happened. 
oldheap = [] # any collisions that have not happened, we need to save if we ever need them afterward

t = 0
while t < T:

    ##################################################################
    ##### If one of them hit a special wall: 
    ########## then tell the box that is being crossed (and add the right rank condition)
    ########## Send info from subbox1 to subbox 2  (from rank0 to rank1) : 
    ################## Subbox 2 needs to receive it and check the time. if proc2 time < proctime 1
    ################## They need to receive the info of the particule, recompute their heap for this particle and go back to time of proc 1 
    ################## We need to check for forward in time (might still send a particle to the other side and we need to know that) 
    ################## and backward in time 

    ########## subbox2 for subbox 1 from rank1 to rank0 : 
    ########## 

    ##################################################################

    print(t)
    # possible collision
    entry = heapq.heappop(heap)

    ## Update the heap structure with the function update heap    
    (L,t,simulation,entry,heap) = update_heap(L,t,simulation,entry,IS,heap,subbox,comm,rank)


############### TRY TO RUN THE COLLISION ##########################################################################################


print("rank and heap",rank,heap)
print("rank and simulation",rank,simulation)



## Our code 
### Denote end of mpi calls.
MPI.Finalize()
###  