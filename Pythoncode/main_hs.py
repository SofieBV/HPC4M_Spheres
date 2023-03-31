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
    inputs_nr = 2
    inputs_pos =  [np.array([0.3,0.1]), np.array([0.5,0.1])]
    inputs_vel =  [np.array([-1,0]), np.array([0,0])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=subbox1
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
    
    if not isinstance(entry[4],str):
        # checking if collision is valid event
        if entry[1] < L[entry[3].n]:
            pass
        elif entry[1] < L[entry[4].n]:
            pass
        else: # collision valid
            
            # updating last collision times
            L[entry[3].n] = entry[0]
            L[entry[4].n] = entry[0]
            
            # new particle pos and vel
            posi, veli, posj, velj = collision(entry)
            
            #save previous pos and vel
            simulation.append([entry[0], entry[3].n, entry[4].n, posi, posj, veli, velj])

            # update pos and vel
            entry[3].update(posi, veli)
            entry[4].update(posj, velj) # this will change all heap pos and vels
            # may not matter as they're no longer be valid?
            
            # update heap
            for i in IS:
                # collisions with first sphere
                dt = check_collision(i, entry[3])
                if dt != None:
                    heapq.heappush(heap, (dt + entry[0],entry[0],i.n,i, entry[3]))
                # collisions with second
                dt = check_collision(i, entry[4])
                if dt != None:
                    heapq.heappush(heap, (dt + entry[0],entry[0],i.n,i, entry[4]))
            
            #update heap with wall collissions
            dtw, w = wall_collisions(10,entry[3],subbox)
            if dtw != None:
                heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))
            dtw, w = wall_collisions(10,entry[4],subbox)
            if dtw != None:
                heapq.heappush(heap,(dtw + entry[0],entry[0],entry[4].n,entry[4],w))
                    
            # update time counter
            t = entry[0]
    else:
        # checking if collision is valid event
        if entry[1] < L[entry[3].n]:
            pass
        else: # collision valid
            
            # updating last collision times
            L[entry[3].n] = entry[0]
            
            # new particle pos and vel
            posi, veli, posj, velj = collision(entry)
            
            #save previous pos and vel
            simulation.append([entry[0], entry[3].n, posi, veli, entry[4]])

            # update pos and vel
            entry[3].update(posi, veli)
            
            # update heap
            for i in IS:
                # collisions with first sphere
                dt = check_collision(i, entry[3])
                if dt != None:
                    heapq.heappush(heap, (dt + entry[0],entry[0],i.n, i, entry[3]))
            
            #update heap with wall collissions
            dtw, w = wall_collisions(10,entry[3],subbox)
            if dtw != None:
                heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))
                
            # update time counter
            t = entry[0]

############### TRY TO RUN THE COLLISION ##########################################################################################


print("rank and heap",rank,heap)
print("rank and simulation",rank,simulation)



## Our code 
### Denote end of mpi calls.
MPI.Finalize()
###  