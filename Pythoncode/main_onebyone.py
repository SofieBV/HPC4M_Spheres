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
# when the particle hits the first special wall, it's the rank 1 that needs to receive it
# for example if the particle hits the wall bottom, rank 1 contains the bottom subbox and should be ready to receive it. 
special_walls_subbox1=[["bottom"],[1]] 
subbox2=[[-l/2,0],[l/2,0],[l/2,-l/2],[-l/2,-l/2]]
special_walls_subbox2=[["top"],[0]] 


if rank==0:
    inputs_nr = 2
    inputs_pos =  [np.array([0.3,0.1]), np.array([0.5,0.1])]
    inputs_vel =  [np.array([-1,0]), np.array([0,0])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=subbox1
    special_walls_subbox = special_walls_subbox1
    IS = initial_state(2,inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)

elif rank==1:
    inputs_nr = 2
    inputs_pos =  [np.array([-1,-0.1]), np.array([1,-1])]
    inputs_vel =  [np.array([0,1]), np.array([-1,0])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=subbox2
    special_walls_subbox = special_walls_subbox2
    IS = initial_state(2,inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)
else:
    heap=[]

############### TRY TO RUN THE COLLISION ##########################################################################################
#### initialise state and run collisions up to time T ####
#### throws up error due to lack of wall ####

T = 10
L = -np.ones(len(IS)) # last collision time for each atom
simulation = [] # any collisions that happened. 
oldheap = [] # any collisions that have not happened, we need to save if we ever need them afterward

t = 0
# possible collision
entry = heapq.heappop(heap)

while t < T:

    print(rank,entry)

    n = 1
    if rank == 0:
        comm.send(entry[0], dest=1, tag=1)
    elif rank == 1:
        comm.send(entry[0], dest=0, tag=1)
        t_r0 = comm.recv(source=0, tag=1)
        if entry[0] >= t_r0:
            n = 0
            t = t_r0 # update time counter
        else:
            t = entry[0] # update time counter
    
    if rank == 0:
        t_r1 = comm.recv(source=1, tag=1)
        if entry[0] <= t_r1:
            n = 0
            t = entry[0] # update time counter
        else:
            t = t_r1 # update time counter
    
    if rank == n:      
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
                dtw, w = wall_collisions(entry[3],subbox)
                if dtw != None:
                    heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))
                dtw, w = wall_collisions(entry[4],subbox)
                if dtw != None:
                    heapq.heappush(heap,(dtw + entry[0],entry[0],entry[4].n,entry[4],w))
            
            # send that it's all good
            comm.send(None, dest=1-n, tag=2)         

        ### Collision with a wall, something extra needs to be done if it's a special wall
        else:
            # checking if collision is valid event
            if entry[1] < L[entry[3].n]:
                # send that it's all good
                comm.send(None, dest=1-n, tag=2)  
                pass
            else: # collision valid
                # updating last collision times
                L[entry[3].n] = entry[0]

                ## subbox[4] contains a list of string with the special walls, if the wall concerned is in the list of the special wall:
                if entry[4] in special_walls_subbox[0]:
                    index_wall = special_walls_subbox[0].index(entry[4])
                    rank_receive = special_walls_subbox[1][index_wall] #gives the corresponding rank of the subbox that receives the information
                ###############         WORK IN PROGRESS       ###############
                    
                    comm.send(entry, dest=rank_receive, tag=2)
                    ## still need some way to check if anything is being sent!

                #### 
                else:                    
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
                    dtw, w = wall_collisions(entry[3],subbox)
                    if dtw != None:
                        heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))
                    
                    # send that it's all good
                    comm.send(None, dest=1-n, tag=2)  
        # get new entry
        entry = heapq.heappop(heap)
    
    elif rank == n-1:
        new_entry = comm.recv(source=n, tag=2)
        if new_entry != None:
            print('do something')
            
            # updating last collision times
            L[new_entry[3].n] = new_entry[0]
            
            veli = entry[3].vel
            dt = 2*entry[3].rad / veli[1]
            posi = entry[3].pos + veli*dt
            # update pos and vel
            entry[3].update(posi, veli)
            
            #save previous pos and vel
            simulation.append([entry[0], entry[3].n, posi, veli, entry[4]])
            
            # update heap
            for i in IS:
                # collisions with first sphere
                dt = check_collision(i, entry[3])
                if dt != None:
                    heapq.heappush(heap, (dt + entry[0],entry[0],i.n, i, entry[3]))
            IS.append(entry[3])
            
            #update heap with wall collissions
            dtw, w = wall_collisions(entry[3],subbox)
            if dtw != None:
                heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))


############### TRY TO RUN THE COLLISION ##########################################################################################


print("rank and heap",rank,heap)
print("rank and simulation",rank,simulation)



## Our code 
### Denote end of mpi calls.
MPI.Finalize()
###  