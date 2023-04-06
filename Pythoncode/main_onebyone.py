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

############################### DEFINING THE INITIAL CONDITIONS #############################################
l=5 #size of the box

# Each processor handle a subbox of coordinates:
# [top left, top right, bottom right, bottom left]
# when the particle hits the first special wall, it's the rank 1 that needs to receive it
# for example if the particle hits the wall bottom, rank 1 contains the bottom subbox and should be ready to receive it. 

if rank==0:
    inputs_nr = 2
    inputs_pos =  [np.array([0.3,0.2]), np.array([1,0.6])]
    inputs_vel =  [np.array([-0.97,0]), np.array([0,-1])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox= [[-l/2,l/2],[l/2,l/2],[l/2,0],[-l/2,0]] 
    special_walls_subbox = [["bottom"],[1]] 
    IS = initial_state([0,1],inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)

elif rank==1:
    inputs_nr = 2
    inputs_pos =  [np.array([-1,-0.2]), np.array([1,-1])]
    inputs_vel =  [np.array([0,1]), np.array([0,-0.1])]
    inputs_rad =   0.1*np.ones(2)
    inputs_mass =  0.1*np.ones(2)
    subbox=[[-l/2,0],[l/2,0],[l/2,-l/2],[-l/2,-l/2]]
    special_walls_subbox = [["top"],[0]]
    IS = initial_state([2,3],inputs_pos,inputs_vel,inputs_rad,inputs_mass)
    heap = create_heap(IS,subbox)
else:
    heap=[]

############### RUN THE COLLISION ##########################################################################################
#### initialise state and run collisions up to time T ####
#### throws up error due to lack of wall ####

T = 10 # until time T
L = -np.ones(4) # last collision time for each atom,
simulation = [] # any collisions that happened. 
t = 0 # intitialise time

# run the time loop
while t < T:

    # get new entry
    entry = heapq.heappop(heap)
    # print to verify
    # print(rank,entry)

    # We are trying to figure which processor should start running and eventually sending a message to the other. 
    n = 1 # intialise the processor (rank 1 is the default one)
    # from box 0(proc 0) to box 1 (proc 1), send the time of the first collision
    if rank == 0: 
        comm.send(entry[0], dest=1, tag=1)
    # from box 1(proc 1) to box 0 (proc 0), send the time of the first collision but also receive the time send by box 1
    elif rank == 1:
        comm.send(entry[0], dest=0, tag=1)
        #while not comm.Iprobe(source = 0, tag=1):
        #    x =1
        t_r0 = comm.recv(source=0, tag=1)
        # we compare the first collision time to see which proc should go first
        if entry[0] >= t_r0: # if time of first collision of rank 1 is larger than the time of the first collision of rank 0
            n = 0 # then we should run rank 0 
            t = t_r0 # update gloabl time counter
            heapq.heappush(heap, entry) # now that we have seen that our first collision in rank 1 is not of interest, push it back to the heap

        else:
            t = entry[0] # update time counter
    
    if rank == 0:
        #while not comm.Iprobe(source = 1, tag=1):
        #    x=1
        t_r1 = comm.recv(source=1, tag=1)
        if entry[0] <= t_r1:
            n = 0
            t = entry[0] # update time counter
        else:
            t = t_r1 # update time counter
            heapq.heappush(heap, entry)
    
    if rank == n:      
        if not isinstance(entry[4],str):
            # checking if collision is valid event
            if entry[1] < L[entry[3].n]: 
                pass
            elif entry[1] < L[entry[4].n]:
                pass
            else: # collision valid
                # update the position of the particle with the global time
                new_pos = entry[3].pos + entry[3].vel*(entry[1] - L[entry[3].n])
                entry[3].update(new_pos, entry[3].vel)
                    
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
                     # let's first evaluate position of that sphere at this time
                    new_pos = i.pos + i.vel*(entry[0] - L[i.n])
                    
                    # collisions with first sphere
                    dt = check_collision(i, new_pos, entry[3])
                    if dt != None:
                        heapq.heappush(heap, (dt + entry[0],entry[0],i.n,i, entry[3]))
                    # collisions with second
                    dt = check_collision(i, new_pos, entry[4])
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
                    IS.remove(entry[3])
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
                        # let's first evaluate position of that sphere at this time
                        new_pos = i.pos + i.vel*(entry[0] - L[i.n])
                        # collisions with first sphere
                        dt = check_collision(i,new_pos, entry[3])
                        if dt != None:
                            heapq.heappush(heap, (dt + entry[0],entry[0],i.n, i, entry[3]))
                    
                    #update heap with wall collissions
                    dtw, w = wall_collisions(entry[3],subbox)
                    if dtw != None:
                        heapq.heappush(heap,(dtw + entry[0],entry[0],entry[3].n,entry[3],w))
                    
                    # send that it's all good
                    comm.send(None, dest=1-n, tag=2)  
    
    elif rank == 1-n:
        #while not comm.Iprobe(source = n, tag=2):
        #    x=1
        new_entry = comm.recv(source=n, tag=2)

        if new_entry != None:
            #print('do something', new_entry)
            
            # updating last collision times
            L[new_entry[3].n] = new_entry[0]
            
            veli = new_entry[3].vel
            if new_entry[4] == 'bottom' or new_entry[4] == 'top':
                dts = new_entry[0] - new_entry[1] + 2*new_entry[3].rad / abs(veli[1])
            else:
                dts = new_entry[0] - new_entry[1] + 2*new_entry[3].rad / abs(veli[0])
            
            posi = new_entry[3].pos + veli*dts
            # update pos and vel
            new_entry[3].update(posi, veli)
            
            #save previous pos and vel
            simulation.append([new_entry[0], new_entry[3].n, posi, veli, new_entry[4]])
            IS.append(new_entry[3])

            # update heap
            for i in IS:
                # let's first evaluate position of that sphere at this time
                new_pos = i.pos + i.vel*(new_entry[0] - L[i.n])
                # collisions with first sphere
                
                dt = check_collision(i,new_pos, new_entry[3])
                if dt != None:
                    heapq.heappush(heap, (dt + new_entry[0],new_entry[0],i.n, i, new_entry[3]))
                        
            #update heap with wall collissions
            dtw, w = wall_collisions(new_entry[3],subbox)
            if dtw != None:
                heapq.heappush(heap,(dtw + new_entry[0],new_entry[0],new_entry[3].n,new_entry[3],w))

############### TRY TO RUN THE COLLISION ##########################################################################################


#print("rank and heap",rank,heap)
print("rank and simulation",rank,simulation)

if rank == 1: 
    comm.send([simulation, inputs_pos, inputs_vel, inputs_rad], dest=0, tag=3)
# from box 1(proc 1) to box 0 (proc 0), send the time of the first collision but also receive the time send by box 1
elif rank == 0:
    from1 = comm.recv(source=1, tag=3)
    sim_r1 = from1[0]
    inputs_pos1 = from1[1]
    inputs_vel1 = from1[2]
    inputs_rad1 = from1[3]
    sim_total = combine_sim(simulation, sim_r1)
    for i in inputs_pos1:
         inputs_pos.append(i)
    for i in inputs_vel1:
         inputs_vel.append(i)

    simulate(sim_total, [[-l/2,l/2],[l/2,l/2],[l/2,-l/2],[-l/2,-l/2]], T, 100, [inputs_pos, inputs_vel,0.1*np.ones(4)], name='animation{}'.format(rank), parallel = True)



## Our code 
### Denote end of mpi calls.
MPI.Finalize()
###  