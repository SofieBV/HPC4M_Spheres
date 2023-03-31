
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Sphere():
    def __init__(self, n, pos, vel, rad, mass):
        self.n = n
        self.pos = pos
        self.vel = vel
        self.rad = rad
        self.mass = mass

    def update(self, new_pos, new_vel):
        self.pos = new_pos
        self.vel = new_vel

def initial_state(nr, positions, velocities, rad, mass):
    IS = []
    for n in range(nr):
        IS.append(Sphere(n,positions[n], velocities[n], rad[n], mass[n]))
    return IS

def check_collision(i,j):
    r = i.pos - j.pos
    v = i.vel - j.vel
    rnorm = np.linalg.norm(r)
    rnorm2 = rnorm*rnorm
    rv = np.dot(r,v)
    rv2 = rv*rv
    vnorm2 = np.linalg.norm(v)*np.linalg.norm(v)
    s = abs(i.rad + j.rad)
    s2 = s*s
    if s < rnorm: #condition 1, eq5
        if rv < 0: # condition 2, eq6
            if rnorm2-rv2/vnorm2 < s2: #condition 3, eq7
                dt = (rnorm2 - s2) / (-rv + np.sqrt(rv2 - (rnorm2-s2)*vnorm2 ))
                if dt > 0:
                    return dt
    return None

#### below is yet to be incorportated in simulation ####
def wall_collisions(l, i,subbox1):
    # collisions with lXl square wall centered at (0,0)
    dts = []
    wall = []

    #coordinate of each corner
    topleft = subbox1[0]
    topright = subbox1[1]
    bottomright=subbox1[2]
    bottomleft=subbox1[3]


    if i.vel[0] == 0 and i.vel[1] ==0:
        return None, None
    else:
        # collision with left wall
        if i.vel[0] < 0:
            rx = -(i.pos[0]-i.rad -topleft[0])
            dt = rx/i.vel[0]
            dts.append(dt)
            wall.append('left')
        elif i.vel[0] > 0:  
            # collision with right wall
            rx = topright[0] - (i.pos[0]+i.rad)
            dt = rx/i.vel[0]
            dts.append(dt)
            wall.append('right')
        
        # collision with top wall
        if i.vel[1] > 0:
            ry = topright[1] - (i.pos[1]+i.rad)
            dt = ry/i.vel[1]
            dts.append(dt)
            wall.append('top')
        elif i.vel[1] < 0:
            # collision with bottom wall
            ry = -(i.pos[1]-i.rad -bottomleft[1])
            dt = ry/i.vel[1]
            dts.append(dt)
            wall.append('bottom')
        
        w = wall[np.argmin(dts)]
        dtw = min(dts)

        return dtw, w

def create_heap(IS,subbox):
    heap_list = []
    for i in IS:
        dtw, w = wall_collisions(10,i,subbox)
        if dtw != None:
            heap_list.append((dtw,0,i.n,i,w))
        for jn in range(i.n+1,len(IS)):
            j = IS[jn]
            dt = check_collision(i,j)
            if dt != None:
                heap_list.append((dt,0,i.n,i,j))
    
    heapq.heapify(heap_list)
    return heap_list


def collision(entry):
    # computes new positions and velocities for collison entry in heap
    dt = entry[0] - entry[1] # col time - time included in heap
    
    if not isinstance(entry[4], str):
        # new positions
        posi = entry[3].pos + entry[3].vel*dt
        posj = entry[4].pos + entry[4].vel*dt
        
        # new velocities
        reduced_mass = 2 / (1/entry[3].mass + 1/entry[4].mass) # eq. 12
        unit_vec = (posi - posj) / (entry[3].rad + entry[4].rad) # eq. 13
        mom_change = reduced_mass*np.dot(unit_vec, entry[3].vel - entry[4].vel)*unit_vec # eq. 12
        veli = entry[3].vel - mom_change/entry[3].mass # eq. 10
        velj = entry[4].vel + mom_change/entry[4].mass # eq. 11

        return posi, veli, posj, velj
    else:
        posi = entry[3].pos + entry[3].vel*dt
        if entry[4] == 'bottom' or entry[4] == 'top':
            veli = np.array([entry[3].vel[0],-entry[3].vel[1]])
        else:
            veli = np.array([-entry[3].vel[0],entry[3].vel[1]])
        
        return posi, veli, None, None


def update_heap(L,t,simulation,entry,initial_state,heap,subbox,special_walls_subbox,comm,rank):
    t=entry[0]
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
            for i in initial_state:
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

    ### Collision with a wall, something extra needs to be done if it's a special wall
    else:
        # checking if collision is valid event
        if entry[1] < L[entry[3].n]:
            pass
        else: # collision valid

            ## subbox[4] contains a list of string with the special walls, if the wall concerned is in the list of the special wall:
            if entry[4] in special_walls_subbox[0]:
                index_wall = special_walls_subbox[0].index(entry[4])
                rank_receive = special_walls_subbox[1][index_wall] #gives the corresponding rank of the subbox that receives the information
            ###############         WORK IN PROGRESS       ###############
            ##############################################################
            ##### If one of them hit a special wall: 
            ##### The 2 boxes need to communicate, one can assume that it is always the rank 0 processor that handles the shits 
            ##### In the case of a particle crossing a special wall in the subbox 2 
                if rank == 1:
                    ## in this case, it is pretty simple, because we only have two subbox, 
                    ## if we were to have more subbox, we would need to handle in which other processor
                    ## this particle crossing the wall would end up in. 
                    ## let's stick to a simple case for now. 
                    comm.send(entry, dest=rank_receive, tag=1)
                    if rank==rank_receive:
                        entry_received = comm.recv(source=1, tag=1)
                        ##################### TO COMPLETE #############################
                        ## do something about that received entry
                    
                elif rank== 0: 
                    comm.send(entry, dest=rank_receive, tag=1)
                    if rank==rank_receive:
                        entry_received = comm.recv(source=0, tag=1)
                        ##################### TO COMPLETE #############################
                        ## do something about that received entry, most likely what is below:
                        ########## then tell the box that is being crossed (and add the right rank condition)
                        ########## Send info from subbox1 to subbox 2  (from rank0 to rank1) : 
                        ################## Subbox 2 needs to receive it and check the time. if proc2 time < proctime 1
                        ################## They need to receive the info of the particule, recompute their heap for this particle and go back to time of proc 1 
                        ################## We need to check for forward in time (might still send a particle to the other side and we need to know that) 
                        ################## and backward in time  
            ##################################################################


                ############ do something
                pass

            #### 
            else:
                # updating last collision times
                L[entry[3].n] = entry[0]
                
                # new particle pos and vel
                posi, veli, posj, velj = collision(entry)
                
                #save previous pos and vel
                simulation.append([entry[0], entry[3].n, posi, veli, entry[4]])

                # update pos and vel
                entry[3].update(posi, veli)
                
                # update heap
                for i in initial_state:
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

    return (L,t,simulation,entry,heap)