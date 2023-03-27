
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
def wall_collisions(l, i):
    # collisions with lXl square wall centered at (0,0)
    dts = []
    wall = []

    if i.vel[0] == 0 and i.vel[1] ==0:
        return None, None
    else:
        # collision with left wall
        if i.vel[0] < 0:
            rx = -(i.pos[0]-i.rad + l/2)
            dt = rx/i.vel[0]
            dts.append(dt)
            wall.append('left')
        elif i.vel[0] > 0:  
            # collision with right wall
            rx = l/2 - (i.pos[0]+i.rad)
            dt = rx/i.vel[0]
            dts.append(dt)
            wall.append('right')
        
        # collision with top wall
        if i.vel[1] > 0:
            ry = l/2 - (i.pos[1]+i.rad)
            dt = ry/i.vel[1]
            dts.append(dt)
            wall.append('top')
        elif i.vel[1] < 0:
            # collision with bottom wall
            ry = -(i.pos[1]-i.rad + l/2)
            dt = ry/i.vel[1]
            dts.append(dt)
            wall.append('bottom')
        
        w = wall[np.argmin(dts)]
        dtw = min(dts)

        return dtw, w

def create_heap(IS,subbox1):
    heap_list = []
    for i in IS:
        dtw, w = wall_collisions(10,i)
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

