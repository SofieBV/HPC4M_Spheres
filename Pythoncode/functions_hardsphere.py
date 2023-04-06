
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
    i = 0
    for n in nr:
        IS.append(Sphere(n,positions[i], velocities[i], rad[i], mass[i]))
        i+=1
    return IS

def check_collision(i, new_posi, j):
    r = new_posi - j.pos
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
def wall_collisions(i,subbox1):
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
            rx = topleft[0]+i.rad - i.pos[0]
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
        dtw, w = wall_collisions(i,subbox)
        if dtw != None:
            heap_list.append((dtw,0,i.n,i,w))
        for jn in range(i.n+1,len(IS)):
            j = IS[jn]
            dt = check_collision(i, i.pos, j)
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

def First(a):
    #### takes first element of list's tuple
    #### for use in combine_sim
    return a[0]


def combine_sim(firstlist, secondlist):
    #### firstlist is the simulation output for one box
    #### secondlist is the simulation output for the other
    
    # appends all elements in second list to first
    for i in secondlist:
         firstlist.append(i)

    # sorts combined list by First, the first element
    firstlist.sort(key=First)
    return firstlist

def simulate(sim_info, subbox, T, steps, inputs, name='animation'):
    ## inputs contain a list of inputs for positions, velocities and radius
    #plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'

    global pos, vel
    #coordinate of each corner
    topright = subbox[1]
    bottomleft=subbox[3]

    # positions, velocities and radius info
    pos = np.array(inputs[0])
    vel = np.array(inputs[1])
    rad = inputs[2]

    # step size for the simulations
    dt = T/steps

    # x and y coordinates of the positions 
    xx = pos[:,0]
    yy = pos[:,1]
    pts_rad = (100*rad)**2  #size of the radius converted to matplotlib size 


    # create a figure 
    fig, ax = plt.subplots()
    points = ax.scatter(xx,yy,s = pts_rad) # plotting all the particles coordinates 
    ax.set_ylim(bottomleft[1],topright[1])
    ax.set_xlim(bottomleft[0],topright[0])

    # define this function inside the simulation function for the purpose of updating the plot (package for the animation)
    def update(i):
        # define as global to make them work properly
        global pos, vel

        # Sim_info is the list of all the collision that are happening, 
        # sim_info[0] is the first entry for the heap that will happen
        # Using the info from the first event, we update only the particles that concerned by 
        # this collision and the rest of the particle continue their life happily with their 
        # originel velocities and positions
        if len(sim_info) != 0:
            sim = sim_info[0]
            t = dt*i
            # when we get to the collision time
            if t > sim[0]:
                print('sim time', t, sim[0])
                # if the lenght is 7, it is a collision bw particles 
                if len(sim)==7:
                    pos[sim[1],:] = sim[3] # new position of particle 1 
                    pos[sim[2],:] = sim[4] # new position of particle 2 
                    vel[sim[1],:] = sim[5] # new velocity of particle 1 
                    vel[sim[2],:] = sim[6] # new velocity of particle 2
                    pos[sim[1],:] = pos[sim[1],:] + vel[sim[1],:]*(t-sim[0])
                    pos[sim[2],:] = pos[sim[2],:] + vel[sim[2],:]*(t-sim[0])
                else: 
                    # if the lenght is less, it is a collision bw particle-wall 
                    pos[sim[1],:] = sim[2] # new position of particle 1
                    vel[sim[1],:] = sim[3] # new velocity of particle 1
                    pos[sim[1],:] = pos[sim[1],:] + vel[sim[1],:]*(t-sim[0])
                
                sim_info.pop(0) # pop out the first entry as we used it 
                
            #else:
                
            pos = pos + vel*dt # for all of the particles, work out where they will be next

            # and this is the plotting
            xx = pos[:,0]
            yy = pos[:,1]
            ax.cla()
            ax.scatter(xx,yy,s = pts_rad)
            ax.set_ylim(bottomleft[1],topright[1])
            ax.set_xlim(bottomleft[0],topright[0])

    def generate_points():
        ax.scatter(xx,yy)


    ani = animation.FuncAnimation(fig, update, init_func=generate_points, frames = steps, interval = T)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=100, bitrate=1800)
    FFwriter = animation.FFMpegWriter(fps = 100)
    ani.save(name+".gif", writer=FFwriter)