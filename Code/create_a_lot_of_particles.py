import numpy as np

def create_particles(N,l):
    ## Function to create a lot of particle inputs 
    np.random.seed(1234)

    # N number of particle in each box
    input_pos_1 = []
    input_pos_2 = []
    input_pos_all= []
    input_vel_1 = []
    input_vel_2 = []
    input_vel_all=[]
    name_box1 = []
    name_box2 = []
    name_box_all=[]


    for i in range(2*N):
        #Fill the first box: 
        if i<N: # needs to be strictly under
            # Box 1: upper box
            # positions
            posbox = np.array([round(np.random.uniform(-l/2+0.1,l/2-0.1),3),round(np.random.uniform(0.1,l/2-0.1),3)])
            input_pos_1.append(posbox)

            # velocities
            velbox= np.array([round(np.random.uniform(-1,1),3), round(np.random.uniform(-20,20),3)])
            input_vel_1.append(velbox)

            # name 
            name_box1.append(i)


        #Fill the second box: 
        else:
            # Box 2: lower box
            # positions
            posbox= np.array([round(np.random.uniform(-l/2+0.1,l/2-0.1),3), round(np.random.uniform(-l/2+0.1,-0.1),3)])
            input_pos_2.append(posbox)

            # velocities
            velbox= np.array([round(np.random.uniform(-1,1),3), round(np.random.uniform(-20,20),3)])
            input_vel_2.append(velbox)

            # name 
            name_box2.append(i)

        # Simulations inputs: 
        input_pos_all.append(posbox)
        input_vel_all.append(velbox)
        name_box_all.append(i)

    return input_pos_1, input_pos_2, input_pos_all, input_vel_1, input_vel_2, input_vel_all, name_box1, name_box2, name_box_all
