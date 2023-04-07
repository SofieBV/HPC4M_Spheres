import numpy as np

### INITIALIZE MPI ######################################################################################
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nr_proc = comm.Get_size()
#########################################################################################################


### COMPUTE THE   SUM ON EACH RANK ################################################################
print(rank)
N = 840                                 # total number of terms in the sum to compute.

N_i = N / nr_proc                       # number of terms computed by each process (CARE: need N % nr_proc == 0).

partial_sum = 0
start_index = int(rank * N_i + 1)       # the sum indices the process is going to cover.
end_index = int(start_index + N_i)

for i in range( start_index, end_index ):

    partial_sum += 1 / ( 1 + ( (i-0.5)/N )**2 )     # compute the partial sum.

print("This is process {} and I have added terms from i={} to {}.".format(rank, start_index, end_index-1))
##########################################################################################################


### COMMUNICATION TO OBTAIN THE FINAL RESULT #############################################################

### Using collective communication.

# partial_sum = np.array([partial_sum], dtype=np.float64)   # turn into np.array (contiguous memory object).

# if rank==0: result = np.zeros(1, dtype=np.float64)        # create receive buffer on receiving process.

# else: result = None                                       # make the name known on all other processes.

# comm.Reduce(partial_sum, result, op=MPI.SUM, root=0)      # the result is collected on process 0.



### Using P2P communication

# There are various ways to do this. Here, the process of rank n is going to send its partial sum
# to the process of rank n-1. Process n-1 adds the received number to its own partial sum, and then
# sends the results to rank n-2, and so on. 


partial_sum = np.array([partial_sum], dtype=np.float64)   # turn into np.array (contiguous memory object).

if rank != nr_proc-1:
    received = np.zeros(1, dtype=np.float64)    # receive buffer.
    comm.Recv(received, source=rank+1)          # receive the message.
    partial_sum += received                     # add received value to the partial sum.

if rank != 0: comm.Send(partial_sum, dest=rank-1)   # send the partial sums to process of rank-1.

if rank == 0: 
    result = 4/N * partial_sum                  # rank 0 holds the final result.
    
    print("This is process {}. The final result is {}.".format(rank, result))
#############################################################################################################

### Denote end of mpi calls.
MPI.Finalize()
###  