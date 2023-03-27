import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
MPI.Init()
###

### the actual program.

comm = MPI.COMM_WORLD   # the global communicator.

rank = comm.Get_rank()  # get process ID within comm.



## lowercase version for generic objects, uses pickle under the hood.

data = None

if rank == 0:
    data = [1,2,"hello"]
    comm.send(data, dest=1, tag=1)
elif rank == 1:
    data = comm.recv(source=0, tag=1)
    data += ["world"]
    
print("This is rank {} and my data is {}.".format(rank, data))


## uppercase version for contiguous memory blocks such as arrays. more efficient but less generic.

# data = None

# if rank == 0:
#     data = np.array([1.,2.])
#     comm.Send(data, dest=1, tag=1)
# elif rank == 1:
#     data = np.empty(2)
#     comm.Recv(data, source=0, tag=1)
#     data += 2
    
# print("This is rank {} and my data is {}.".format(rank, data))


### end of actual program.


### denote end of mpi calls.
MPI.Finalize()
###