#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size, token;

    MPI_Init(&argc, &argv);               
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int next = (rank + 1) % size;        
    int prev = (rank - 1 + size) % size; 

    if (rank == 0)
    {
        token = 0;
        printf("Process %d starts with token %d\n", rank, token);
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    
    MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    token += rank;
    printf("Process %d received token %d from Process %d\n", rank, token, prev);

    MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Final token at Process %d: %d (Sum of ranks)\n", rank, token);
    }

    MPI_Finalize(); 
    return 0;
}
