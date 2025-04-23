#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char **argv)
{
    int rank, size;
    int array_size = 7;
    int array[array_size]; 
    int val,max;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        for (int i = 0; i < array_size; i++)
        {
            array[i] = rand() % 101;
            printf("array[%d] = %d\n", i, array[i]);
        }
    }

    MPI_Scatter(array, 1, MPI_INT, &val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d has element: %d\n", rank, val);

    MPI_Reduce(&val, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Maximum value is: %d\n", max);
    }

    MPI_Finalize();

    return 0;
}
