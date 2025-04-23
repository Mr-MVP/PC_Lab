#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char strings[4][4] = {{'V', 'J', 'T', 'I'}, {'V', 'J', 'T', 'I'}, {'V', 'J', 'T', 'I'}, {'V', 'J', 'T', 'I'}};
    char recv_str[4];

    MPI_Scatter(strings, 4, MPI_CHAR, recv_str, 4, MPI_CHAR, 0, MPI_COMM_WORLD);

    printf("Process %d received string: %s\n", rank, recv_str);

    MPI_Finalize();
    return 0;
}