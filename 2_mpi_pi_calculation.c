#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double step = 1.0 / n;
    int local_size = n / size;
    int start = rank * local_size;
    int end = start + local_size;

    double psum = 0.0;
    for (int i = start; i < end; ++i)
    {
        double x = (i + 0.5) * step;
        psum += 4.0 / (1.0 + x * x);
    }

    double total = 0.0;
    MPI_Reduce(&psum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi = total * step;
    printf("PI: %.12f\n", pi);

    MPI_Finalize();
    return 0;
}
