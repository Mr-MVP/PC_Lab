#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>



int main(int argc, char *argv[]) {
  int rank, size;
  double i, local_count = 0, total_count;
  double local_points,x,y,pi;
  int total_points = 1000000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  local_points = total_points / size;

#pragma omp parallel private(x, y) reduction(+ : local_count)
  {
#pragma omp for
    for (i = 0; i < local_points; i++) {
      x = (double)rand() / RAND_MAX;
      y = (double)rand() / RAND_MAX;
      if (x * x + y * y <= 1.0) {
        local_count++;
      }
    }
  }

  MPI_Reduce(&local_count, &total_count, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    pi = 4.0 * (double)total_count / total_points;
    printf("PI: %.12f\n", pi);
  }

  MPI_Finalize();
  return 0;
}
