#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
   
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int name_len, core_id = sched_getcpu();
    

    int workload = 1;
    printf("Rank %d process, running on core %d, workload duration: %d seconds\n", 
    	world_rank, core_id, workload);

    double start_time = MPI_Wtime(); 
    sleep(workload);    
    double end_time = MPI_Wtime();   

    printf("Rank %d process finished. Execution time: %.2f seconds\n", 
    	world_rank, end_time - start_time);

    MPI_Finalize();
    return 0;
}

