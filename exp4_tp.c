#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MAX_N 500

void serial_multiplication(int N, int a[N][N], int b[N][N], int r[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            r[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void parallel_multiplication(int N, int a[N][N], int b[N][N], int r[N][N])
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(a, b, r)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            r[i][j] = 0;
            for (k = 0; k < N; k++)
            {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void matrix_generate(int N, int matrix[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() % 10;
        }
    }
}

int main()
{
    srand(time(0));
    FILE *file = fopen("execution_times.csv", "w");
    if (!file)
    {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(file, "N,Serial Time,Parallel Time\n");

    for (int N = 1; N <= MAX_N; N++)
    {
        int(*a)[N] = malloc(sizeof(int[N][N]));
        int(*b)[N] = malloc(sizeof(int[N][N]));
        int(*r)[N] = malloc(sizeof(int[N][N]));

        if (!a || !b || !r)
        {
            printf("Memory allocation failed for N=%d\n", N);
            return 1;
        }

        matrix_generate(N, a);
        matrix_generate(N, b);

        clock_t start_serial = clock();
        serial_multiplication(N, a, b, r);
        clock_t end_serial = clock();
        double serial_time = ((double)(end_serial - start_serial)) / CLOCKS_PER_SEC;

        double start_parallel = omp_get_wtime();
        parallel_multiplication(N, a, b, r);
        double end_parallel = omp_get_wtime();
        double parallel_time = end_parallel - start_parallel;

        fprintf(file, "%d,%f,%f\n", N, serial_time, parallel_time);

        free(a);
        free(b);
        free(r);
    }

    fclose(file);
    printf("Execution times recorded in execution_times.csv\n");

    return 0;
}
