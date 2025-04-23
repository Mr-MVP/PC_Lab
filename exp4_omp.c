#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 500

void multiplation(int a[N][N], int b[N][N], int c[N][N])
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) shared(a, b, c)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0;
            for (k = 0; k < N; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main()
{
    int a[N][N], b[N][N], c[N][N];
    int i, j;
    double start_time, end_time;

    srand(time(NULL));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = rand() % 100;
            b[i][j] = rand() % 100;
        }
    }

    start_time = omp_get_wtime();
    multiplation(a, b, c);
    end_time = omp_get_wtime();

    printf("Parallel: %f seconds\n", end_time - start_time);

    return 0;
}