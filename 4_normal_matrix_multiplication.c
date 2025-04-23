#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#define N 500

void multiplication(int a[N][N], int b[N][N], int r[N][N])
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

void matrix_generate(int matrix[N][N])
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
    int a[N][N];
    int b[N][N];
    int r[N][N];

    matrix_generate(a);
    matrix_generate(b);

    clock_t start_time = clock();
    multiplication(a, b, r);
    clock_t end_time = clock();

    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Serial: %f seconds.\n", execution_time);

    return 0;
}