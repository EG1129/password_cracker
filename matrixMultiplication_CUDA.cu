#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

__global__ void matMulKernel(const double* A, const double* B, double* C,
                             int r, int c, int c1)
{
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;   // row of A, row of C
    int c1_idx = blockIdx.x * blockDim.x + threadIdx.x;  // column of B, column of C

    if (r_idx < r && c1_idx < c1)
    {
        double sum = 0.0;

        for (int t = 0; t < c; ++t)   // shared dimension: A's c, B's r1
        {
            int A_index = r_idx * c + t;       // A[r_idx][t]
            int B_index = t * c1 + c1_idx;     // B[t][c1_idx]
            sum += A[A_index] * B[B_index];
        }

        C[r_idx * c1 + c1_idx] = sum;          // C[r][c1]
    }
}

int main()
{
    int r, c;       // A: r x c
    int r1, c1;     // B: r1 x c1

    cout << "Enter rows and columns of matrix A (r c): ";
    cin >> r >> c;

    cout << "Enter rows and columns of matrix B (r1 c1): ";
    cin >> r1 >> c1;

    size_t sizeA = (size_t)r * c * sizeof(double);
    size_t sizeB = (size_t)r1 * c1 * sizeof(double);
    size_t sizeC = (size_t)r * c1 * sizeof(double);

    vector<double> h_A(r * c);
    vector<double> h_B(r1 * c1);
    vector<double> h_C(r * c1);

    srand((unsigned)time(nullptr));

    for (int i = 0; i < r * c; ++i) h_A[i] = rand() % 10;
    for (int i = 0; i < r1 * c1; ++i) h_B[i] = rand() % 10;

    // Print A, can comment this out for faster results
    cout << "\nMatrix A (" << r << " x " << c << "):\n";
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            cout << h_A[i * c + j] << " ";
        }
        cout << "\n";
    }

    // Print B, can comment this out for faster results
    cout << "\nMatrix B (" << r1 << " x " << c1 << "):\n";
    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < c1; ++j)
        {
            cout << h_B[i * c1 + j] << " ";
        }
        cout << "\n";
    }
    cout << endl;

    // Allocate device memory
    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((c1 + 15) / 16, (r + 15) / 16);

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, r, c, c1);

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // print result matrix C
    cout << "Result matrix C (" << r << " x " << c1 << "):\n";
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c1; ++j)
        {
            cout << h_C[i * c1 + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



