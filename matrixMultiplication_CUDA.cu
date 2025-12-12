#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// GPU KERNEL
__global__ void matMulKernel(double* A, double* B, double* C,
    int r, int c, int c1)
{
    int r_idx = blockIdx.y * blockDim.y + threadIdx.y;   // row of A, row of C
    int c1_idx = blockIdx.x * blockDim.x + threadIdx.x;  // column of B, column of C

    if (r_idx < r && c1_idx < c1)
    {
        double sum = 0.0;

        for (int t = 0; t < c; ++t)   
        {
            int A_index = r_idx * c + t;       // A[r_idx][t]
            int B_index = t * c1 + c1_idx;     // B[t][c1_idx]
            sum += A[A_index] * B[B_index];
        }

        C[r_idx * c1 + c1_idx] = sum;
    }
}

int main()
{
    int r, c;       // A dimensions
    int r1, c1;     // B dimensions

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

    // Allocate device memory
    double* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    
    auto start = high_resolution_clock::now();

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((c1 + 15) / 16, (r + 15) / 16);

    matMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, r, c, c1);
    cudaDeviceSynchronize(); // ensure kernel completed

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    auto end = high_resolution_clock::now();

    double total_time = duration<double, milli>(end - start).count();
    cout << "\nTotal GPU Execution Time: " << total_time << " ms\n\n";

    //// Print C
    //cout << "Result matrix C (" << r << " x " << c1 << "):\n";
    //for (int i = 0; i < r; ++i)
    //{
    //    for (int j = 0; j < c1; ++j)
    //    {
    //        cout << h_C[i * c1 + j] << " ";
    //    }
    //    cout << "\n";
    //}

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}





