#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// operation that will be done on the GPU
__global__ void matMulKernel(const double* A, const double* B, double* C,int m, int k, int n) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) // out of bounds check
    {
        double sum = 0.0;
        for (int t = 0; t < k; ++t) 
        {
            sum += A[row * k + t] * B[t * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() 
{
    int m, k, k2, n;

    cout << "Enter rows and columns of matrix A: ";
    cin >> m >> k;

    cout << "Enter rows and columns of matrix B: ";
    cin >> k2 >> n;

    if (k != k2) 
    {
        cout << "Inner dimensions must match (A: m x k, B: k x n).\n";
        return 1;
    }

    size_t sizeA = (size_t)m * k * sizeof(double);
    size_t sizeB = (size_t)k * n * sizeof(double);
    size_t sizeC = (size_t)m * n * sizeof(double);

    vector<double> h_A(m * k);
    vector<double> h_B(k * n);
    vector<double> h_C(m * n);

    srand((unsigned)time(nullptr));

    for (int i = 0; i < m * k; ++i) h_A[i] = rand() % 10;
    for (int i = 0; i < k * n; ++i) h_B[i] = rand() % 10;

    // Print A
    cout << "\nMatrix A (" << m << " x " << k << "):\n";
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j) 
        {
            cout  << h_A[i * k + j] << " ";
        }
        cout << "\n";
    }

    // Print B
    cout << "\nMatrix B (" << k << " x " << n << "):\n";
    for (int i = 0; i < k; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            cout  << h_B[i * n + j] << " ";
        }
        cout << "\n";
    }
    cout << endl;

    // device memory
    double* d_A, * d_B, * d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + 15) / 16, (m + 15) / 16);

    matMulKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, m, k, n);

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print C
    cout << "Result matrix C =:\n";
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            cout  << h_C[i * n + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

