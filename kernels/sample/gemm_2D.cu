#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include "../src/base_gemm.cu"


#define M 2048
#define K 1024
#define N 2048

#ifdef INT
	#define DT int
#elif FLOAT
	#define DT float
#else
	#define DT int
#endif


// CPU gemm
void cpu_gemm(DT* A, DT* B, DT* C, int m, int n, int k) {
	for (int idx_m = 0; idx_m < m; ++idx_m) {
		for (int idx_n = 0; idx_n < n; ++idx_n) {
			for (int idx_k = 0; idx_k < k; ++idx_k) {
				C[idx_n + idx_m * n] += A[idx_n * k + idx_k] * B[idx_m + idx_k * k];
			}
		}
	}
}


int main(int argc, char* argv[]) {
	if (argc > 1) {
		
	}
	int blockSizeM = 32;
	int blockSizeN = 32;
	int threadSizeM = 1;
	int threadSizeN = 1;

	DT *A = new DT[M * K];
	DT *B = new DT[K * N];
	DT *C = new DT[M * N];
	std::fill_n(A, M * K, 1);
	std::fill_n(B, K * N, 1);
	std::fill_n(C, M * N, 0);
	
	// CPU gemm
	cpu_gemm(A, B, C, M, N, K);
	
	//GPU gemm
	//  init device
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	cudaSetDevice(dev);
	
	//  set block and thread size
	dim3 grid(M/blockSizeM, N/blockSizeN);
	dim3 block(blockSizeM / threadSizeM, blockSizeN / threadSizeN);
	
	//  alloc device memory
	DT *D_A, *D_B, *D_C;
	cudaMalloc((DT**)(&D_A), M * K * sizeof(DT));
	cudaMalloc((DT**)(&D_B), N * K * sizeof(DT));
	cudaMalloc((DT**)(&D_C), M * N * sizeof(DT));

	cudaMemcpy(D_A, A, M * K * sizeof(DT), cudaMemcpyHostToDevice);
	cudaMemcpy(D_B, B, N * K * sizeof(DT), cudaMemcpyHostToDevice);

	gemm_base_2D<DT, DT, DT><<<grid, block>>>(D_A, D_B, D_C, M, N, K);
	cudaDeviceSynchronize();
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		//Error check
		std::cerr << "CUDA kernel error: " << cudaGetErrorString(cudaError) << std::endl;
	}

	//Correctness check
	DT *H_C = new DT[M * N];
	cudaMemcpy(H_C, D_C, N * M * sizeof(DT), cudaMemcpyDeviceToHost);

	bool flag = true;
	for (int i = 0; i < M; ++i) {
		if (flag)
			for (int j = 0; j < N; ++j) {
				if (H_C[i * N + j] != C[i * N + j]) {
					std::cout << "wrong" << std::endl;
					std::cout << H_C[i * N + j]  << " " << i << " " << j << std::endl;
					flag = false;
					break;
				}
			}
	}
	
	delete A;
	delete B;
	delete C;
	delete H_C;

}
