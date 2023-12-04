#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../src/softmax_base.cu"


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


// CPU softmax
void cpu_softmax(DT* input, DT* output, int m, int n) {
  int * tmp = new DT[m];

  // max
  std::fill(tmp, tmp + m, std::numeric_limits<DT>::min());
	for (int idx_m = 0; idx_m < m; ++idx_m) {
		for (int idx_n = 0; idx_n < n; ++idx_n) {
       if (input[idx_m * n + idx_n] > tmp[idx_m]) tmp[idx_m] = input[idx_m * n + idx_n];
		}
	}

  // sub and exp
	for (int idx_m = 0; idx_m < m; ++idx_m) {
		for (int idx_n = 0; idx_n < n; ++idx_n) {
       output[idx_m * n + idx_n] = std::exp(input[idx_m * n + idx_n] - tmp[idx_m]);
		}
	}

  // sum
  std::fill(tmp, tmp + m, 0);
	for (int idx_m = 0; idx_m < m; ++idx_m) {
		for (int idx_n = 0; idx_n < n; ++idx_n) {
       tmp[idx_m] += output[idx_m * n + idx_n];
		}
	}

  // div
	for (int idx_m = 0; idx_m < m; ++idx_m) {
		for (int idx_n = 0; idx_n < n; ++idx_n) {
       output[idx_m * n + idx_n] = output[idx_m * n + idx_n] / tmp[idx_m];
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

	DT *A = new DT[M * N];
	DT *B = new DT[M * N];
	DT *C = new DT[M * N];
	std::fill_n(A, M * N, 1);
	std::fill_n(B, M * N, 1);
	std::fill_n(C, M * N, 0);
	
	//  CPU softmax
	cpu_softmax(A, B, M, N);
	
	//  GPU softmax
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
	cudaMalloc((DT**)(&D_A), M * N * sizeof(DT));
	cudaMalloc((DT**)(&D_B), M * N * sizeof(DT));
	cudaMalloc((DT**)(&D_C), M * N * sizeof(DT));

	cudaMemcpy(D_A, A, M * N * sizeof(DT), cudaMemcpyHostToDevice);

	softmax_base_2D<DT><<<grid, block>>>(D_A, D_B, D_C, M, N, M * N);
	cudaDeviceSynchronize();
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		//  Error check
		std::cerr << "CUDA kernel error: " << cudaGetErrorString(cudaError) << std::endl;
	}

	//  Correctness check
	DT *H_B = new DT[M * N];
	cudaMemcpy(H_B, D_B, N * M * sizeof(DT), cudaMemcpyDeviceToHost);

	bool flag = true;
	for (int i = 0; i < M; ++i) {
		if (flag)
			for (int j = 0; j < N; ++j) {
				if (H_B[i * N + j] != B[i * N + j]) {
					std::cout << "wrong" << std::endl;
					std::cout << H_B[i * N + j]  << " " << i << " " << j << std::endl;
					flag = false;
					break;
				}
			}
	}
	
	delete A;
	delete B;
	delete C;
	delete H_B;

}
