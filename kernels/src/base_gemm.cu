template<class T_A, class T_B, class T_C>
__global__ void gemm_base_2D(T_A *A, T_B *B, T_C *C, int M, int N, int K) {
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;
	if (tx < M && ty < N) {
		T_C c = 0;
		for (int idx_k = 0; idx_k < K; ++idx_k) {
			c += A[ty * K + idx_k] * B[N * idx_k + tx];
		}
		C[ty * N + tx] = c;
	}
}
