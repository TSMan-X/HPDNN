template<class T>
__global__ void reduce_max_2D(T * input, T * tmp, int M, int N) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  extern __shared__ T shared[];
  if (gx < N && gy < M) {
    shared[ty * blockDim.x + tx] = input[gy * N + gx];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) { // calculate the max value of a row, and will be stored at shared[y][0]
    if ( tx < s && gy < M ) {
      shared[ty * blockDim.x + tx] = max(shared[ty * blockDim.x + tx + s], shared[ty * blockDim.x + tx]); 
    }
    __syncthreads();
  }

  if (tx == 0) {
    atomicMax(tmp + gy, shared[0]);
  }
}

template<class T>
__global__ void broadcast_sub_elementwise_exp(T * input, T * output, T * tmp, int M, int N){
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  if ( gx < N && gy < M ) {
    output[gy * N + gx] = __expf(input[gy * N + gx] - tmp[gy]);
  }
}

template<class T>
__global__ void reduce_sum_2D(T * output, T * tmp, int M, int N){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  extern __shared__ T shared[];
  if (gx < N && gy < M) {
    shared[ty * blockDim.x + tx] = output[gy * N + gx];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) { // calculate the max value of a row, and will be stored at shared[y][0]
    if ( tx < s && gy < M ) {
      shared[ty * blockDim.x + tx] = shared[ty * blockDim.x + tx + s] + shared[ty * blockDim.x + tx]; 
    }
    __syncthreads();
  }

  if (tx == 0) {
    atomicAdd(tmp + gy, shared[0]);
  }
}

template<class T>
__global__ void broadcast_div(T * output, T * tmp, int M, int N){
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  if ( gx < N && gy < M ) {
    output[gy * N + gx] = output[gy * N + gx] / tmp[gy];
  }
}

// Only support softmax operation at dim 1.
template<class T>
void softmax_base_2D(T * input, T * output, T * tmp, int M, int N, const dim3 & grid, const dim3 & block) {
  // input: M * N
  // output: M * N
  // tmp : M

  // 1. reduceMax
  reduce_max_2D<T><<<grid, block, block.x * block.y * sizeof(T)>>>(input, tmp, M, N);
  cudaDeviceSynchronize();

  // 2. broadcastSub and elementwiseExp
  broadcast_sub_elementwise_exp<T><<<grid, block>>>(input, output, tmp, M, N);
  cudaDeviceSynchronize();

  // 3. reduceSum
  reduce_sum_2D<T><<<grid, block, block.x * block.y * sizeof(T)>>>(input, tmp, M, N);
  cudaDeviceSynchronize();

  // 4. broadcastDiv
  broadcast_div<T><<<grid, block>>>(output, tmp, M, N);
  cudaDeviceSynchronize();
  
  return;
}
