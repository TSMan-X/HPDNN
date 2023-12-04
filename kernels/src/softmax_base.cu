// Only support softmax operation at dim 1.
template<class T>
__global__ void softmax_base_2D(T * input, T * output, T * tmp, int M, int N, int size) {
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  // 1. reduceMax
  if (gx < N && gx < M) {
    tmp[gy * N + gx] = input[gy * N + gx];
  }
  cudaDeviceSynchronize();

  for (int s = N / 2; s > 0; s >>= 1) { // calculate the max value of a row, and will be stored at tmp[y][0]
    if ( gx < s && gy < M ) {
      tmp[gy * N + gx] = max(tmp[gy * N + gx], tmp[gy * N + gx]); 
    }
    cudaDeviceSynchronize();
  }

  // 2. broadcastSub and elementwiseExp
  if ( gx < N && gy < M ) {
    output[gy * N + gx] = __expf(input[gy * N + gx] - tmp[gy * N]);
  }
  cudaDeviceSynchronize();

  // 3. reduceSum
  if (gx < N && gx < M) {
    tmp[gy * N + gx] = output[gy * N + gx];
  }
  cudaDeviceSynchronize();

  for (int s = N / 2; s > 0; s >>= 1) { // calculate the sum value of a row, and will be stored at tmp[y][0]
    if ( gx < s && gy < M ) {
      tmp[gy * N + gx] = tmp[gy * N + gx] + tmp[gy * N + gx]; 
    }
    cudaDeviceSynchronize();
  }

  // 4. broadcastDiv
  if ( gx < N && gy < M ) {
    output[gy * N + gx] = output[gy * N + gx] / tmp[gy * N];
  }

}
