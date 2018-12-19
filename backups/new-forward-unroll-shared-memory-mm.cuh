
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
  namespace op
  {

#define TILE_WIDTH 32
#define CUDA_MAX_NUM_THREADS 1024

    __global__ void unroll_kernel(int b, int C, int H, int W, int K, float* x, float* X_unroll){
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
      int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
      int H_out = H - K + 1;
      int W_out = W - K + 1;
      int W_unroll = H_out * W_out;

      if(t < C * W_unroll) {
        int c = t / W_unroll;
        int s = t % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;
        int h_unroll = h_out * W_out + w_out;
        int w_base = c*K*K;
        for(int p = 0; p < K; p++) {
          for(int q = 0; q < K; q++) {
            int w_unroll = w_base + p*K + q;
            X_unroll[w_unroll * W_unroll + h_unroll] = x4d(b, c, h_out+p, w_out+q);
          }
        }
      }

    }


    __global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
      int bx = blockIdx.x; int by = blockIdx.y;
      int tx = threadIdx.x; int ty = threadIdx.y;

      int Row = by*TILE_WIDTH + ty;
      int Col = bx*TILE_WIDTH + tx;

      __shared__ float M[TILE_WIDTH][TILE_WIDTH];
      __shared__ float N[TILE_WIDTH][TILE_WIDTH];

      float Cval = 0;

      for(int i = 0; i < ceil(numAColumns/(float)TILE_WIDTH); ++i) {
        if((Row < numARows) && (i*TILE_WIDTH+tx < numAColumns))
          M[ty][tx] = A[Row*numAColumns + i*TILE_WIDTH + tx];
        else
          M[ty][tx] = 0;

        if((Col < numBColumns) && (i*TILE_WIDTH+ty < numBRows))
          N[ty][tx] = B[(i*TILE_WIDTH + ty)*numBColumns+Col];
        else
          N[ty][tx]=0;

        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; ++j) {
          Cval += M[ty][j] * N[j][tx];
        }
        __syncthreads();
      }

      if(Row < numCRows && Col < numCColumns)
        C[Row*numCColumns + Col] = Cval;
    }


    __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
    {

      /*
         Modify this function to implement the forward pass described in Chapter 16.
         We have added an additional dimension to the tensors to support an entire mini-batch
         The goal here is to be correct AND fast.
         We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
       */

      const int H_out = H - K + 1;
      const int W_out = W - K + 1;
      (void)H_out; // silence declared but never referenced warning. remove this line when you start working
      (void)W_out; // silence declared but never referenced warning. remove this line when you start working
      const int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));

      // An example use of these macros:
      // float a = y4d(0,0,0,0)
      // y4d(0,0,0,0) = a
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

      int n, m, h, w, c, p, q;
      n = blockIdx.x;
      m = blockIdx.y;
      h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
      w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

      if (n < B && m < M && h < H_out && w < W_out) {

        float acc = 0;
        for (c = 0; c < C; c++) {
          for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
              acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
            }
          }
        }
        y4d(n, m, h, w) = acc;
      }


#undef y4d
#undef x4d
#undef k4d
    }

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
 */
    template <>
    void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
    {

      // Use mxnet's CHECK_EQ to do assertions.
      // Remove this assertion when you do your implementation!
      // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

      // Extract the tensor dimensions into B,M,C,H,W,K
      const int B = x.shape_[0];
      const int M = y.shape_[1];
      const int C = x.shape_[1];
      const int H = x.shape_[2];
      const int W = x.shape_[3];
      const int K = w.shape_[3];
      // const int mnist_dim = 28;

      printf("B = %d\n", B);
      printf("M = %d\n", M);
      printf("C = %d\n", C);
      printf("H = %d\n", H);
      printf("W = %d\n", W);
      printf("K = %d\n", K);

      // Set the kernel dimensions
      const int H_out = H - K + 1;
      const int W_out = W - K + 1;
      int W_unroll = H_out * W_out;
      int H_unroll = C * K * K;
      const int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));
      const int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
      const int Z = H_grid * W_grid;
      dim3 gridDim(B, M, Z);
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

      float* X_unrolled;
      cudaMalloc((void **) &X_unrolled, H_unroll * W_unroll * sizeof(float));

      for(int n = 0; n < B; n++) {
        unroll_kernel<<<ceil((float)(C*H_out*W_out/CUDA_MAX_NUM_THREADS)),
												CUDA_MAX_NUM_THREADS>>>(
													n, C, H, W, K, x.dptr_, X_unrolled);
        cudaDeviceSynchronize();
        matrixMultiplyShared<<<dim3(ceil((float)W_unroll/TILE_WIDTH), ceil((float)M/TILE_WIDTH), 1),
																dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(
																	w.dptr_, X_unrolled, &y.dptr_[n*M*H_out*W_out], M,
																	H_unroll, H_unroll, W_unroll, M, W_unroll);
        cudaDeviceSynchronize();
      }


      // Call the kernel
      //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
      // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
      MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
 */
    template <typename gpu, typename DType>
    void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
    {
      CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
    }
  }
}

#endif
