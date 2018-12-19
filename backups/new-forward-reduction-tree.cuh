
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
  namespace op
  {

#define TILE_WIDTH 32
#define CUDA_MAX_NUM_THREADS 1024

    __global__ void reduction(float *input, float *output, int len, int C) {
      int bx = blockIdx.x;
      int tx = threadIdx.x;

      extern __shared__ float sInput[];
      sInput[tx] = (tx < C) && (bx*C + tx < len) ? input[bx*C + tx] : 0;
      sInput[tx + blockDim.x] = (tx + blockDim.x < C) && (bx*C + blockDim.x + tx < len) ? input[bx*C + blockDim.x + tx] : 0;
      __syncthreads();

      for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx + stride < C && tx < stride) {
          sInput[tx] += sInput[tx + stride];
        }
      }

      if (tx == 0) {
        output[bx] = sInput[0];
      }
    }

    __global__ void forward_kernel_for_reduction(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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
#define y5d(i4, i3, i2, i1, i0) y[(i4) * (M * H_out * W_out * C) + (i3) * (H_out * W_out * C) + (i2) * (W_out * C) + (i1) * (C) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

      int n, m, h, w, c, p, q;
      n = blockIdx.x;
      m = blockIdx.y;
      h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
      w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

      if (n < B && m < M && h < H_out && w < W_out) {
        for (c = 0; c < C; c++) {
          float acc = 0;
          for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
              acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
            }
          }
          y5d(n, m, h, w, c) = acc;
        }
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
      const int H_grid = ceil(H_out / (TILE_WIDTH * 1.0));
      const int W_grid = ceil(W_out / (TILE_WIDTH * 1.0));
      const int Z = H_grid * W_grid;
      dim3 gridDim(B, M, Z);
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

      float* y_for_reduction;
      if (B <= 1000) {
        cudaMalloc((void **) &y_for_reduction, B*M*H_out*W_out*C*sizeof(float) );
        forward_kernel_for_reduction<<<gridDim, blockDim>>>(y_for_reduction, x.dptr_, w.dptr_, B,M,C,H,W,K);
        reduction<<<dim3(B*M*H_out*W_out, 1, 1),
            dim3(8, 1, 1), C*sizeof(float)>>>(y_for_reduction, y.dptr_, B*M*H_out*W_out*C, C);
        cudaFree(y_for_reduction);
      } else {
        cudaMalloc((void **) &y_for_reduction, 1000*M*H_out*W_out*C*sizeof(float) );
        for (int i = 0; i < 10; i++) {
          forward_kernel_for_reduction<<<dim3(1000, M, Z), blockDim>>>(y_for_reduction, x.dptr_ + i*1000*C*H*W, w.dptr_, 1000,M,C,H,W,K);
          reduction<<<dim3(1000*M*H_out*W_out, 1, 1),
              dim3(8, 1, 1), C*sizeof(float)>>>(y_for_reduction, y.dptr_ + i*1000*M*H_out*W_out, 1000*M*H_out*W_out*C, C);
        }
        cudaFree(y_for_reduction);
      }

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
