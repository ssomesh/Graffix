#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "bfs.h"


__global__ void
initialize(Graph G, int* d_level, uint64_t nnodes){
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  if(gid < nnodes) {
    d_level[gid] = -1;
  }
}



void bfs_parallel(Graph& G, int* h_level, int* d_level, uint64_t _src, unsigned num_blocks, unsigned block_size) {
  cudaProfilerStart(); // start of profiling region
  initialize<<<num_blocks, block_size>>>(G, d_level, G.h_nnodes);
  cudaProfilerStop(); // end of profiling region
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError() );

  unsigned zero = 0; // the level of the source vertex 
  uint64_t src = _src; // setting the source vertex level by specifying the node-id.
  gpuErrchk(cudaMemcpy(&d_level[src],&zero, sizeof(zero), cudaMemcpyHostToDevice));


  gpuErrchk(cudaMemcpy(h_level, d_level, G.h_nnodes * sizeof(int), cudaMemcpyDeviceToHost));


}
