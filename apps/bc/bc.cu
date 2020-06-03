#include "bc.h"

void bc_exact (Graph& G, double* nodeBC) {
  
  // declare the auxillary data structures for the bc computation
  
  unsigned n = G.nnodes;
  unsigned * d_sigma;
  int hops_from_source, * d_level, * d_hops_from_source;
  double * d_delta, * d_nodeBC;
  bool takeNextIter, * d_takeNextIter; // to decide whether to go for the next iteration or not
  gpuErrchk(cudaMalloc(&d_sigma, sizeof(unsigned) * n));
  gpuErrchk(cudaMalloc(&d_level, sizeof(int) * n));
  gpuErrchk(cudaMalloc(&d_delta, sizeof(double) * n));
  gpuErrchk(cudaMalloc(&d_nodeBC, sizeof(double) * n));

  gpuErrchk(cudaMalloc(&d_hops_from_source, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_takeNextIter, sizeof(bool)));
  
  gpuErrchk(cudaMemset(d_nodeBC, 0, sizeof(double)*n)); // initializing node bc

  
  unsigned blockSize = 512;
  unsigned gridSize = (n + blockSize - 1) / blockSize; // ceil(n/blockSize)
  
  for(unsigned s=0; s<n; ++s) {  // outer loop of Brandes' Algorithm
     hops_from_source = 0; // keeps track of the number of hops from source in the current iteration. 
     
     initialize<<<gridSize, blockSize>>>(d_sigma, d_delta, d_level, d_hops_from_source, s, n);
#ifdef DEBUG
     cudaDeviceSynchronize();
     cudaError_t errCode = cudaPeekAtLastError();
     if (errCode != cudaSuccess) {
       fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
       errCode, cudaGetErrorString(errCode));
    }
#endif
    
    // forward pass 
    do {
      cudaMemset(d_takeNextIter,false,sizeof(bool));
      bc_forward_pass<<<gridSize, blockSize>>>(G, d_sigma, d_level, d_hops_from_source, n, d_takeNextIter);
      cudaDeviceSynchronize();
#ifdef DEBUG
     cudaError_t errCode = cudaPeekAtLastError();
     if (errCode != cudaSuccess) {
       fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
       errCode, cudaGetErrorString(errCode));
    }
#endif

      ++hops_from_source; // updating the level to process in the next iteration
//      gpuErrchk(cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice));
      incHop<<<1,1>>>(d_hops_from_source);

      gpuErrchk(cudaMemcpy(&takeNextIter, d_takeNextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    }while(takeNextIter);
    
    
  // backward pass
  
  --hops_from_source;
  gpuErrchk(cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice));
  while(hops_from_source > 1) {
    bc_backward_pass<<<gridSize,blockSize>>>(G, d_sigma, d_delta, d_nodeBC, d_level, d_hops_from_source, n);
#ifdef DEBUG
     cudaError_t errCode = cudaPeekAtLastError();
     if (errCode != cudaSuccess) {
       fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
       errCode, cudaGetErrorString(errCode));
    }
#endif
    --hops_from_source;
    gpuErrchk(cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice));
  }

  accumulate_bc<<<gridSize, blockSize>>>(d_delta, d_nodeBC, d_level, s, n);
#ifdef DEBUG
     errCode = cudaPeekAtLastError();
     if (errCode != cudaSuccess) {
       fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
       errCode, cudaGetErrorString(errCode));
    }
#endif
  cudaDeviceSynchronize();



  } // end of outerloop of brandes' algorithm
   


    gpuErrchk(cudaMemcpy(nodeBC, d_nodeBC, sizeof(double) * n, cudaMemcpyDeviceToHost));

}



__global__ void initialize(unsigned* d_sigma, double* d_delta, int* d_level, int* d_hops_from_source, unsigned s, unsigned n) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n) return;
    d_level[tid] = -1;
    d_delta[tid] = 0.0;
    d_sigma[tid] = 0;
    
    if(tid == s) { // for the source
      d_level[tid] = 0;
      d_sigma[tid] = 1;
      *d_hops_from_source = 0;
    }
  }

  __global__ void bc_forward_pass(Graph G, unsigned* d_sigma, int* d_level, int* d_hops_from_source, unsigned n, bool* d_takeNextIter) {
    unsigned u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= n) return;

    // only processing the nodes at level '*d_hops_from_source' -- a level synchronous processing, though not work efficient
    if(d_level[u] == *d_hops_from_source) {  
       unsigned end = G.d_offset[u+1];
       for(unsigned i = G.d_offset[u]; i < end; ++i) { // going over the neighbors of u
          unsigned v = G.d_edges[i];
          if(d_level[v] == -1) {  // v is seen for the first time
            d_level[v] = *d_hops_from_source + 1; // no atomics required since this is benign data race due to level synchronous implementation
            *d_takeNextIter = true;
          }
          if(d_level[v] == *d_hops_from_source + 1) { // 'v' is indeed the neighbor of u
            atomicAdd(&d_sigma[v], d_sigma[u]);
          }
       }
    }
  }

  __global__ void bc_backward_pass(Graph G, unsigned* d_sigma, double* d_delta, double* d_nodeBC, int* d_level, int* d_hops_from_source, unsigned n) {
      
    unsigned u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= n) return;

    if(d_level[u] == *d_hops_from_source - 1) { // backward traversal of DAG, one level at a time 
      
       unsigned end = G.d_offset[u+1];
       double sum = 0.0;
       for(unsigned i = G.d_offset[u]; i < end; ++i) { // going over the neighbors of u for which it is the predecessor in the DAG
          unsigned v = G.d_edges[i];
          if(d_level[v] == *d_hops_from_source) {
            sum += (1.0 * d_sigma[u]) / d_sigma[v] * (1.0 + d_delta[v]);
          }
       }

       d_delta[u] += sum;

    }

  }

  __global__ void accumulate_bc(double * d_delta, double* d_nodeBC, int* d_level, unsigned s, unsigned n) {
    
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n || tid == s || d_level[tid] == -1) return;

    d_nodeBC[tid] += d_delta[tid]/2.0;

  }

__global__ void incHop(int* d_hops_from_source) {
    *d_hops_from_source = *d_hops_from_source + 1;
  }
