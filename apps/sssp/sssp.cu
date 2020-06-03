#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "sssp.h"
#include "timer.h"


//#define INF 1000000000
__constant__ unsigned INF = 1000000000;

__global__ void
initialize(Graph G, unsigned* d_dist, uint64_t nnodes){
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  if(gid < nnodes) {
    d_dist[gid] = INF;
    //printf("%ld\n",INF); // works fine
 // printf("%d",d_dist[gid]);
  }
  //printf("%d",G.d_offset[gid]);
}

__device__ bool
processedge(Graph& G, uint64_t nnodes, unsigned* d_dist, uint64_t worknode, unsigned i, uint64_t& dst) {
  dst = G. getDest(worknode,i); // get the i-th neighbor of worknode
  if(dst >= nnodes)
     return false;

  unsigned wt = G.getWt(worknode, i); // get edge-weight of the i-th edge
  if(wt >= INF) return false;

  unsigned altdist = d_dist[worknode] + wt;
  if(altdist < d_dist[dst]) { // a possible site for thread divergence
    unsigned olddist = atomicMin(&d_dist[dst], altdist);
    if(altdist < olddist) return true; // dist is updated to a lower value (another possible site for thread divergence)
  }
  return false;
}

__device__ bool
processnode(Graph& G, uint64_t nnodes, unsigned* d_dist, uint64_t worknode) {

  if(worknode >= nnodes)
    return false;
  bool changed = false; // thread-local
  unsigned outDegree = G.getDegree(worknode);
  for(unsigned i=0; i<outDegree; ++i) {
      uint64_t dst = nnodes;
      unsigned olddist = processedge(G, nnodes, d_dist, worknode, i, dst);
      if(olddist)
        changed = true;
    }
    return changed;
}

__global__ void
ssspCompute(Graph G, uint64_t nnodes, uint64_t nedges, unsigned* d_dist, bool* d_changed) {
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t src = gid; // node under consideration
  if(processnode(G, nnodes, d_dist, src))
    *d_changed = true;
}

void sssp_parallel(Graph& G, unsigned* h_dist, unsigned* d_dist, uint64_t _src, unsigned num_blocks, unsigned block_size) {
  cudaProfilerStart(); // start of profiling region
  initialize<<<num_blocks, block_size>>>(G, d_dist, G.h_nnodes);
  cudaProfilerStop(); // end of profiling region
  cudaDeviceSynchronize();
  gpuErrchk(cudaPeekAtLastError() );
  
  bool h_changed, *d_changed;
  gpuErrchk(cudaMalloc(&d_changed,sizeof(bool)));  
  
  unsigned zero = 0; // the distance zero from source
  uint64_t src = _src; // setting the source vertex by specifying the node-id.

  gpuErrchk(cudaMemcpy(&d_dist[src],&zero, sizeof(zero), cudaMemcpyHostToDevice));
  do {
    h_changed = false;
    gpuErrchk(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));

    ssspCompute <<<num_blocks, block_size>>> (G, G.h_nnodes, G.h_nedges, d_dist, d_changed );
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while(h_changed);
 
  gpuErrchk(cudaMemcpy(h_dist, d_dist, G.h_nnodes * sizeof(unsigned), cudaMemcpyDeviceToHost));
}
