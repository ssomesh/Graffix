#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "sssp-divergence.h"
#include "timer.h"
#include <queue> 
#include <omp.h>
#include <unordered_map>


__constant__ unsigned INF = 1000000000;
const int INF_NEW = 1000000;

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

	uint64_t start = gid * (1024 / blockDim.x), end = (gid + 1) * (1024 / blockDim.x);
  // uint64_t src = gid; // node under consideration
	for (unsigned src = start; src < end; ++src) {
    if(processnode(G, nnodes, d_dist, src))
      *d_changed = true;
  }
}

void sssp_parallel(Graph& G, unsigned* h_dist, unsigned* d_dist, uint64_t _src, unsigned num_blocks, unsigned block_size) {
  reduceThreadDivergence(G);
  initialize<<<num_blocks, block_size>>>(G, d_dist, G.h_nnodes);
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

    ssspCompute <<<num_blocks, block_size>>> (G, G.h_nnodes, G.h_nedges, d_dist, d_changed);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while(h_changed);
    cudaDeviceSynchronize();
 
  gpuErrchk(cudaMemcpy(h_dist, d_dist, G.h_nnodes * sizeof(unsigned), cudaMemcpyDeviceToHost));
}

__global__ void levelInit(int * d_level, uint64_t nnodes) {
  uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= nnodes) return;
    d_level[gid] = INF_NEW;
}

__global__ void populateDegree(Graph G,int * d_nodeDegree, uint64_t* d_nodeId, uint64_t nnodes) {
  uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= nnodes) return;
  d_nodeDegree[gid] = G.getDegree(gid);
  d_nodeId[gid] = gid;
}

 void merge(int* a, int* b, uint64_t* c, uint64_t * c_aux, uint64_t lo, uint64_t mid, uint64_t hi, uint64_t n) {
       if (mid >= n) return;
       if (hi > n) hi = n;
       int i = lo, j = mid, ii = lo, jj = mid, k;
       for (k = lo; k < hi; k++) {
          if      (i == mid)       {  b[k] = a[j++];  c_aux[k]  = c[jj++]; } 
          else if (j == hi)        {  b[k] = a[i++];  c_aux[k]  = c[ii++]; }
          else if (a[j] > a[i])    {  b[k] = a[j++];  c_aux[k]  = c[jj++]; }// '>' means descending order
          else                     {  b[k] = a[i++];  c_aux[k]  = c[ii++]; }
       }
       // copy back
       for (k = lo; k < hi; k++) {
          a[k] = b[k];
          c[k] = c_aux[k];
       }
    }



void Merge_Sort_Par(int *a,int *b,uint64_t *c, uint64_t* c_aux, uint64_t n) //, int nThreads)
{
   omp_set_num_threads(24);
  uint64_t blockSize, start;
 
 for(blockSize=1;blockSize<n; blockSize=blockSize+blockSize){
  #pragma omp parallel for  private(start) schedule(static)
  for(start=0; start < n; start += blockSize + blockSize){
 // std::cout << "Get num threads " << omp_get_num_threads() << std::endl;
    merge(a, b, c, c_aux, start, start+blockSize, start + 2*blockSize, n);
}
 }

}




void reduceThreadDivergence(Graph& G) { 
  /* Step-1: change the node ids of the nodes so that threads are assigned to nodes in the order of */

  // store the nodes' degrees in an array and sort the array in descending order
  int * h_nodeDegree = (int*) malloc(G.h_nnodes*sizeof(int));
  int * h_nodeDegree_aux = (int*) malloc(G.h_nnodes*sizeof(int)); // this is for the merge sort
  int * d_nodeDegree;
  uint64_t * h_nodeId = (uint64_t*) malloc(G.h_nnodes*sizeof(uint64_t));
  uint64_t * h_nodeId_aux = (uint64_t*) malloc(G.h_nnodes*sizeof(uint64_t));
  uint64_t * d_nodeId;
  gpuErrchk(cudaMalloc(&d_nodeId,G.h_nnodes*sizeof(uint64_t))); 
  gpuErrchk(cudaMalloc(&d_nodeDegree,G.h_nnodes*sizeof(int))); 

  uint64_t * h_newId = (uint64_t*) malloc(sizeof(uint64_t)*G.h_nnodes); // stores the new id of the node, i.e., newId[i] = j means that new id assigned to nodes 'i' is 'j'.

  unsigned blockSize = 256;
  unsigned numBlocks = (G.h_nnodes+blockSize-1)/blockSize;

  CPUTimer cputimer;
  cputimer.Start();

  populateDegree<<<numBlocks, blockSize>>>(G, d_nodeDegree,d_nodeId,G.h_nnodes);
  gpuErrchk(cudaMemcpy(h_nodeDegree, d_nodeDegree, G.h_nnodes*sizeof(int), cudaMemcpyDeviceToHost));  
  gpuErrchk(cudaMemcpy(h_nodeId, d_nodeId, G.h_nnodes*sizeof(uint64_t), cudaMemcpyDeviceToHost));  

  gpuErrchk(cudaFree(d_nodeDegree));
  gpuErrchk(cudaFree(d_nodeId));
  
  // sort the node degree in descending order and maintain another array to store the corresponding node id  
   Merge_Sort_Par(h_nodeDegree, h_nodeDegree_aux, h_nodeId, h_nodeId_aux, G.h_nnodes);

  cputimer.Stop();

  free(h_nodeDegree_aux);
  free(h_nodeId_aux);

  for(uint64_t s = 0;  s < G.h_nnodes; ++s) {
      h_newId[s] = h_nodeId[s];
  }

  std::cout << "Time elapsed = " << cputimer.Elapsed() << " second" << std::endl;


  for(uint64_t s = 0;  s < G.h_nnodes; ++s) {
    std::cout << s << " : " << h_newId[s] << std::endl;
  }

  // transforming the graph according to the new numbering
  uint64_t *h_edges_aux, *h_offset_aux;  // the auxillary offset and edges array.

  // based on the new id's first modify the edges array. 
  // Thereafter, make the change to the offset array.


  h_offset_aux = (uint64_t*)malloc(sizeof(uint64_t) * (G.h_nnodes+1));

  h_offset_aux[0] = 0; 
  for(uint64_t s = 1;  s < G.h_nnodes; ++s) {
    h_offset_aux[h_newId[s]] = G.h_offset[s+1] - G.h_offset[s]; // this stores the degree of the nodes in the new locations. 
  }


 // going over the array h_offset_aux and updating the entries:

  for(uint64_t s = 0;  s < G.h_nnodes; ++s) {
    h_offset_aux[s+1] = G.h_offset[s+1] + G.h_offset[s]; // this stores the degree of the nodes in the new locations. 
  }
  h_offset_aux[G.h_nnodes] = G.h_offset[G.h_nnodes];




} // end of function
