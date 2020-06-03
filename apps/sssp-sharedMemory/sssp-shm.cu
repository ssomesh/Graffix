#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "sssp-shm.h"
#include "timer.h"

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
  extern __shared__ unsigned shm[];
  unsigned * sh_offset = shm; 
  unsigned * sh_edges  = &shm[nodeCount];
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t src = gid; // node under consideration
  for(unsigned id = threadIdx.x; id < medges; id += blockIdx.x) {
      sh_offset[id] = d_offset[threadIdx.x];
      for(unsigned e = d_offset[id]; e < d_offset[id+1]; ++e)
			sh_edges[ii] = d_edges[e]; 
 } 
  
   for(unsigned k=0; k < threshold; ++k){
  	(processnode(G, nnodes, d_dist, src))
   }
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
    unsigned shm = sizeof(unsigned) * sz;
    ssspCompute <<<num_blocks, block_size,shm>>> (G, G.h_nnodes, G.h_nedges, d_dist, d_changed );
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while(h_changed);
 
  gpuErrchk(cudaMemcpy(h_dist, d_dist, G.h_nnodes * sizeof(unsigned), cudaMemcpyDeviceToHost));
}

void getClusteringCoeff(Graph& G) {
    /* Compute the in-clustering coefficient of a node..
     * 1. to find the in-CC among the incoming neighbors count the incoming or outgoing edges of the nighbors of the neighbors.
     * 2. to find the out-clustering coeff. (using the out-neighbors), count the outgoing edges of the neighbors.
     */
    std::cout << "call to getClusteringCoeff" << std::endl;
  	unsigned long long *d_count;
  	count=(unsigned long long *)malloc(sizeof(unsigned long long));
  	*count=0;
	  cudaMemcpy(d_count,count,sizeof(unsigned long long),cudaMemcpyHostToDevice);
	  int BPG=(e+191)/192,TPB=192;

	  CC<<<BPG,TPB>>>(G,d_count,vn,e);

    
  }

__global__ void CC(int* d_adj,int* d_pos,unsigned long long* d_count,int dvn,int de){
  unsigned int num=0;
  int eid =(blockDim.x)*(blockIdx.x)+(threadIdx.x);
  int u,v;

  if(eid<de){
    int middle;
    int begin=0,end=dvn-1;
    while(1){
      middle=(end+begin)/2;
      if(d_pos[middle]<eid){
        begin=middle+1;
      }
      else if(d_pos[middle]>=eid){
        if(end==begin+1){
          u=middle;
          v=d_adj[eid];
          break;
        }
        else if(end==begin){
          u=middle;
          v=d_adj[eid];
          break;
        }
        else{
          if(d_pos[middle-1]>=eid){
            end=middle-1;
          }
          else{
            u=middle;
            v=d_adj[eid];
            break;
          }
        }
      }
    }

    int us,ue,vs,ve;
    if(u==0){
      us=0;
      ue=d_pos[u];
    }
    else{
      us=d_pos[u-1]+1;
      ue=d_pos[u];
    }
    if(v==0){
      vs=0;
      ve=d_pos[v];
    }
    else{
      vs=d_pos[v-1]+1;
      ve=d_pos[v];
    }
    while(us<=ue&&vs<=ve){
      if(d_adj[us]==d_adj[vs]){
        num++;
        us++;
        vs++;
      }
      else if(d_adj[us]<d_adj[vs]){
        us++;
      }
      else{
        vs++;
      }
    }
    atomicAdd(d_count,num);
  }
}

