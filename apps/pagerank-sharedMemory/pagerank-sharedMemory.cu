#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "pagerank-sharedMemory.h"
#include "timer.h"
#include <queue> 
#include <omp.h>
#include <unordered_map>


__global__ void initialize(float *rank, Graph graph, float *contributeRankToNeigh, float init_const) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < graph.nnodes) {
		rank[ii] = 0.0f;
                if(graph.getOutDegree(ii) != 0 )
                	contributeRankToNeigh[ii] = (float)(init_const/(float)graph.getOutDegree(ii));
	        else
                	contributeRankToNeigh[ii] = (float)(init_const/(float)graph.nnodes);
	}
}

__global__ void reinitializeContributions(float *rank, Graph graph, float *contributeRankToNeigh, float adjustfactor, float d) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < graph.nnodes) {
	        float newrank = rank[ii];
        	newrank *= d;
	        newrank += adjustfactor;
                rank[ii] = 0.0f;
                if(graph.getOutDegree(ii) != 0 )
                	contributeRankToNeigh[ii] = (float)(newrank/(float)graph.getOutDegree(ii));
	        else
                	contributeRankToNeigh[ii] = (float)(newrank/(float)graph.nnodes);
        }
}

__global__ void addResidueToRanks(float *rank, Graph graph, float *contributeRankToNeigh, float adjustfactor, float d) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < graph.nnodes) {
	        float newrank = rank[ii];
        	newrank *= d;
	        newrank += adjustfactor;
                rank[ii] = newrank;
        }
}

__global__ void processnode(float *rank, Graph graph, float *contributeRankToNeigh){//, unsigned work) {
      extern __shared__ unsigned shm[];
      unsigned * sh_offset = shm; 
      unsigned * sh_edges  = &shm[nodeCount];
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  for(unsigned id = threadIdx.x; id < medges; id += blockIdx.x) {
      sh_offset[iid] = d_offset[threadIdx.x];
      for(unsigned e = d_offset[iid]; e < d_offset[iid+1]; ++e)
			sh_edges[ii] = d_edges[e]; 
 } 
  
  for(unsigned k=0; k < threshold; ++k){
	unsigned src = id; //work;
	if (src >= graph.nnodes) return;
	
	unsigned neighborsize = graph.getOutDegree(src);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
             unsigned dst = graph.getDestination(src, ii);
                if(dst < graph.nnodes)
	           atomicAdd(&rank[dst], contributeRankToNeigh[src]);
    }
	}
  __syncthreads();
}


void pagerank(float *hrank, float *rank, float *contributeRankToNeigh, Graph &graph, unsigned num_blocks, unsigned block_size)
{
	float d = 0.85f; // damping factor
	float adjustfactor; // adjustfactor = (1-d)/N is constant
	int iteration = 0;
	adjustfactor = (1.0f-d);
        adjustfactor = (adjustfactor/(float)graph.nnodes);
	do {
		++iteration;
        unsigned shm = sizeof(unsigned) * sz;
		processnode <<<num_blocks, block_size,shm>>> (rank, graph, contributeRankToNeigh);
		cudaDeviceSynchronize();
		reinitializeContributions <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (rank, graph, contributeRankToNeigh, adjustfactor, d);
		cudaDeviceSynchronize();
	} while (iteration < 9); // running for 9 iterations
	
	//running for 10th iteration
		processnode <<<num_blocks, block_size>>> (rank, graph, contributeRankToNeigh);
		cudaDeviceSynchronize();
		reinitializeContributions <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (rank, graph, contributeRankToNeigh, adjustfactor, d);
		cudaDeviceSynchronize();

  gpuErrchk(cudaMemcpy(hrank, drank, G.h_nnodes * sizeof(float), cudaMemcpyDeviceToHost));
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

