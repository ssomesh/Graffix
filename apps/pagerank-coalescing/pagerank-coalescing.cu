#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "graph.h"
#include "pagerank-coalescing.h"
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
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned src = id; //work;
	if (src >= graph.nnodes) return;
	
	unsigned neighborsize = graph.getOutDegree(src);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
             unsigned dst = graph.getDestination(src, ii);
                if(dst < graph.nnodes)
	           atomicAdd(&rank[dst], contributeRankToNeigh[src]);
	}
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
		processnode <<<num_blocks, block_size>>> (rank, graph, contributeRankToNeigh);
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
   omp_set_num_threads(16);
  uint64_t blockSize, start;
 
 for(blockSize=1;blockSize<n; blockSize=blockSize+blockSize){
  #pragma omp parallel for  private(start) schedule(static)
  for(start=0; start < n; start += blockSize + blockSize){
 // std::cout << "Get num threads " << omp_get_num_threads() << std::endl;
    merge(a, b, c, c_aux, start, start+blockSize, start + 2*blockSize, n);
}
 }

}




/*renumber and replicate the nodes */
void renumber_replicate(Graph& G) { 
  /* Step-1: renumber the nodes */

  // store the nodes' degrees in an array and sort the array in descending order
  int * h_nodeDegree = (int*) malloc(G.h_nnodes*sizeof(int));
  int * h_nodeDegree_aux = (int*) malloc(G.h_nnodes*sizeof(int)); // this is for the merge sort
  int * d_nodeDegree;
  uint64_t * h_nodeId = (uint64_t*) malloc(G.h_nnodes*sizeof(uint64_t));
  uint64_t * h_nodeId_aux = (uint64_t*) malloc(G.h_nnodes*sizeof(uint64_t));
  uint64_t * d_nodeId;
  gpuErrchk(cudaMalloc(&d_nodeId,G.h_nnodes*sizeof(uint64_t))); 
  gpuErrchk(cudaMalloc(&d_nodeDegree,G.h_nnodes*sizeof(int))); 
  unsigned blockSize = 256;
  unsigned numBlocks = (G.h_nnodes+blockSize-1)/blockSize;

  CPUTimer cputimer;
  cputimer.Start();

  populateDegree<<<numBlocks, blockSize>>>(G, d_nodeDegree,d_nodeId,G.h_nnodes);
  gpuErrchk(cudaMemcpy(h_nodeDegree, d_nodeDegree, G.h_nnodes*sizeof(int), cudaMemcpyDeviceToHost));  
  gpuErrchk(cudaMemcpy(h_nodeId, d_nodeId, G.h_nnodes*sizeof(uint64_t), cudaMemcpyDeviceToHost));  

  gpuErrchk(cudaFree(d_nodeDegree));
  gpuErrchk(cudaFree(d_nodeId));
  

  // sort the nodes in descending order and maintain another array to store the corresponding node id  
   Merge_Sort_Par(h_nodeDegree, h_nodeDegree_aux, h_nodeId, h_nodeId_aux, G.h_nnodes);

  cputimer.Stop();

  free(h_nodeDegree_aux);
  free(h_nodeId_aux);

  std::cout << "Time elapsed = " << cputimer.Elapsed() << " second" << std::endl;

  
#if 1

  int *h_level = (int*) malloc(G.h_nnodes*sizeof(int));
  int * d_level;
  gpuErrchk(cudaMalloc(&d_level,G.h_nnodes*sizeof(int))); 

  cputimer.Start();

  levelInit<<<numBlocks,blockSize>>>(d_level,G.h_nnodes);

  gpuErrchk(cudaMemcpy(h_level, d_level, G.h_nnodes*sizeof(int), cudaMemcpyDeviceToHost));  // initializing h_level for the first iteration

  bool h_changed, *d_changed;
  gpuErrchk(cudaMalloc(&d_changed,sizeof(bool)));

  int zero = 0; // the distance zero from source


  uint64_t src;
  for(uint64_t j = 0; j < G.h_nnodes; ++j) {
    src = h_nodeId[j];
    if(h_nodeDegree[j] == 0) {
      std::cout << "Nodes with degree 0 start at: " << j << std::endl;
      std::cout << "Number of nodes with degree 0 : " << G.h_nnodes-1-j << std::endl;
      break;
    }
  if(h_level[src] == INF_NEW ) {

  gpuErrchk(cudaMemcpy(&d_level[src],&zero, sizeof(zero), cudaMemcpyHostToDevice));

  do {
    h_changed = false;
    gpuErrchk(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));

    getLevel<<<numBlocks,blockSize>>>(G, G.h_nnodes, G.h_nedges, d_level, d_changed); // making it true all the time, so getting stuck in an infinite loop
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while(h_changed);

  gpuErrchk(cudaMemcpy(h_level, d_level, G.h_nnodes*sizeof(int), cudaMemcpyDeviceToHost));  

  }

}
  


 // assigning level 0 to nodes that have not been reached so far.. i.e., these are unreachable
  for(uint64_t s=0; s<G.h_nnodes; ++s) { 
  //  h_level[s] = (h_level[s] != INF_NEW) * h_level[s]; // this is optimal
     if(h_level[s] == INF_NEW) {
       h_level[s] = 0;
     }
  }

  cputimer.Stop();

  std::cout << "Time elapsed in assigning levels = " << cputimer.Elapsed() << " second" << std::endl;



#endif

// counting the number of nodes of each type
std::unordered_map<int,uint64_t> countPerLevel; // map of level:# nodes at that level
for(uint64_t i=0; i<G.h_nnodes; ++i) {
  countPerLevel[h_level[i]]++; 
}

int chunkSize = 32; // specifying the chunk size

std::cout << "Level : #nodes ; #holes" << std::endl;
unsigned holeSum = 0;
for(auto it=countPerLevel.begin(); it != countPerLevel.end(); ++it) {
  int  temp = chunkSize - ( (it->second) % chunkSize );
  std::cout << it->first << " : " << it->second << " ; " << temp  << std::endl;
  holeSum += temp;
}

std::cout << "total holes : " << holeSum << std::endl;




uint64_t * h_newId = (uint64_t*) malloc(sizeof(uint64_t)*G.h_nnodes); // stores the new id of the node, i.e., newId[i] = j means that new id assigned to nodes 'i' is 'j'.

// Step-1 : Assign the new id's to the nodes at level 0.

int maxLevel = 0;
uint64_t seqNum = 0; // the new id assigned to the nodes
for(uint64_t s = 0; s < G.h_nnodes; ++s) {
  if(h_level[s] == 0) {
    h_newId[s] = seqNum++;
  }
  else {
      maxLevel = max(h_level[s], maxLevel); // finding the number of levels in the bfs forest
    }
}

// Step-2 : Assign the new id's to the nodes at each level in a level-synchronous manner
seqNum = seqNum + ( chunkSize - (seqNum % chunkSize) ); // bump-up seqNum to the next multple of chunkSize

    // writing output to a file (for correctness check)

  const char filename[] = "bfs_output.txt";
  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for(uint64_t i = 0; i < G.h_nnodes; i++) {
    fprintf(o, "%d: %d\n", i, h_level[i]);
  }

  fclose(o);

} // end of function



__global__ void getLevel(Graph G, uint64_t nnodes, uint64_t nedges, int* d_level, bool* d_changed) {
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t src = gid; // node under consideration

  if(src >= nnodes)
    return;  // exit the kernel

  unsigned outDegree = G.getDegree(src);
  for(unsigned i=0; i<outDegree; ++i) {
      uint64_t dst = G. getDest(src,i); // get the i-th neighbor of src
      if(dst >= nnodes){
        return;
      }
//      unsigned wt = 1; // the edge weight is 1

  int altdist = d_level[src] + 1; // each edge has weight = 1
  if(altdist < d_level[dst]) { // a possible site for thread divergence
    int olddist = atomicMin(&d_level[dst], altdist);
    if(altdist < olddist) (*d_changed) =  true; // dist is updated to a lower value (another possible site for thread divergence)
  } 
}
}
