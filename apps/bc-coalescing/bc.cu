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
