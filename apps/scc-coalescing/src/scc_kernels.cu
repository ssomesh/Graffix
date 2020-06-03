#include "scc_kernels.h"

__global__ void selectPivots(const uint32_t *range, uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field, const int max_pivot_count){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;    

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if( pivot_field[ range[row] % max_pivot_count] == row ) {
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setBackwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}

__global__ void pollForPivots(const uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const int max_pivot_count, const uint32_t *Fr, const uint32_t *Br){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    
    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t index = range[row];

    uint32_t oldRow = pivot_field[index % max_pivot_count];
    uint32_t oldDegree = (Fr[oldRow+1] - Fr[oldRow]) * (Br[oldRow+1] - Br[oldRow]);
    uint32_t newDegree = (Fr[row+1] - Fr[row]) * (Br[row+1] - Br[row]);

    if(newDegree > oldDegree)
        pivot_field[ index % max_pivot_count ] = row;
}

__global__ void update(uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if ( isForwardVisited(myTag) && isBackwardVisited(myTag)){
        rangeSet(&tags[row]);
    }
    else{
        *terminate = false;
        uint32_t index = 3 * range[row] + (uint32_t)isForwardVisited(myTag) + ((uint32_t)isBackwardVisited(myTag) << 1);
        range[row] = index;
        tags[row] = 0;
    }
}

__global__ void trim1(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    uint32_t myRange = range[row];

	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool eliminate = true;
	for(uint32_t i = 0; i < cnt; i++){
	    uint32_t index = nbrs[i];

		if ( !isRangeSet(tags[index]) && range[index] == myRange){
			eliminate = false;
            break;
        }
	}

	if ( !eliminate ) {
		eliminate = true;
		cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];
			
		for(uint32_t i = 0; i < cnt; i++){
	        uint32_t index = nbrs[i];

			if ( !isRangeSet(tags[index]) && range[index] == myRange){
				eliminate = false;
                break;
            }
		}
	}

	if ( eliminate ) {
		rangeSet(&myTag);
        setTrim1(&myTag);
        tags[row] = myTag;
		*terminate = false;
	}
	return;
}


__global__ void trim2(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t myRange = range[row];
    uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];
    uint32_t inDegree = 0;
    uint32_t k = 0;  //other neighbour

    bool eliminate = false;
    for(uint32_t i = 0; i < cnt; i++){
        uint32_t index = nbrs[i];

        if (!isRangeSet(tags[index]) && range[index] == myRange){
            inDegree++;

            if(inDegree == 2)
                break;

            k = index;
        }
    }

    if(inDegree == 1){
        cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];

        for(uint32_t i = 0; i < cnt; i++){
            uint32_t index = nbrs[i];
            
            if(index == k){
                
                uint32_t kCnt = Br[k + 1] - Br[k];
                const uint32_t *kNbrs = &Bc[Br[k]];
                uint32_t kRange = range[k];
                inDegree = 0;

                for(uint32_t j = 0; j < kCnt; j++){
                    uint32_t tindex = kNbrs[j];

                    if(!isRangeSet(tags[tindex]) && range[tindex] == kRange){
                        inDegree++;
        
                        if(inDegree==2)
                            break;
                    }
                }

                if(inDegree == 1)
                    eliminate = true;

                break;
            }
        }
    }


    if(!eliminate){
        cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];
        inDegree=0;
        k = 0;
            
        for( uint32_t i = 0; i < cnt; i++ ){
            uint32_t index = nbrs[i];

            if ( !isRangeSet(tags[index]) && range[index] == myRange){
                inDegree++;

                if(inDegree == 2)
                    break;

                k = index;
            }
        }

        if(inDegree == 1){
            cnt = Br[row + 1] - Br[row];
            nbrs = &Bc[Br[row]];

            for(uint32_t i = 0; i < cnt; i++){
                uint32_t index = nbrs[i];

                if(index == k){

                    uint32_t kCnt = Fr[k + 1] - Fr[k];
                    const uint32_t *kNbrs = &Fc[Fr[k]];
                    uint32_t kRange = range[k];
                    inDegree = 0;

                    for(uint32_t j = 0; j < kCnt; j++){
                        uint32_t tindex = kNbrs[j];

                        if(!isRangeSet(tags[tindex]) && range[tindex] == kRange){
                            inDegree++;

                            if(inDegree==2)
                                break;
                        }
                    }

                    if(inDegree == 1)
                        eliminate = true;

                    break;
                }
            }
        }
    }

    if(eliminate){
        uint32_t temp = min(row, k);
        rangeSet(&tags[row]);
        rangeSet(&tags[k]);
        setTrim2(&tags[temp]); //Only one of the two will be set as pivot for 2-SCC
    }
    return;
}


__global__ void fwd(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isForwardPropagate(myTag) || !isForwardVisited(myTag))
        return;
	
    uint32_t myRange = range[row];
	uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
			continue;

		setForwardVisitedBit(&tags[index]);
		end = false;
	}
	setForwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}


__global__ void bwd(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isBackwardPropagate(myTag) || !isBackwardVisited(myTag))
        return;

    uint32_t myRange = range[row];
	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isBackwardVisited(nbrTag) || range[index] != myRange )
			continue;

		setBackwardVisitedBit(&tags[index]);
		end = false;
	}
	setBackwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}

__global__ void assignUniqueRange(uint32_t *range, const uint8_t *tags, const uint32_t num_rows){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    range[row] = row;
}


__global__ void propagateRange1(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;
    
    uint32_t myRange = range[row];
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];
    bool end = true;

    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint32_t nbrRange = range[index];

        if(!isRangeSet(tags[index]) && nbrRange < myRange){
            myRange = nbrRange;
            end = false;
        }
    }

    if(!end){
        range[row] = myRange;
        *terminate = false;
    }
}

__global__ void propagateRange2(uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;
    
    uint32_t myRange = range[row];
    uint32_t newRange;

    if(myRange != row && myRange != (newRange = range[myRange])){
        range[row] = newRange;
        *terminate = false;
    }
}

//Coloring
__global__ void colorPropagation(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t mx = max(row, range[row]);
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];
    bool end = true;

    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint32_t nbrRange = range[index];

        if(!isRangeSet(tags[index]) && mx < nbrRange){
            mx = nbrRange;
            end = false;
        }
    }
    
    if(!end){
        range[row] = mx;
        *terminate = false;
    }
}

//coloring
__global__ void selectPivotColoring(const uint32_t *range, uint8_t *tags, const uint32_t num_rows){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if(range[row] == row){
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}


//coloring
__global__ void fwdColoring(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || !isForwardVisited(myTag))
        return;

    uint32_t myRange = range[row];
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];

    bool end = true;
    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

        if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
            continue;

        setForwardVisitedBit(&tags[index]);
        end = false;
    }
    rangeSet(&tags[row]);
    if (!end)
        *terminate = false;
}


//coloring
__global__ void updateColoring(uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    *terminate = false;
    tags[row] = 0;
}


__global__ void selectFirstPivot(uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if( pivot_field[0] == row ) {
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setBackwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}

__global__ void pollForFirstPivot(const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const uint32_t *Fr, const uint32_t *Br){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t oldRow = pivot_field[0];
    uint32_t oldDegree = (Fr[oldRow+1] - Fr[oldRow]) * (Br[oldRow+1] - Br[oldRow]);
    uint32_t newDegree = (Fr[row+1] - Fr[row]) * (Br[row+1] - Br[row]);

    if(newDegree > oldDegree)
        pivot_field[0] = row;
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
