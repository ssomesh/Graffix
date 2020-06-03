#include <iostream>
#include "graph.h"
#include "utils.h"
#include "timer.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>

__global__ void dinit(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;	
		goaheadnodeofcomponent[id] = graph.nnodes;
		phores[id] = 0;
		partners[id] = id;
		processinnextiteration[id] = false;
	}
}
__global__ void dfindelemin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		// if I have a cross-component edge,
		// 	find my minimum wt cross-component edge,
		//	inform my boss about this edge e (atomicMin).
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = graph.nnodes;
		foru minwt = MYINFINITY;
		unsigned degree = graph.getOutDegree(src);
		for (unsigned ii = 0; ii < degree; ++ii) {
			foru wt = graph.getWeight(src, ii);
			if (wt < minwt) {
				unsigned dst = graph.getDestination(src, ii);
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {	// cross-component edge.
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		dprintf("\tminwt[%d] = %d\n", id, minwt);
		eleminwts[id] = minwt;
		partners[id] = dstboss;

		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			// inform boss.
			foru oldminwt = atomicMin(&minwtcomponent[srcboss], minwt);
		}
	}
}

__global__ void dfindelemin2(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (id < graph.nnodes) {
		unsigned src = id;
		unsigned srcboss = cs.find(src);

		if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != graph.nnodes)
		  {
		    unsigned degree = graph.getOutDegree(src);
		    for (unsigned ii = 0; ii < degree; ++ii) {
		      foru wt = graph.getWeight(src, ii);
		      if (wt == eleminwts[id]) {
			unsigned dst = graph.getDestination(src, ii);
			unsigned tempdstboss = cs.find(dst);
			if (tempdstboss == partners[id]) {	// cross-component edge.
			  //atomicMin(&goaheadnodeofcomponent[srcboss], id);
			  
			  if(atomicCAS(&goaheadnodeofcomponent[srcboss], graph.nnodes, id) == graph.nnodes)
			    {
			    }
			}
		      }
		    }
		  }
	}
}



__global__ void verify_min_elem(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(cs.isBoss(id))
	    {
	      if(goaheadnodeofcomponent[id] == graph.nnodes)
		{
		  return;
		}

	      unsigned minwt_node = goaheadnodeofcomponent[id];

	      unsigned degree = graph.getOutDegree(minwt_node);
	      foru minwt = minwtcomponent[id];

	      if(minwt == MYINFINITY)
		return;
		
	      bool minwt_found = false;
	      for (unsigned ii = 0; ii < degree; ++ii) {
		foru wt = graph.getWeight(minwt_node, ii);

		if (wt == minwt) {
		  minwt_found = true;
		  unsigned dst = graph.getDestination(minwt_node, ii);
		  unsigned tempdstboss = cs.find(dst);
		  if(tempdstboss == partners[minwt_node] && tempdstboss != id)
		    {
		      processinnextiteration[minwt_node] = true;
		      return;
		    }
		}
	      }

	      printf("component %d is wrong %d\n", id, minwt_found);
	    }
	}
}

__global__ void elim_dups(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(processinnextiteration[id])
	    {
	      unsigned srcc = cs.find(id);
	      unsigned dstc = partners[id];
	      
	      if(minwtcomponent[dstc] == eleminwts[id])
		{
		  if(id < goaheadnodeofcomponent[dstc])
		    {
		      processinnextiteration[id] = false;
		    }
		}
	    }
	}
}

__global__ void dfindcompmin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(partners[id] == graph.nnodes)
	    return;

	  unsigned srcboss = cs.find(id);
	  unsigned dstboss = cs.find(partners[id]);
	  if (id != partners[id] && srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id && goaheadnodeofcomponent[srcboss] == id) {	// my edge is min outgoing-component edge.
	    if(!processinnextiteration[id]);
	  }
	  else
	    {
	      if(processinnextiteration[id]);
	    }
	}
}

__global__ void dfindcompmintwo(unsigned *mstwt, Graph graph, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, GlobalBarrier gb, bool *repeat, unsigned *count) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id, nthreads = blockDim.x * gridDim.x;
	if (inpid < graph.nnodes) id = inpid;

	unsigned up = (graph.nnodes + nthreads - 1) / nthreads * nthreads;
	unsigned srcboss, dstboss;


	for(id = tid; id < up; id += nthreads) {
	  if(id < graph.nnodes && processinnextiteration[id])
	    {
	      srcboss = csw.find(id);
	      dstboss = csw.find(partners[id]);
	    }
	  
	  gb.Sync();
	  	  
	  if (id < graph.nnodes && processinnextiteration[id] && srcboss != dstboss) {
	    dprintf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);

	    if (csw.unify(srcboss, dstboss)) {
	      atomicAdd(mstwt, eleminwts[id]);
	      atomicAdd(count, 1);
	      dprintf("u %d -> %d (%d)\n", srcboss, dstboss, eleminwts[id]);
	      processinnextiteration[id] = false;
	      eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
	    }
	    else {
	      *repeat = true;
	    }

	    dprintf("\tcomp[%d] = %d.\n", srcboss, csw.find(srcboss));
	  }

	  gb.Sync(); 
	}
}

int main(int argc, char *argv[]) {
  unsigned *mstwt, hmstwt = 0;
  int iteration = 0;
  Graph hgraph, graph;

  unsigned *partners, *phores;
  foru *eleminwts, *minwtcomponent;
  bool *processinnextiteration;
  unsigned *goaheadnodeofcomponent;

  double starttime, endtime;
  GlobalBarrierLifetime gb;
  const size_t compmintwo_res = maximum_residency(dfindcompmintwo, 384, 0);
  gb.Setup(nSM * compmintwo_res);

  if (argc != 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }

  hgraph.read(argv[1]);
  hgraph.cudaCopy(graph);

  // preprocessing the graph to make it amenable for coalescing
  renumber_replicate(hgraph); // this will modify the graph
  if (cudaMalloc((void **)&mstwt, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating mstwt failed");
  CUDA_SAFE_CALL(cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice));	// mstwt = 0.

  if (cudaMalloc((void **)&eleminwts, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating eleminwts failed");
  if (cudaMalloc((void **)&minwtcomponent, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating minwtcomponent failed");
  if (cudaMalloc((void **)&partners, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating partners failed");
  if (cudaMalloc((void **)&phores, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating phores failed");
  if (cudaMalloc((void **)&processinnextiteration, graph.nnodes * sizeof(bool)) != cudaSuccess) CudaTest("allocating processinnextiteration failed");
  if (cudaMalloc((void **)&goaheadnodeofcomponent, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating goaheadnodeofcomponent failed");

  

  unsigned prevncomponents, currncomponents = graph.nnodes;

  bool repeat = false, *grepeat;
  CUDA_SAFE_CALL(cudaMalloc(&grepeat, sizeof(bool) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));

  unsigned edgecount = 0, *gedgecount;
  CUDA_SAFE_CALL(cudaMalloc(&gedgecount, sizeof(unsigned) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

  printf("finding mst.\n");
  starttime = rtclock();

  do {
    ++iteration;
    prevncomponents = currncomponents;
    dinit 		<<<num_blocks, block_size>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    CudaTest("dinit failed");
    dfindelemin 	<<<num_blocks, block_size>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    dfindelemin2 	<<<num_blocks, block_size>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    verify_min_elem 	<<<num_blocks, block_size>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    CudaTest("dfindelemin failed");
    if(debug) print_comp_mins(cs, graph, minwtcomponent, goaheadnodeofcomponent, partners, processinnextiteration);


    do {
      repeat = false;

      CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
      dfindcompmintwo <<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, gb, grepeat, gedgecount);
      CudaTest("dfindcompmintwo failed");
		  
      CUDA_SAFE_CALL(cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost));
    } while (repeat); 

    currncomponents = cs.numberOfComponentsHost();
    CUDA_SAFE_CALL(cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
      printf("\titeration %d, number of components = %d , mstwt = %u mstedges = %u\n", iteration, currncomponents, hmstwt, edgecount);

      edgecount = 0; // reinitializing for the next iteration
    CUDA_SAFE_CALL(cudaMemcpy(gedgecount,&edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

      
  } while (currncomponents != prevncomponents);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  endtime = rtclock();
	
  printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
  printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
  printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));


  return 0;
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
