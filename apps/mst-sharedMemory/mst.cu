
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
    unsigned shm = sizeof(unsigned) * sz;
    dinit 		<<<num_blocks, block_size>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    //printf("0 %d\n", cs.numberOfComponentsHost());
    CudaTest("dinit failed");
    dfindelemin 	<<<num_blocks, block_size, shm>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    dfindelemin2 	<<<num_blocks, block_size, shm>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    verify_min_elem 	<<<num_blocks, block_size, shm>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
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
