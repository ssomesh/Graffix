#ifndef _SSSP_H_
#define _SSSP_H_
#include <cuda_profiler_api.h>
#include "graph.h"

__global__ void initialize(uint64_t*, uint64_t);

__global__ void levelInit(int*, uint64_t);

__global__ void getLevel();

__global__ void getLevel(Graph, uint64_t, uint64_t, int*, bool*);

__global__ void neighborsDegreeSum(Graph, int*, uint64_t);

void pagerank(Graph&, float*, float*, uint64_t, unsigned, unsigned);


uint64_t getSrc(Graph&, int*);


#endif
