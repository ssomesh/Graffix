#ifndef _SSSP_H_
#define _SSSP_H_
#include <cuda_profiler_api.h>
#include "graph.h"

__global__ void initialize(uint64_t*, uint64_t);

void sssp_parallel(Graph& G, unsigned* h_dist, unsigned* d_dist, uint64_t _src, unsigned num_blocks, unsigned block_size);

void getClusteringCoeff(Graph& G, double* h_cc, double* d_cc, unsigned num_blocks, unsigned block_size); 

#endif
