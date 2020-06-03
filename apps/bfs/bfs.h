#ifndef _SSSP_H_
#define _SSSP_H_
#include <cuda_profiler_api.h>
#include "graph.h"

__global__ void initialize(Graph, int*, uint64_t);

void bfs_parallel(Graph& G, int* h_level, int* d_level, uint64_t _src, unsigned num_blocks, unsigned block_size);

#endif
