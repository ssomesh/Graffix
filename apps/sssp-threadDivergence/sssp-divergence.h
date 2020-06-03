#ifndef _SSSP_H_
#define _SSSP_H_
#include <cuda_profiler_api.h>
#include "graph.h"

void sssp_parallel(Graph&, unsigned*, unsigned*, uint64_t, unsigned, unsigned);

void reduceThreadDivergence(Graph&);



#endif
