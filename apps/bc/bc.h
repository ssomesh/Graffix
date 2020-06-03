
#ifndef APPROX_H_
#define APPROX_H_

#include "graph.h"

  void bc_exact(Graph& G, double* nodeBC);

  __global__ void initialize(unsigned* d_sigma, double* d_delta, int* d_level, int* d_hops_from_source, unsigned s, unsigned n);
  __global__ void bc_forward_pass(Graph G, unsigned* d_sigma, int* d_level, int* d_hops_from_source, unsigned n, bool* d_takeNextIter);
  __global__ void bc_backward_pass(Graph G, unsigned* d_sigma, double* d_delta, double* d_nodeBC, int* d_level, int* d_hops_from_source, unsigned n);
  __global__ void accumulate_bc(double * d_delta, double* d_nodeBC, int* d_level, unsigned s, unsigned n);

__global__ void incHop(int* d_hops_from_source);
#endif
