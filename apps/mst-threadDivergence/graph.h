
#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <map>
#include <set>
#include "utils/utils.h"


struct Graph {
  unsigned nedges, nnodes; // on the cpu
  unsigned* edges;   // on the cpu
  unsigned* offset;   // on the cpu


  unsigned *d_edges, *d_offset; // the edges and the offset array on the device

  Graph();
  void readCSR(std::string);
  void printGraph();

  __device__ unsigned getDegree(unsigned node);

  __device__ unsigned getDest(unsigned node, unsigned edgeId);
};


#endif
