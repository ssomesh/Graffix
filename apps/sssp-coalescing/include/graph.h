#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream>
#include <fstream> 
#include <cstring>
#include <cstdint>
#include "../utils/utils.h"

struct Graph {
 
 uint64_t h_nnodes, h_nedges; // the nodes and edges count in the graph (on the host)
 uint64_t *h_edges, *h_offset; // the edges and the offset array on the host
 uint64_t *d_edges, *d_offset; // the edges and the offset array on the device
 unsigned *h_weights;
 unsigned *d_weights; // edge-weights array 
 //uint64_t d_nnodes, d_nedges; // the nodes and edges count in the array (on the device) // not required. just pass the number of nodes to each kernel. The stack is copied from the CPU to the GPU. 
 
 // consider moving d_nnodes and d_nedges in the constant memory / texture memory since it is not changing
 Graph();
 void read (std::string);
 void printGraph();

 __device__  unsigned getDegree(uint64_t);
 __device__ uint64_t getDest(uint64_t, unsigned);
 __device__ unsigned getWt(uint64_t, unsigned);
 
};

#endif

/*Note: "uint64_t" is "long unsigned int"   */
