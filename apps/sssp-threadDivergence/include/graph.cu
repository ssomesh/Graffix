#include "graph.h"
#include <stdio.h>
 Graph::Graph() {
   h_nnodes = 0;
   h_nedges = 0;
   h_edges = nullptr;
   h_offset = nullptr;
   h_weights = nullptr;
   d_edges = nullptr;
   d_offset = nullptr;
   d_weights = nullptr;

 }


 void Graph::read(std::string filename) {


  // FIXME: the read function is buggy

  std::ifstream input(filename.c_str());
  if(!input.is_open()) {
    std::cerr << "Could not open the file \"" << filename << "\""  << std::endl;
    exit(1);
  }

  // file is found

  input >> h_nnodes >> h_nedges;

/******************************************************************/
// Allocation starts
  unsigned numbytes_offset = sizeof(uint64_t) * (h_nnodes+1);
  unsigned numbytes_edges = sizeof(uint64_t) * h_nedges;
  unsigned numbytes_weights = sizeof(unsigned) * h_nedges;
  /***************************************************/
  // on host
  h_offset = (uint64_t*)malloc(numbytes_offset);
  //if(h_offset == NULL)
  //{
  //  printf("Memory allocation failed");
  //  return;
  //}
  h_edges = (uint64_t*)malloc(numbytes_edges);
  h_weights = (unsigned*)malloc(numbytes_weights);
  memset(h_offset, 0, numbytes_offset);
  memset(h_edges, 0, numbytes_edges);
  memset(h_weights, 0, numbytes_weights);
  /***************************************************/
#if 1
 // getCSR()
 // generating the CSR representation and populating the h_offset and h_edges array as the deliverable
 
 // Assumption:
 // 1. There is a edge list representation of the graph available, sorted by source vertex ids.
 // 2. The node ids always start from 0.
 
 // there are h_edges lines left in the file since
 uint64_t srcPrev, srcCurr; // storing the ids of the previous and the current vertices
 uint64_t offset = 0; // the offset in the h_edges array
 uint64_t index = 0; // the index of the h_offset array to which the value of offset has to be written
 input >> srcPrev >> h_edges[0] >> h_weights[0]; // reading the src and dest of the first edge
 h_offset[index] = offset;
 for (int i=1; i<h_nedges; i++) {
  input >> srcCurr >> h_edges[i] >> h_weights[i];
//  if(srcCurr == srcPrev) { // we are in the middle of the edge list of the same source vertex
//    ++offset;
//  }

  ++offset;
  if(srcPrev != srcCurr) { // srcCurr has a new source id
//    ++offset;
    uint64_t diff = srcCurr - srcPrev;
    while(diff-- /*&& (index <= h_nnodes)*/ ) { // to account for the values of offset for the vertices that do not have any neighbors
      ++index;
      h_offset[index] = offset;
    }
  }

  srcPrev = srcCurr; // making the current node as the previous node, for the next run
 }
 
 // putting the offset to 'h_nedges' for the last nodes that do not have any outgoing edges.
 for(int i=index+1; i<=h_nnodes; i++)
  h_offset[i] = h_nedges;

#endif 

  /***************************************************/

 // on device
 gpuErrchk(cudaMalloc(&d_offset, numbytes_offset));
 gpuErrchk(cudaMalloc(&d_edges, numbytes_edges));
 gpuErrchk(cudaMalloc(&d_weights, numbytes_weights));
  
 //gpuErrchk(cudaMalloc(&d_nnodes, numbytes_edges));
 //gpuErrchk(cudaMalloc(&d_nedges, numbytes_edges));

 /***************************************************/

 // copying to device
 
 gpuErrchk(cudaMemcpy(d_offset, h_offset, numbytes_offset, cudaMemcpyHostToDevice));
 gpuErrchk(cudaMemcpy(d_edges, h_edges, numbytes_edges, cudaMemcpyHostToDevice));
 gpuErrchk(cudaMemcpy(d_weights, h_weights, numbytes_weights, cudaMemcpyHostToDevice));

}


void Graph::printGraph() {
  std::cout << "offset array: " << std::endl;
  for(int i=0; i<h_nnodes+1; i++)
    std::cout << h_offset[i] << std::endl;
  std::cout << "edges array: " << std::endl;
  for(int i=0; i<h_nedges; i++)
    std::cout << h_edges[i] << std::endl;
}

__device__ unsigned Graph::getDegree(uint64_t node) {
    return (d_offset[node+1] - d_offset[node]);
  }

__device__ uint64_t Graph::getDest(uint64_t node, unsigned edgeId) {
  unsigned id = d_offset[node] + edgeId;
  return d_edges[id];
}

__device__ unsigned Graph::getWt(uint64_t node, unsigned edgeId) {
  unsigned id = d_offset[node] + edgeId;
  return d_weights[id]; 
}
