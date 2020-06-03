/*
 * TODO: profile the code on different types of input graphs to check for performance bottlenecks and assess the resource utilization
 * */

#include <iostream>
#include "bfs.h"
#include "graph.h"
#include "utils.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>

void print_output(const char *filename, int *h_level, int *d_level, Graph graph) {
 gpuErrchk(cudaMemcpy(h_level, d_level, graph.h_nnodes * sizeof(int), cudaMemcpyDeviceToHost));

  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for(int i = 0; i < graph.h_nnodes; i++) {
    fprintf(o, "%d: %d\n", i, h_level[i]);
  }

  fclose(o);
}

int main(int argc, char** argv) {
  Graph G;

  std::string input_file;
  std::string output_file;
  uint64_t src;
  switch(argc){
    case 3:
     input_file = std::string(argv[1]);
     src =  atoi(argv[2]);
     output_file = "bfs-output.txt";
     break;

    case 4:
     input_file = std::string(argv[1]);
     src =  atoi(argv[2]);
     output_file = std::string(argv[3]);

     break;
      
    default: 
     std::cerr << "Usage: " << argv[0] << " " << "<input_file> <src_node> [output_file]" << std::endl; 
     exit(1);
  }

  G.read(input_file);  
 // G.printGraph();
  
  unsigned numbytes_level = sizeof(int) * G.h_nnodes;
  unsigned num_blocks, block_size;
  block_size = 256;
  num_blocks = (G.h_nnodes + block_size - 1)/block_size; // finding ceil(nnodes/block_size)
  int *d_level, *h_level; // making 'dist' unsigned because atomicMin() is not supported on uint64_t
  h_level = (int*)malloc(numbytes_level);
  gpuErrchk(cudaMalloc(&d_level, numbytes_level));

  bfs_parallel(G, h_level, d_level, src, num_blocks, block_size);
  print_output(output_file.c_str(), h_level, d_level, G); 
  

  return 0;
}
