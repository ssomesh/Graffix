
#include <iostream>
#include "pagerank-threadDivergence.h"
#include "graph.h"
#include "utils.h"
#include "timer.h"
#include <stdio.h>
#include <cstring>
#include <cstdlib>

void print_output(const char *filename, float *hrank, float *drank, Graph graph) {
 gpuErrchk(cudaMemcpy(hrank, drank, graph.h_nnodes * sizeof(float), cudaMemcpyDeviceToHost));

  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for(int i = 0; i < graph.h_nnodes; i++) {
    fprintf(o, "%d: %d\n", i, h_dist[i]);
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
     output_file = "sssp-output.txt";
     src =  atoi(argv[2]);
     break;

    case 4:
     input_file = std::string(argv[1]);
     output_file = std::string(argv[2]);
     src =  atoi(argv[3]);

     break;
      
    default: 
     std::cerr << "Usage: " << argv[0] << " " << "<input_file> <src_node>" << std::endl; 
     exit(1);
  }

  G.read(input_file);  
 // G.printGraph();
  

  uint64_t numbytes_rank = sizeof(float) * G.h_nnodes;
  unsigned num_blocks, block_size;
  block_size = 256;
  num_blocks = (G.h_nnodes + block_size - 1)/block_size; // finding ceil(nnodes/block_size)
  float *drank, *hrank; // making 'dist' unsigned because atomicMin() is not supported on uint64_t
  hrank = (float*)malloc(numbytes_dist);
  gpuErrchk(cudaMalloc(&d_dist, numbytes_dist));

  GPUTimer gputimer;
  gputimer.Start();
  pagerank(G, hrank, drank, src, num_blocks, block_size);
  gputimer.Stop();
  print_output(output_file.c_str(), h_dist, d_dist, G); 
  
  printf("The code ran in %f ms\n", gputimer.Elapsed()); 
  

  return 0;
}
