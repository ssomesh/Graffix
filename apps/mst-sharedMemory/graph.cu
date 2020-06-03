#include "graph.h"

using namespace std;


 Graph::Graph() {
  nnodes = 0u;
  nedges = 0u;
  offset = nullptr;
  edges = nullptr;
  d_edges = nullptr;
  d_offset = nullptr;
 }



 void Graph::readCSR(std::string filename) {
  std::ifstream in(filename.c_str());
  if(!in.is_open()) {
    std::cerr << "Could not open the file \"" << filename << "\""  << std::endl;
    exit(1);
  }

  // file is found

 // begin reading the graph in CSR format. Make the graph a bidirectional, i.e. (undirected)
 
 // Assumption: The file containing the graph is sorted in ascending order by src node-id.

    in >> nnodes >> nedges;
    
    offset = (unsigned*)malloc(sizeof(unsigned) * (nnodes+1));  

    edges = (unsigned*)malloc(sizeof(unsigned) * (2 * nedges));   // note: only send the part of the edges array that is filled to the GPU, i.e. 0 .. edgeIndex-1.
    unsigned src, dst;
    map<unsigned, set<unsigned> > adjList; // set ensures no parallel edges

    for (unsigned e = 0; e < nedges; ++e) { 
      in >> src >> dst;
      adjList[src].insert(dst);
      adjList[dst].insert(src);
    }

    map<unsigned,set<unsigned> >::iterator it = adjList.begin();
    set<unsigned>::iterator itr;

    unsigned edge_index = 0u;
    src =  it->first;
    for(unsigned i=0; i <= src; ++i) {
      offset[src] = 0u; // to take care of the case when node 0, and the initial few nodes, have degree zero  
    }

    set<unsigned> neighbors = it->second;
    for(itr = neighbors.begin(); itr != neighbors.end(); ++itr) {
      edges[edge_index++] = *itr;
    }

    ++it; // to get to the second vertex in the graph

    for(; it != adjList.end(); ++it) {
      // populate the csr representation arrays 
      src = it->first;
      offset[src] = edge_index;      
      neighbors = it->second;
      for(itr = neighbors.begin(); itr != neighbors.end(); ++itr) {
        edges[edge_index++] = *itr;
      }
    }

    for (unsigned s = src+1; s <= nnodes; ++s) { // making the entries of the last nodes without neighbors compatible with the csr representation.
      offset[s] = edge_index;
    }

    // updating the number of edges in the graph, so that we may use it later
    // as a graph property in the remainder of the code.

    nedges = edge_index;

    /**************** GPU allocation *******************/

     gpuErrchk(cudaMalloc(&d_offset, sizeof(unsigned) * (nnodes+1)));
     gpuErrchk(cudaMalloc(&d_edges, sizeof(unsigned) * nedges));

   // copying to device
   
    gpuErrchk(cudaMemcpy(d_offset, offset, sizeof(unsigned) * (nnodes+1), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_edges, edges, sizeof(unsigned) * nedges, cudaMemcpyHostToDevice));
}




void Graph::printGraph() {
  //printing the csr represented array
  
  cout << "Edges array \n";

  for(unsigned i=0; i < nedges; ++i) {
    cout << edges[i] << " \n"[i+1 == nedges];
  }
  
  cout << "Offset array \n";
  for(unsigned i=0; i <= nnodes; ++i) {
    cout << offset[i] << " \n"[i == nnodes];
  }
}


__device__ unsigned Graph::getDegree(unsigned node) {
    return (d_offset[node+1] - d_offset[node]);
  }

__device__ unsigned Graph::getDest(unsigned node, unsigned edgeId) {
  unsigned id = d_offset[node] + edgeId;
  return d_edges[id];
}
