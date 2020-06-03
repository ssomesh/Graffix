#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils/timer.h"
#include "graph.h"
#include "approx.h"

// Conditional Compilation for timing the code using #ifdef TIMER. 
// Set the macro using -DTIMER from the command line 

using namespace std;

void write_output_to_file(const char* filename, double* nodeBC, unsigned n) {
  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for(unsigned s = 0; s < n; s++) {
    fprintf(o, "%d %.2lf\n", s, nodeBC[s]);
  }

  fclose(o);
}

void write_topK_to_file(const char* filename, unsigned* topVertices, unsigned k, double* nodeBC) {
  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

// prints node rank and nodeId
  for(unsigned rank = 1; rank <= k; rank++) {
    fprintf(o, "%d %d %.2lf\n", rank, topVertices[rank-1], nodeBC[rank-1]);
  }

  fclose(o);
}


 void merge(double* a, double* a_aux, unsigned* b, unsigned* b_aux, unsigned lo, unsigned mid, unsigned hi, unsigned n) {
       if (mid >= n) return;
       if (hi > n) hi = n;
       unsigned i = lo, j = mid, ii = lo, jj = mid, k;
       for (k = lo; k < hi; k++) {
          if      (i == mid)       {  a_aux[k] = a[j++];  b_aux[k]  = b[jj++]; } 
          else if (j == hi)        {  a_aux[k] = a[i++];  b_aux[k]  = b[ii++]; }
          else if (a[j] - a[i] > 0.000001)    {  a_aux[k] = a[j++];  b_aux[k]  = b[jj++]; } // '>' means descending order
          else                     {  a_aux[k] = a[i++];  b_aux[k]  = b[ii++]; }
       }
       // copy back
       for (k = lo; k < hi; k++) {
          a[k] = a_aux[k];
          b[k] = b_aux[k];
       }
    }

void mergeSort(double* a, double* a_aux, unsigned* b, unsigned* b_aux, unsigned n) {
 // omp_set_num_threads(16);
  unsigned blockSize, start;

  for(blockSize=1;blockSize<n; blockSize=blockSize+blockSize){
  //  #pragma omp parallel for  private(start) schedule(static)
    for(start=0; start < n; start += blockSize + blockSize){
      // std::cout << "Get num threads " << omp_get_num_threads() << std::endl;
      merge(a, a_aux, b, b_aux, start, start+blockSize, start + 2*blockSize, n);
    }
  }

}


void vertex_rank_exact(Graph& G, double* nodeBC, unsigned* topVertices) {
  
  unsigned*  topVertices_aux = (unsigned*) malloc(sizeof(unsigned) * G.nnodes); // required for merge sort
  double*  nodeBC_aux = (double*) malloc(sizeof(double) * G.nnodes); // required for merge sort 

 // sort the nodes in descending order by BC values.
  mergeSort(nodeBC, nodeBC_aux, topVertices, topVertices_aux, G.nnodes); 


//  for(unsigned i = 0; i < k; ++i) {  // to check if the sorting is correct.
//    printf("%d %.2lf\n", topVertices[i], nodeBC[i]);
//   }
}




int main(int argc, char** argv) {

  if(argc < 3) {
    cerr << "Usage: " << argv[0] << " " << "<input graph> k"  << endl; 
    exit(1);
  }
  ifstream in (argv[1]);
  Graph G;
  string input_file = string(argv[1]);

#ifdef TIMER
  CPUTimer cputimer;
  cputimer.Start();
#endif

  G.readCSR(input_file);  
  //G.printGraph();
  reduceThreadDivergence(G);

  // allocating space for vertex BC, both on CPU and GPU
  double*  nodeBC = (double*) malloc(sizeof(double) * G.nnodes); // the node attribute in this case is node BC
  




#ifdef TIMER
  cputimer.Stop();
  cout << "time elapsed in reading the graph =  " << cputimer.Elapsed()  << " second" << endl;
#endif


#ifdef TIMER
  cputimer.Start();
#endif
  bc_exact(G, nodeBC);
#ifdef TIMER
  cputimer.Stop();
  cout << "time elapsed in bc computation =  " << cputimer.Elapsed()  << " second" << endl;
#endif


#if 1
/* writing the BC of each node to a file */

 // extracting only the file name from the path
 string output_file = "";
 bool flag = false; // to tell that a '.' is encountered
 for(const char* cp = argv[1]; *cp; ++cp) {
   if(flag && *cp == 'e') // a ".e" is encountered, which is potentially the beginning of the extension ".edges"
     break; // break from the for loop
   switch(*cp) {
     case '/' : 
       output_file = "";
       break;
     case '.' :
       flag = true;
       break;
     default: 
         output_file += *cp;
   }
 }


  string output_file1 = "bc_exact_" + output_file + ".txt" ; // creating the desired output filename
 
  write_output_to_file(output_file1.c_str(), nodeBC, G.nnodes);
 /* for ranking top-K BC vertices */

  unsigned* topVertices = (unsigned*) malloc(sizeof(unsigned) * G.nnodes); // holds the vertices sorted in descending order by bc values.
  for(unsigned i = 0; i < G.nnodes; ++i) { // initialization
    topVertices[i] = i;
  }

  int k = atoi(argv[2]); // the 'k' in topK.

#ifdef TIMER
  cputimer.Start();
#endif
  vertex_rank_exact(G, nodeBC, topVertices); // 'k' should also be made a parameter
#ifdef TIMER
  cputimer.Stop();
  cout << "time elapsed in top-k computation =  " << cputimer.Elapsed()  << " second" << endl;
#endif
  string output_file2 = "topK_approx_" + output_file + "_" + to_string(k) + ".txt" ; // creating the desired output filename
  write_topK_to_file(output_file2.c_str(), topVertices, k, nodeBC);

#endif

  return 0;
}

