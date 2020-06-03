#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <queue>
#include <cstring>

struct AdjListNode // a node in the adjacency list of a node
{
  int nodeId;
  AdjListNode * ptr;
};

// A structure to represent an adjacency list
struct AdjList
{
    AdjListNode *head;  // pointer to head node of list
};


//template <int nnodes, int nedges>
struct Graph
{
  int n, m; // nodes, edges
  AdjList * adj;  


  Graph(int nnodes, int nedges)
  {
    n = nnodes;
    m = nedges;
    adj = (AdjList*)(malloc(sizeof(AdjList)*n));
    for (int i = 0; i < n; ++i)
    {
      adj[i].head = NULL; // initializing the adj array with NULL
    }
  }

  // Adds an edge to an undirected graph
  void addEdge(int u, int v) // add edge from u to v
  {
    // Add an edge from src to dest.  A new node is added to the adjacency
    // list of src.  The node is added at the begining
    AdjListNode * temp = (AdjListNode *)malloc(sizeof(AdjListNode)); // temp is a pointer to the new node
    temp->ptr = adj[u].head; // adding a node to the start of a linkedlist
    temp->nodeId = v;
    adj[u].head = temp;
     
  }

};


void BFS(Graph &G, int s, bool* marked, int* edgeTo, int * level ) // passing graph by ref
{
  std::queue <int> bfsQueue;
  bfsQueue.push(s);
  marked[s] = true;
  level[s] = 1; // the source is given the level 1.
  while (!bfsQueue.empty())
  {
    int v = bfsQueue.front(); // accesses the element in the front of the queue
    //std::cout << v << std::endl;
    bfsQueue.pop(); // removes the element from the front of the queue
    int w; // stores a neighbor of v
    AdjListNode * current = G.adj[v].head;
    while(current != NULL) // going over the neighbors of s
    {
      w = current->nodeId;
      if(!marked[w])
      {
        bfsQueue.push(w);
        marked[w] = true;
        edgeTo[w] = v;
        level[w] = level[v] + 1;
      }
      else if (marked[w]) { // the node has been visited, but there is a smaller level# can be assigned to it
        edgeTo[w] = v; // updating the path so that the edge is from a node which gives the smallest level#
        level[w] = std::min(level[w], level[v]+1); // updating the level# for a node 
      }
     current = current->ptr;
    }

  }
}


#if 0
void DFS(Graph &G, int s, bool* marked, int* edgeTo ) // passing graph by ref
{
  //std::cout<< G.adj[1].head->nodeId << std::endl;
  marked[s] = true;
  int w; // stores a neighbor of s
  // traversing the edgelist of s
  AdjListNode * current = G.adj[s].head;
  while(current != NULL) // going over the neighbors of s
  {
    w = current->nodeId;
    if(!marked[w])
    {
      std::cout << w << std::endl;
      DFS(G, w, marked, edgeTo);
      edgeTo[w] = s;
    }
   current = current->ptr;
  }
}
#endif

int main(int argc, char** argv)
{
  // graph representation: assume adjacency list representation
  /*
   * source: neighbor1 neighbor2 neighbor3 ...
   * 
   * */


  if(argc != 2) 
  {
    printf("Usage: %s input_graph\n",argv[0]);
    exit(1);
  }
  std::ifstream in(argv[1]);
  int nnodes, nedges;
  in >> nnodes;
  in >> nedges;
  Graph G(nnodes, nedges);

  //std::cout << G.n << " " << G.m << std::endl;

  for (int i = 0; i<nedges; i++)
  {
   int x, y;
   in >> x >> y;
   G.addEdge(x,y);
  }

  //int s = 0; // source: nodeId
  bool * marked = (bool*)malloc(nnodes * sizeof(bool)); // marked[v] = true if v is connected to source
  int * edgeTo = (int*)malloc(nnodes *sizeof(int)); // edgeTo[v] = previous vertex on path from s to vv
  int * level = (int*)malloc(nnodes *sizeof(int)); //  level of each node
  
  // setting marked to false for all nodes initially
  memset(marked,0,nnodes*sizeof(bool));
//  DFS(G,s, marked, edgeTo); // should perform a DFS traversal of the graph starting at s
  
  for(int s=0; s<G.n; s++) {
    if(!marked[s])
      BFS(G,s, marked, edgeTo, level); // should perform a BFS traversal of the graph starting at s
  }


//  for(int i=0; i<G.n; i++)
//  {
//    std::cout << "marked:" << marked[i] << std::endl;
//  }

//  for(int i=0; i<G.n; i++)
//  {
//    std::cout << edgeTo[i] << std::endl;
//  }
    std::cout << "level info:" << std::endl;

    std::cout << "NodeId:level" << std::endl;
  for(int i=0; i<G.n; i++)
  {
    std::cout << i << ":"<< level[i] << std::endl;
  }
  return 0;
}
