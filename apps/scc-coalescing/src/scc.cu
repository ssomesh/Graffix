#include "scc.h"
#include<stack>
#include<unistd.h>
#include "load.h"
#include "scc_kernels.h"
using namespace std;


int main( int argc, char** argv ){

    if ( argc < 11 ) {
	print_help();
	return 1;
    }

    char *file = NULL;
    char c, algo;
    bool trim1 = true, trim2 = true;
    int warpSize = 1;

    while((c = getopt(argc, argv, "a:p:q:w:f:")) != -1){
        switch(c){
            case 'a':
                algo = optarg[0];
                break;    

            case 'p':
                trim1 = optarg[0]=='0'?false:true;
                break;

            case 'q':
                trim2 = optarg[0]=='0'?false:true;
                break;

            case 'w':
                warpSize = atoi(optarg);
                break;

            case 'f':
                file = optarg;
		break;

		default: 
			print_help();
			return 1;		
        }
    }

    // CSR representation 
    uint32_t CSize; // column arrays size
    uint32_t RSize; // range arrays size
    // Forwards arrays
    uint32_t *Fc = NULL; // forward columns
    uint32_t *Fr = NULL; // forward ranges
    // Backwards arrays
    uint32_t *Bc = NULL; // backward columns
    uint32_t *Br = NULL; // backward ranges

    //obtain a CSR graph representation
    loadFullGraph(file, &CSize, &RSize, &Fc, &Fr, &Bc, &Br );
    
  // preprocessing the graph to make it amenable for coalescing
    renumber_replicate(&CSize, &RSize, &Fc, &Fr, &Bc, &Br); // this will modify the graph


    try {

        switch(algo){
            case 'y':
                wSlota( CSize, RSize, Fc, Fr, Bc, Br, trim1, trim2, warpSize);
                break;

		default:
			return 1;
        }
    }
    catch (const char * e)
    {
        printf("%s\n",e);
        return 1;
    }
	printf("\n");
    delete [] Fr;
    delete [] Fc;
    delete [] Br;
    delete [] Bc;
    return 0;
}
