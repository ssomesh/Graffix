#include "scc_kernels.h"

__global__ void selectPivots(const uint32_t *range, uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field, const int max_pivot_count){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;    

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if( pivot_field[ range[row] % max_pivot_count] == row ) {
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setBackwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}

__global__ void pollForPivots(const uint32_t *range, const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const int max_pivot_count, const uint32_t *Fr, const uint32_t *Br){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    
    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t index = range[row];

    uint32_t oldRow = pivot_field[index % max_pivot_count];
    uint32_t oldDegree = (Fr[oldRow+1] - Fr[oldRow]) * (Br[oldRow+1] - Br[oldRow]);
    uint32_t newDegree = (Fr[row+1] - Fr[row]) * (Br[row+1] - Br[row]);

    if(newDegree > oldDegree)
        pivot_field[ index % max_pivot_count ] = row;
}

__global__ void update(uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if ( isForwardVisited(myTag) && isBackwardVisited(myTag)){
        rangeSet(&tags[row]);
    }
    else{
        *terminate = false;
        uint32_t index = 3 * range[row] + (uint32_t)isForwardVisited(myTag) + ((uint32_t)isBackwardVisited(myTag) << 1);
        range[row] = index;
        tags[row] = 0;
    }
}

__global__ void trim1(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    uint32_t myRange = range[row];

	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool eliminate = true;
	for(uint32_t i = 0; i < cnt; i++){
	    uint32_t index = nbrs[i];

		if ( !isRangeSet(tags[index]) && range[index] == myRange){
			eliminate = false;
            break;
        }
	}

	if ( !eliminate ) {
		eliminate = true;
		cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];
			
		for(uint32_t i = 0; i < cnt; i++){
	        uint32_t index = nbrs[i];

			if ( !isRangeSet(tags[index]) && range[index] == myRange){
				eliminate = false;
                break;
            }
		}
	}

	if ( eliminate ) {
		rangeSet(&myTag);
        setTrim1(&myTag);
        tags[row] = myTag;
		*terminate = false;
	}
	return;
}


__global__ void trim2(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t myRange = range[row];
    uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];
    uint32_t inDegree = 0;
    uint32_t k = 0;  //other neighbour

    bool eliminate = false;
    for(uint32_t i = 0; i < cnt; i++){
        uint32_t index = nbrs[i];

        if (!isRangeSet(tags[index]) && range[index] == myRange){
            inDegree++;

            if(inDegree == 2)
                break;

            k = index;
        }
    }

    if(inDegree == 1){
        cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];

        for(uint32_t i = 0; i < cnt; i++){
            uint32_t index = nbrs[i];
            
            if(index == k){
                
                uint32_t kCnt = Br[k + 1] - Br[k];
                const uint32_t *kNbrs = &Bc[Br[k]];
                uint32_t kRange = range[k];
                inDegree = 0;

                for(uint32_t j = 0; j < kCnt; j++){
                    uint32_t tindex = kNbrs[j];

                    if(!isRangeSet(tags[tindex]) && range[tindex] == kRange){
                        inDegree++;
        
                        if(inDegree==2)
                            break;
                    }
                }

                if(inDegree == 1)
                    eliminate = true;

                break;
            }
        }
    }


    if(!eliminate){
        cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];
        inDegree=0;
        k = 0;
            
        for( uint32_t i = 0; i < cnt; i++ ){
            uint32_t index = nbrs[i];

            if ( !isRangeSet(tags[index]) && range[index] == myRange){
                inDegree++;

                if(inDegree == 2)
                    break;

                k = index;
            }
        }

        if(inDegree == 1){
            cnt = Br[row + 1] - Br[row];
            nbrs = &Bc[Br[row]];

            for(uint32_t i = 0; i < cnt; i++){
                uint32_t index = nbrs[i];

                if(index == k){

                    uint32_t kCnt = Fr[k + 1] - Fr[k];
                    const uint32_t *kNbrs = &Fc[Fr[k]];
                    uint32_t kRange = range[k];
                    inDegree = 0;

                    for(uint32_t j = 0; j < kCnt; j++){
                        uint32_t tindex = kNbrs[j];

                        if(!isRangeSet(tags[tindex]) && range[tindex] == kRange){
                            inDegree++;

                            if(inDegree==2)
                                break;
                        }
                    }

                    if(inDegree == 1)
                        eliminate = true;

                    break;
                }
            }
        }
    }

    if(eliminate){
        uint32_t temp = min(row, k);
        rangeSet(&tags[row]);
        rangeSet(&tags[k]);
        setTrim2(&tags[temp]); //Only one of the two will be set as pivot for 2-SCC
    }
    return;
}


__global__ void fwd(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isForwardPropagate(myTag) || !isForwardVisited(myTag))
        return;
	
    uint32_t myRange = range[row];
	uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
			continue;

		setForwardVisitedBit(&tags[index]);
		end = false;
	}
	setForwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}


__global__ void bwd(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isBackwardPropagate(myTag) || !isBackwardVisited(myTag))
        return;

    uint32_t myRange = range[row];
	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isBackwardVisited(nbrTag) || range[index] != myRange )
			continue;

		setBackwardVisitedBit(&tags[index]);
		end = false;
	}
	setBackwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}

__global__ void assignUniqueRange(uint32_t *range, const uint8_t *tags, const uint32_t num_rows){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    range[row] = row;
}


__global__ void propagateRange1(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;
    
    uint32_t myRange = range[row];
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];
    bool end = true;

    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint32_t nbrRange = range[index];

        if(!isRangeSet(tags[index]) && nbrRange < myRange){
            myRange = nbrRange;
            end = false;
        }
    }

    if(!end){
        range[row] = myRange;
        *terminate = false;
    }
}

__global__ void propagateRange2(uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;
    
    uint32_t myRange = range[row];
    uint32_t newRange;

    if(myRange != row && myRange != (newRange = range[myRange])){
        range[row] = newRange;
        *terminate = false;
    }
}

//Coloring
__global__ void colorPropagation(const uint32_t *Fc, const uint32_t *Fr, uint32_t *range, const uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t mx = max(row, range[row]);
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];
    bool end = true;

    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint32_t nbrRange = range[index];

        if(!isRangeSet(tags[index]) && mx < nbrRange){
            mx = nbrRange;
            end = false;
        }
    }
    
    if(!end){
        range[row] = mx;
        *terminate = false;
    }
}

//coloring
__global__ void selectPivotColoring(const uint32_t *range, uint8_t *tags, const uint32_t num_rows){
    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if(range[row] == row){
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}


//coloring
__global__ void fwdColoring(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || !isForwardVisited(myTag))
        return;

    uint32_t myRange = range[row];
    uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];

    bool end = true;
    for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

        if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
            continue;

        setForwardVisitedBit(&tags[index]);
        end = false;
    }
    rangeSet(&tags[row]);
    if (!end)
        *terminate = false;
}


//coloring
__global__ void updateColoring(uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    *terminate = false;
    tags[row] = 0;
}


__global__ void selectFirstPivot(uint8_t *tags, const uint32_t num_rows, const uint32_t *pivot_field){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
    uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    if( pivot_field[0] == row ) {
        myTag = 0;
        setForwardVisitedBit(&myTag);
        setBackwardVisitedBit(&myTag);
        setPivot(&myTag);
        tags[row] = myTag;
    }
}

__global__ void pollForFirstPivot(const uint8_t *tags, const uint32_t num_rows, uint32_t* pivot_field, const uint32_t *Fr, const uint32_t *Br){

    uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (row > num_rows || isRangeSet(tags[row]))
        return;

    uint32_t oldRow = pivot_field[0];
    uint32_t oldDegree = (Fr[oldRow+1] - Fr[oldRow]) * (Br[oldRow+1] - Br[oldRow]);
    uint32_t newDegree = (Fr[row+1] - Fr[row]) * (Br[row+1] - Br[row]);

    if(newDegree > oldDegree)
        pivot_field[0] = row;
}

