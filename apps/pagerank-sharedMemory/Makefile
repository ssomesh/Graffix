
# Note: any file containg the definition of a device function or global kernel must have the extension '.cu'
# -rdc=true is required if the device function and the kernel from which it is invoked are not all in the same file

main: main.cpp include/graph.cu pagerank-sharedMemory.cu
	mkdir -p build
	nvcc main.cpp include/graph.cu  pagerank-sharedMemory.cu -std=c++11 -Iinclude/ -I../../utils -rdc=true -o build/pagerank-sharedMemory -G -g -O3 -Xcompiler -fopenmp

clean:
	rm -f build/pagerank-sharedMemory   # omitting the 'r' from the customary 'rm -rf' since pagerank is a regular file.
