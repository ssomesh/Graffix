
# Note: any file containg the definition of a device function or global kernel must have the extension '.cu'
# -rdc=true is required if the device function and the kernel from which it is invoked are not all in the same file

# to enable the macro 'TIMER' from the command line, add the switch '-DTIMER' to
# the compilation line.

main: graph.cu mst.cu
	mkdir -p bin
	nvcc mst.cu graph.cu -std=c++11 -I../../utils -rdc=true -o bin/mst -G -g -O3 -Xcompiler -fopenmp -DDEBUG -DTIMER 


clean:
	rm -f bin/mst   # omitting the 'r' from the customary 'rm -rf' since sssp is a regular file.
