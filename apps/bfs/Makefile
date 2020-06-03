#main: sssp.cu 
#	nvcc sssp.cu -o sssp
#	mv -f sssp ../bin
#
#clean:
#	rm -rf ../bin/sssp

# Note: any file containg the definition of a device function or global kernel must have the extension '.cu'
# -rdc=true is required if the device function and the kernel from which it is invoked are not all in the same file

main: main.cpp ../../include/graph.cu bfs.cu
	nvcc main.cpp ../../include/graph.cu  bfs.cu -std=c++11 -I../../include/ -I../../utils -rdc=true -o bfs -G -g

clean:
	rm -f bfs   # omitting the 'r' from the customary 'rm -rf' since bfs is a regular file.
