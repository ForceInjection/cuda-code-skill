# 8. Interaction with Other Libraries and Tools


This section describes important requirements and recommendations that ensure correct use of cuBLAS with other libraries and utilities.


##  8.1. nvprune 

`nvprune` enables pruning relocatable host objects and static libraries to only contain device code for the specific target architectures. In case of cuBLAS, particular care must be taken if using `nvprune` with compute capabilities, whose minor revision number is different than 0. To reduce binary size, cuBLAS may only store major revision equivalents of CUDA binary files for kernels reused between different minor revision versions. Therefore, to ensure that a pruned library does not fail for arbitrary problems, the user must keep binaries for a selected architecture and all prior minor architectures in its major architecture.

For example, the following call prunes `libcublas_static.a` to contain only sm_75 (Turing) and sm_70 (Volta) cubins:
    
    
    nvprune --generate-code code=sm_70 --generate-code code=sm_75 libcublasLt_static.a -o libcublasLt_static_sm70_sm75.a
    

which should be used instead of:
    
    
    nvprune -arch=sm_75 libcublasLt_static.a -o libcublasLt_static_sm75.a
    
