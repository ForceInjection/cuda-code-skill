# 7. cuBLAS Fortran Bindings


The cuBLAS library is implemented using the C-based CUDA toolchain. Thus, it provides a C-style API. This makes interfacing to applications written in C and C++ trivial, but the library can also be used by applications written in Fortran. In particular, the cuBLAS library uses 1-based indexing and Fortran-style column-major storage for multidimensional data to simplify interfacing to Fortran applications. Unfortunately, Fortran-to-C calling conventions are not standardized and differ by platform and toolchain. In particular, differences may exist in the following areas:


  * symbol names (capitalization, name decoration)

  * argument passing (by value or reference)

  * passing of string arguments (length information)

  * passing of pointer arguments (size of the pointer)

  * returning floating-point or compound data types (for example single-precision or complex data types)


To provide maximum flexibility in addressing those differences, the cuBLAS Fortran interface is provided in the form of wrapper functions and is part of the Toolkit delivery. The C source code of those wrapper functions is located in the `src` directory and provided in two different forms:


  * the thunking wrapper interface located in the file `fortran_thunking.c`

  * the direct wrapper interface located in the file `fortran.c`


The code of one of those two files needs to be compiled into an application for it to call the cuBLAS API functions. Providing source code allows users to make any changes necessary for a particular platform and toolchain.


The code in those two C files has been used to demonstrate interoperability with the compilers g77 3.2.3 and g95 0.91 on 32-bit Linux, g77 3.4.5 and g95 0.91 on 64-bit Linux, Intel Fortran 9.0 and Intel Fortran 10.0 on 32-bit and 64-bit Microsoft Windows XP, and g77 3.4.0 and g95 0.92 on Mac OS X.


Note that for g77, use of the compiler flag `-fno-second-underscore` is required to use these wrappers as provided. Also, the use of the default calling conventions with regard to argument and return value passing is expected. Using the flag -fno-f2c changes the default calling convention with respect to these two items.


The thunking wrappers allow interfacing to existing Fortran applications without any changes to the application. During each call, the wrappers allocate GPU memory, copy source data from CPU memory space to GPU memory space, call cuBLAS, and finally copy back the results to CPU memory space and deallocate the GPU memory. As this process causes very significant call overhead, these wrappers are intended for light testing, not for production code. To use the thunking wrappers, the application needs to be compiled with the file `fortran_thunking.c`.


The direct wrappers, intended for production code, substitute device pointers for vector and matrix arguments in all BLAS functions. To use these interfaces, existing applications need to be modified slightly to allocate and deallocate data structures in GPU memory space (using `cuBLAS_ALLOC` and `cuBLAS_FREE`) and to copy data between GPU and CPU memory spaces (using `cuBLAS_SET_VECTOR`, `cuBLAS_GET_VECTOR`, `cuBLAS_SET_MATRIX`, and `cuBLAS_GET_MATRIX`). The sample wrappers provided in `fortran.c` map device pointers to the OS-dependent type `size_t`, which is 32-bit wide on 32-bit platforms and 64-bit wide on a 64-bit platforms.


One approach to deal with index arithmetic on device pointers in Fortran code is to use C-style macros, and use the C preprocessor to expand these, as shown in the example below. On Linux and Mac OS X, one way of pre-processing is to use the option `-E -x f77-cpp-input` when using g77 compiler, or simply the option `-cpp` when using g95 or gfortran. On Windows platforms with Microsoft Visual C/C++, using ’cl -EP’ achieves similar results.


    ! Example B.1. Fortran 77 Application Executing on the Host
    ! ----------------------------------------------------------
        subroutine modify ( m, ldm, n, p, q, alpha, beta )
        implicit none
        integer ldm, n, p, q
        real*4 m (ldm, *) , alpha , beta
        external cublas_sscal
        call cublas_sscal (n-p+1, alpha , m(p,q), ldm)
        call cublas_sscal (ldm-p+1, beta, m(p,q), 1)
        return
        end
    
        program matrixmod
        implicit none
        integer M,N
        parameter (M=6, N=5)
        real*4 a(M,N)
        integer i, j
        external cublas_init
        external cublas_shutdown
    
        do j = 1, N
            do i = 1, M
                a(i, j) = (i-1)*M + j
            enddo
        enddo
        call cublas_init
        call modify ( a, M, N, 2, 3, 16.0, 12.0 )
        call cublas_shutdown
        do j = 1 , N
            do i = 1 , M
                write(*,"(F7.0$)") a(i,j)
            enddo
            write (*,*) ""
        enddo
        stop
        end
    


When traditional fixed-form Fortran 77 code is ported to use the cuBLAS library, line length often increases when the BLAS calls are exchanged for cuBLAS calls. Longer function names and possible macro expansion are contributing factors. Inadvertently exceeding the maximum line length can lead to run-time errors that are difficult to find, so care should be taken not to exceed the 72-column limit if fixed form is retained.


The examples in this chapter show a small application implemented in Fortran 77 on the host and the same application with the non-thunking wrappers after it has been ported to use the cuBLAS library.


The second example should be compiled with ARCH_64 defined as 1 on 64-bit OS system and as 0 on 32-bit OS system. For example for g95 or gfortran, this can be done directly on the command line by using the option `-cpp -DARCH_64=1`.


    ! Example B.2. Same Application Using Non-thunking cuBLAS Calls
    !-------------------------------------------------------------
    #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
        subroutine modify ( devPtrM, ldm, n, p, q, alpha, beta )
        implicit none
        integer sizeof_real
        parameter (sizeof_real=4)
        integer ldm, n, p, q
    #if ARCH_64
        integer*8 devPtrM
    #else
        integer*4 devPtrM
    #endif
        real*4 alpha, beta
        call cublas_sscal ( n-p+1, alpha,
        1                   devPtrM+IDX2F(p, q, ldm)*sizeof_real,
        2                   ldm)
        call cublas_sscal(ldm-p+1, beta,
        1                 devPtrM+IDX2F(p, q, ldm)*sizeof_real,
        2                 1)
        return
        end
        program matrixmod
        implicit none
        integer M,N,sizeof_real
    #if ARCH_64
        integer*8 devPtrA
    #else
        integer*4 devPtrA
    #endif
        parameter(M=6,N=5,sizeof_real=4)
        real*4 a(M,N)
        integer i,j,stat
        external cublas_init, cublas_set_matrix, cublas_get_matrix
        external cublas_shutdown, cublas_alloc
        integer cublas_alloc, cublas_set_matrix, cublas_get_matrix
        do j=1,N
            do i=1,M
                a(i,j)=(i-1)*M+j
            enddo
        enddo
        call cublas_init
        stat= cublas_alloc(M*N, sizeof_real, devPtrA)
        if (stat.NE.0) then
            write(*,*) "device memory allocation failed"
            call cublas_shutdown
            stop
        endif
        stat = cublas_set_matrix(M,N,sizeof_real,a,M,devPtrA,M)
        if (stat.NE.0) then
            call cublas_free( devPtrA )
            write(*,*) "data download failed"
            call cublas_shutdown
            stop
        endif
    


—


 _— Code block continues below. Space added for formatting purposes. —_


—


    call modify(devPtrA, M, N, 2, 3, 16.0, 12.0)
    stat = cublas_get_matrix(M, N, sizeof_real, devPtrA, M, a, M )
    if (stat.NE.0) then
    call cublas_free ( devPtrA )
    write(*,*) "data upload failed"
    call cublas_shutdown
    stop
    endif
    call cublas_free ( devPtrA )
    call cublas_shutdown
    do j = 1 , N
        do i = 1 , M
            write (*,"(F7.0$)") a(i,j)
        enddo
        write (*,*) ""
    enddo
    stop
    end
    
