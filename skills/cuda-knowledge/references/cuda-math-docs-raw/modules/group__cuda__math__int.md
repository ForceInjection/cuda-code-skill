# 12. Integer Mathematical Functions

**Source:** [group__CUDA__MATH__INT.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html)

---

#  12\. Integer Mathematical Functions[](#integer-mathematical-functions "Permalink to this headline")

This section describes integer mathematical functions. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ long int [abs](#group__cuda__math__int_1ga1151eb3014afb625325efbf497a0f53c)(long int a)
    

Calculate the absolute value of the input `long` `int` argument.

__device__ int [abs](#group__cuda__math__int_1ga76abb22f186c5d612673111bc922763c)(int a)
    

Calculate the absolute value of the input `int` argument.

__device__ long long int [abs](#group__cuda__math__int_1ga9b72c49887ae8759de62bae6c1fd7d74)(long long int a)
    

Calculate the absolute value of the input `long` `long` `int` argument.

__device__ long int [labs](#group__cuda__math__int_1ga7c12a7dadd4d909fb67bf09a5561dd41)(long int a)
    

Calculate the absolute value of the input `long` `int` argument.

__device__ long long int [llabs](#group__cuda__math__int_1gad81be6a75fda2c13bdb8a059e0ca83bb)(long long int a)
    

Calculate the absolute value of the input `long` `long` `int` argument.

__device__ long long int [llmax](#group__cuda__math__int_1ga99ba97b47d3fecf5788195dab122c9a0)(const long long int a, const long long int b)
    

Calculate the maximum value of the input `long` `long` `int` arguments.

__device__ long long int [llmin](#group__cuda__math__int_1ga07560f8e4fc530633e9ff767461ab234)(const long long int a, const long long int b)
    

Calculate the minimum value of the input `long` `long` `int` arguments.

__device__ unsigned long int [max](#group__cuda__math__int_1ga01b614ebc329901458498e8cf16c492f)(const long int a, const unsigned long int b)
    

Calculate the maximum value of the input `long` `int` and `unsigned` `long` `int` arguments.

__device__ unsigned long long int [max](#group__cuda__math__int_1ga155e4676c16909797772bc8985b83354)(const unsigned long long int a, const unsigned long long int b)
    

Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned int [max](#group__cuda__math__int_1ga476ad18352849fd22f7657154e31c6eb)(const unsigned int a, const int b)
    

Calculate the maximum value of the input `unsigned` `int` and `int` arguments.

__device__ unsigned long long int [max](#group__cuda__math__int_1ga6b0debb4cc697b72f6b0f9352fafc28e)(const long long int a, const unsigned long long int b)
    

Calculate the maximum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

__device__ unsigned long int [max](#group__cuda__math__int_1ga74404bdf3d59f4e927de1dc072466f63)(const unsigned long int a, const unsigned long int b)
    

Calculate the maximum value of the input `unsigned` `long` `int` arguments.

__device__ long long int [max](#group__cuda__math__int_1ga866dbfe2604ba86cabcf7a5fd4746615)(const long long int a, const long long int b)
    

Calculate the maximum value of the input `long` `long` `int` arguments.

__device__ unsigned long long int [max](#group__cuda__math__int_1ga92c725f6f30417c57f6a2d6fa276e3f8)(const unsigned long long int a, const long long int b)
    

Calculate the maximum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

__device__ unsigned long int [max](#group__cuda__math__int_1gabcd75cfe90bc913aafcd27dc36ed1bba)(const unsigned long int a, const long int b)
    

Calculate the maximum value of the input `unsigned` `long` `int` and `long` `int` arguments.

__device__ long int [max](#group__cuda__math__int_1gac4b0740e6d92a79e111e53b5692fc2be)(const long int a, const long int b)
    

Calculate the maximum value of the input `long` `int` arguments.

__device__ int [max](#group__cuda__math__int_1gacd95edd79e83ba55edb31cce43f4de42)(const int a, const int b)
    

Calculate the maximum value of the input `int` arguments.

__device__ unsigned int [max](#group__cuda__math__int_1gadadbde8421bbe39bc410723b475b4f01)(const unsigned int a, const unsigned int b)
    

Calculate the maximum value of the input `unsigned` `int` arguments.

__device__ unsigned int [max](#group__cuda__math__int_1gaf0541e5366e86e7017ee30080dcb8384)(const int a, const unsigned int b)
    

Calculate the maximum value of the input `int` and `unsigned` `int` arguments.

__device__ unsigned long int [min](#group__cuda__math__int_1ga1a23219e1efa70361c66b957edb24ee7)(const long int a, const unsigned long int b)
    

Calculate the minimum value of the input `long` `int` and `unsigned` `long` `int` arguments.

__device__ unsigned long long int [min](#group__cuda__math__int_1ga1aab8c188e41186bdf213c17182381bf)(const unsigned long long int a, const unsigned long long int b)
    

Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned long long int [min](#group__cuda__math__int_1ga397e8a6a22225c6fe429ed6a1c2c7371)(const unsigned long long int a, const long long int b)
    

Calculate the minimum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

__device__ int [min](#group__cuda__math__int_1ga58e735f4a25da078e0b2b84c58fe0beb)(const int a, const int b)
    

Calculate the minimum value of the input `int` arguments.

__device__ unsigned int [min](#group__cuda__math__int_1ga7c01d07e95c8c5d92d44ececba5dc286)(const unsigned int a, const int b)
    

Calculate the minimum value of the input `unsigned` `int` and `int` arguments.

__device__ unsigned long long int [min](#group__cuda__math__int_1ga7f7076014ad218b8fdfc390fb5108db6)(const long long int a, const unsigned long long int b)
    

Calculate the minimum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

__device__ long long int [min](#group__cuda__math__int_1ga802401c69360435f4db0ad7b473746c0)(const long long int a, const long long int b)
    

Calculate the minimum value of the input `long` `long` `int` arguments.

__device__ unsigned int [min](#group__cuda__math__int_1gab80b17dced2786d4cd9cd1d8884979e9)(const int a, const unsigned int b)
    

Calculate the minimum value of the input `int` and `unsigned` `int` arguments.

__device__ long int [min](#group__cuda__math__int_1gaca909621ba314c58e146af2e5aebd5a7)(const long int a, const long int b)
    

Calculate the minimum value of the input `long` `int` arguments.

__device__ unsigned int [min](#group__cuda__math__int_1gaf977b0326ecf1e84c73ba6469b1c195d)(const unsigned int a, const unsigned int b)
    

Calculate the minimum value of the input `unsigned` `int` arguments.

__device__ unsigned long int [min](#group__cuda__math__int_1gafaea95a7ffc0f0c460ee81844f5dc63b)(const unsigned long int a, const long int b)
    

Calculate the minimum value of the input `unsigned` `long` `int` and `long` `int` arguments.

__device__ unsigned long int [min](#group__cuda__math__int_1gafb3b206ef2d1d5e8cfc2f4a4483c9eb7)(const unsigned long int a, const unsigned long int b)
    

Calculate the minimum value of the input `unsigned` `long` `int` arguments.

__device__ unsigned long long int [ullmax](#group__cuda__math__int_1gace3212701af84c61bb59dfc171ed52c4)(const unsigned long long int a, const unsigned long long int b)
    

Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned long long int [ullmin](#group__cuda__math__int_1gad47917925a05d1598854fc5897f37eba)(const unsigned long long int a, const unsigned long long int b)
    

Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned int [umax](#group__cuda__math__int_1gaf3504ee1f7dbdc07170e20ae82238722)(const unsigned int a, const unsigned int b)
    

Calculate the maximum value of the input `unsigned` `int` arguments.

__device__ unsigned int [umin](#group__cuda__math__int_1ga49a8735b305c8892e57e8e86070e0b2b)(const unsigned int a, const unsigned int b)
    

Calculate the minimum value of the input `unsigned` `int` arguments.

##  12.1. Functions[](#functions "Permalink to this headline")

__device__ long int abs(long int a)[](#_CPPv43absl "Permalink to this definition")  

    

Calculate the absolute value of the input `long` `int` argument. 

Calculate the absolute value of the input argument `a`.

Returns
    

Returns the absolute value of the input argument.

  * abs(`LONG_MIN`) is `Undefined`


__device__ int abs(int a)[](#_CPPv43absi "Permalink to this definition")  

    

Calculate the absolute value of the input `int` argument. 

Calculate the absolute value of the input argument `a`.

Returns
    

Returns the absolute value of the input argument.

  * abs(`INT_MIN`) is `Undefined`


__device__ long long int abs(long long int a)[](#_CPPv43absx "Permalink to this definition")  

    

Calculate the absolute value of the input `long` `long` `int` argument. 

Calculate the absolute value of the input argument `a`.

Returns
    

Returns the absolute value of the input argument.

  * abs(`LLONG_MIN`) is `Undefined`


__device__ long int labs(long int a)[](#_CPPv44labsl "Permalink to this definition")  

    

Calculate the absolute value of the input `long` `int` argument. 

Calculate the absolute value of the input argument `a`.

Returns
    

Returns the absolute value of the input argument.

  * labs(`LONG_MIN`) is `Undefined`


__device__ long long int llabs(long long int a)[](#_CPPv45llabsx "Permalink to this definition")  

    

Calculate the absolute value of the input `long` `long` `int` argument. 

Calculate the absolute value of the input argument `a`.

Returns
    

Returns the absolute value of the input argument.

  * llabs(`LLONG_MIN`) is `Undefined`


__device__ long long int llmax(const long long int a, const long long int b)[](#_CPPv45llmaxKxKx "Permalink to this definition")  

    

Calculate the maximum value of the input `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ long long int llmin(const long long int a, const long long int b)[](#_CPPv45llminKxKx "Permalink to this definition")  

    

Calculate the minimum value of the input `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned long int max(const long int a, const unsigned long int b)[](#_CPPv43maxKlKm "Permalink to this definition")  

    

Calculate the maximum value of the input `long` `int` and `unsigned` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long long int max(const unsigned long long int a, const unsigned long long int b)[](#_CPPv43maxKyKy "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned int max(const unsigned int a, const int b)[](#_CPPv43maxKjKi "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `int` and `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long long int max(const long long int a, const unsigned long long int b)[](#_CPPv43maxKxKy "Permalink to this definition")  

    

Calculate the maximum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long int max(const unsigned long int a, const unsigned long int b)[](#_CPPv43maxKmKm "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ long long int max(const long long int a, const long long int b)[](#_CPPv43maxKxKx "Permalink to this definition")  

    

Calculate the maximum value of the input `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned long long int max(const unsigned long long int a, const long long int b)[](#_CPPv43maxKyKx "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long int max(const unsigned long int a, const long int b)[](#_CPPv43maxKmKl "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `long` `int` and `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ long int max(const long int a, const long int b)[](#_CPPv43maxKlKl "Permalink to this definition")  

    

Calculate the maximum value of the input `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ int max(const int a, const int b)[](#_CPPv43maxKiKi "Permalink to this definition")  

    

Calculate the maximum value of the input `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned int max(const unsigned int a, const unsigned int b)[](#_CPPv43maxKjKj "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned int max(const int a, const unsigned int b)[](#_CPPv43maxKiKj "Permalink to this definition")  

    

Calculate the maximum value of the input `int` and `unsigned` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long int min(const long int a, const unsigned long int b)[](#_CPPv43minKlKm "Permalink to this definition")  

    

Calculate the minimum value of the input `long` `int` and `unsigned` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long long int min(const unsigned long long int a, const unsigned long long int b)[](#_CPPv43minKyKy "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned long long int min(const unsigned long long int a, const long long int b)[](#_CPPv43minKyKx "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ int min(const int a, const int b)[](#_CPPv43minKiKi "Permalink to this definition")  

    

Calculate the minimum value of the input `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned int min(const unsigned int a, const int b)[](#_CPPv43minKjKi "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `int` and `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long long int min(const long long int a, const unsigned long long int b)[](#_CPPv43minKxKy "Permalink to this definition")  

    

Calculate the minimum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ long long int min(const long long int a, const long long int b)[](#_CPPv43minKxKx "Permalink to this definition")  

    

Calculate the minimum value of the input `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned int min(const int a, const unsigned int b)[](#_CPPv43minKiKj "Permalink to this definition")  

    

Calculate the minimum value of the input `int` and `unsigned` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ long int min(const long int a, const long int b)[](#_CPPv43minKlKl "Permalink to this definition")  

    

Calculate the minimum value of the input `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned int min(const unsigned int a, const unsigned int b)[](#_CPPv43minKjKj "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned long int min(const unsigned long int a, const long int b)[](#_CPPv43minKmKl "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `long` `int` and `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first. 

__device__ unsigned long int min(const unsigned long int a, const unsigned long int b)[](#_CPPv43minKmKm "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned long long int ullmax(const unsigned long long int a, const unsigned long long int b)[](#_CPPv46ullmaxKyKy "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned long long int ullmin(const unsigned long long int a, const unsigned long long int b)[](#_CPPv46ullminKyKy "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`. 

__device__ unsigned int umax(const unsigned int a, const unsigned int b)[](#_CPPv44umaxKjKj "Permalink to this definition")  

    

Calculate the maximum value of the input `unsigned` `int` arguments. 

Calculate the maximum value of the arguments `a` and `b`. 

__device__ unsigned int umin(const unsigned int a, const unsigned int b)[](#_CPPv44uminKjKj "Permalink to this definition")  

    

Calculate the minimum value of the input `unsigned` `int` arguments. 

Calculate the minimum value of the arguments `a` and `b`.