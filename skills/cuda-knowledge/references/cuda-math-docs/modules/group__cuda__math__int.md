# 12. Integer Mathematical Functions

**Source:** group__CUDA__MATH__INT.html


#  12\. Integer Mathematical Functions

This section describes integer mathematical functions.

To use these functions, you do not need to include any additional header file in your program.

Functions

__device__ long int abs(long int a)


Calculate the absolute value of the input `long` `int` argument.

__device__ int abs(int a)


Calculate the absolute value of the input `int` argument.

__device__ long long int abs(long long int a)


Calculate the absolute value of the input `long` `long` `int` argument.

__device__ long int labs(long int a)


Calculate the absolute value of the input `long` `int` argument.

__device__ long long int llabs(long long int a)


Calculate the absolute value of the input `long` `long` `int` argument.

__device__ long long int llmax(const long long int a, const long long int b)


Calculate the maximum value of the input `long` `long` `int` arguments.

__device__ long long int llmin(const long long int a, const long long int b)


Calculate the minimum value of the input `long` `long` `int` arguments.

__device__ unsigned long int max(const long int a, const unsigned long int b)


Calculate the maximum value of the input `long` `int` and `unsigned` `long` `int` arguments.

__device__ unsigned long long int max(const unsigned long long int a, const unsigned long long int b)


Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned int max(const unsigned int a, const int b)


Calculate the maximum value of the input `unsigned` `int` and `int` arguments.

__device__ unsigned long long int max(const long long int a, const unsigned long long int b)


Calculate the maximum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

__device__ unsigned long int max(const unsigned long int a, const unsigned long int b)


Calculate the maximum value of the input `unsigned` `long` `int` arguments.

__device__ long long int max(const long long int a, const long long int b)


Calculate the maximum value of the input `long` `long` `int` arguments.

__device__ unsigned long long int max(const unsigned long long int a, const long long int b)


Calculate the maximum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

__device__ unsigned long int max(const unsigned long int a, const long int b)


Calculate the maximum value of the input `unsigned` `long` `int` and `long` `int` arguments.

__device__ long int max(const long int a, const long int b)


Calculate the maximum value of the input `long` `int` arguments.

__device__ int max(const int a, const int b)


Calculate the maximum value of the input `int` arguments.

__device__ unsigned int max(const unsigned int a, const unsigned int b)


Calculate the maximum value of the input `unsigned` `int` arguments.

__device__ unsigned int max(const int a, const unsigned int b)


Calculate the maximum value of the input `int` and `unsigned` `int` arguments.

__device__ unsigned long int min(const long int a, const unsigned long int b)


Calculate the minimum value of the input `long` `int` and `unsigned` `long` `int` arguments.

__device__ unsigned long long int min(const unsigned long long int a, const unsigned long long int b)


Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned long long int min(const unsigned long long int a, const long long int b)


Calculate the minimum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

__device__ int min(const int a, const int b)


Calculate the minimum value of the input `int` arguments.

__device__ unsigned int min(const unsigned int a, const int b)


Calculate the minimum value of the input `unsigned` `int` and `int` arguments.

__device__ unsigned long long int min(const long long int a, const unsigned long long int b)


Calculate the minimum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

__device__ long long int min(const long long int a, const long long int b)


Calculate the minimum value of the input `long` `long` `int` arguments.

__device__ unsigned int min(const int a, const unsigned int b)


Calculate the minimum value of the input `int` and `unsigned` `int` arguments.

__device__ long int min(const long int a, const long int b)


Calculate the minimum value of the input `long` `int` arguments.

__device__ unsigned int min(const unsigned int a, const unsigned int b)


Calculate the minimum value of the input `unsigned` `int` arguments.

__device__ unsigned long int min(const unsigned long int a, const long int b)


Calculate the minimum value of the input `unsigned` `long` `int` and `long` `int` arguments.

__device__ unsigned long int min(const unsigned long int a, const unsigned long int b)


Calculate the minimum value of the input `unsigned` `long` `int` arguments.

__device__ unsigned long long int ullmax(const unsigned long long int a, const unsigned long long int b)


Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned long long int ullmin(const unsigned long long int a, const unsigned long long int b)


Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

__device__ unsigned int umax(const unsigned int a, const unsigned int b)


Calculate the maximum value of the input `unsigned` `int` arguments.

__device__ unsigned int umin(const unsigned int a, const unsigned int b)


Calculate the minimum value of the input `unsigned` `int` arguments.

##  12.1. Functions

__device__ long int abs(long int a)



Calculate the absolute value of the input `long` `int` argument.

Calculate the absolute value of the input argument `a`.

Returns


Returns the absolute value of the input argument.

  * abs(`LONG_MIN`) is `Undefined`


__device__ int abs(int a)



Calculate the absolute value of the input `int` argument.

Calculate the absolute value of the input argument `a`.

Returns


Returns the absolute value of the input argument.

  * abs(`INT_MIN`) is `Undefined`


__device__ long long int abs(long long int a)



Calculate the absolute value of the input `long` `long` `int` argument.

Calculate the absolute value of the input argument `a`.

Returns


Returns the absolute value of the input argument.

  * abs(`LLONG_MIN`) is `Undefined`


__device__ long int labs(long int a)



Calculate the absolute value of the input `long` `int` argument.

Calculate the absolute value of the input argument `a`.

Returns


Returns the absolute value of the input argument.

  * labs(`LONG_MIN`) is `Undefined`


__device__ long long int llabs(long long int a)



Calculate the absolute value of the input `long` `long` `int` argument.

Calculate the absolute value of the input argument `a`.

Returns


Returns the absolute value of the input argument.

  * llabs(`LLONG_MIN`) is `Undefined`


__device__ long long int llmax(const long long int a, const long long int b)



Calculate the maximum value of the input `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ long long int llmin(const long long int a, const long long int b)



Calculate the minimum value of the input `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned long int max(const long int a, const unsigned long int b)



Calculate the maximum value of the input `long` `int` and `unsigned` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long long int max(const unsigned long long int a, const unsigned long long int b)



Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned int max(const unsigned int a, const int b)



Calculate the maximum value of the input `unsigned` `int` and `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long long int max(const long long int a, const unsigned long long int b)



Calculate the maximum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long int max(const unsigned long int a, const unsigned long int b)



Calculate the maximum value of the input `unsigned` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ long long int max(const long long int a, const long long int b)



Calculate the maximum value of the input `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned long long int max(const unsigned long long int a, const long long int b)



Calculate the maximum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long int max(const unsigned long int a, const long int b)



Calculate the maximum value of the input `unsigned` `long` `int` and `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ long int max(const long int a, const long int b)



Calculate the maximum value of the input `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ int max(const int a, const int b)



Calculate the maximum value of the input `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned int max(const unsigned int a, const unsigned int b)



Calculate the maximum value of the input `unsigned` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned int max(const int a, const unsigned int b)



Calculate the maximum value of the input `int` and `unsigned` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long int min(const long int a, const unsigned long int b)



Calculate the minimum value of the input `long` `int` and `unsigned` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long long int min(const unsigned long long int a, const unsigned long long int b)



Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned long long int min(const unsigned long long int a, const long long int b)



Calculate the minimum value of the input `unsigned` `long` `long` `int` and `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ int min(const int a, const int b)



Calculate the minimum value of the input `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned int min(const unsigned int a, const int b)



Calculate the minimum value of the input `unsigned` `int` and `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long long int min(const long long int a, const unsigned long long int b)



Calculate the minimum value of the input `long` `long` `int` and `unsigned` `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ long long int min(const long long int a, const long long int b)



Calculate the minimum value of the input `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned int min(const int a, const unsigned int b)



Calculate the minimum value of the input `int` and `unsigned` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ long int min(const long int a, const long int b)



Calculate the minimum value of the input `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned int min(const unsigned int a, const unsigned int b)



Calculate the minimum value of the input `unsigned` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned long int min(const unsigned long int a, const long int b)



Calculate the minimum value of the input `unsigned` `long` `int` and `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`, perform integer promotion first.

__device__ unsigned long int min(const unsigned long int a, const unsigned long int b)



Calculate the minimum value of the input `unsigned` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned long long int ullmax(const unsigned long long int a, const unsigned long long int b)



Calculate the maximum value of the input `unsigned` `long` `long` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned long long int ullmin(const unsigned long long int a, const unsigned long long int b)



Calculate the minimum value of the input `unsigned` `long` `long` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.

__device__ unsigned int umax(const unsigned int a, const unsigned int b)



Calculate the maximum value of the input `unsigned` `int` arguments.

Calculate the maximum value of the arguments `a` and `b`.

__device__ unsigned int umin(const unsigned int a, const unsigned int b)



Calculate the minimum value of the input `unsigned` `int` arguments.

Calculate the minimum value of the arguments `a` and `b`.