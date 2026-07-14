# 13. Integer Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__INT.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)

---

#  13\. Integer Intrinsics[](#integer-intrinsics "Permalink to this headline")

This section describes integer intrinsic functions. 

All of these functions are supported in device code. For some of the functions, host-specific implementations are also provided. For example, see `[__nv_bswap16()](#group__cuda__math__intrinsic__int_1ga4641c0ff23b709195539777c655f8f68)`. To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ unsigned int [__brev](#group__cuda__math__intrinsic__int_1gac84ef1115946870edbcebddc0d539c5a)(unsigned int x)
    

Reverse the bit order of a 32-bit unsigned integer.

__device__ unsigned long long int [__brevll](#group__cuda__math__intrinsic__int_1ga554354459699a11acfb072cf536974cd)(unsigned long long int x)
    

Reverse the bit order of a 64-bit unsigned integer.

__device__ unsigned int [__byte_perm](#group__cuda__math__intrinsic__int_1ga0b8b8156fe619205bf8d65acd8f29131)(unsigned int x, unsigned int y, unsigned int s)
    

Return selected bytes from two 32-bit unsigned integers.

__device__ __CLZ_RETURN_TYPE [__clz](#group__cuda__math__intrinsic__int_1gabaea5b9bb7d2c017f1483ebb3a470535)(__CLZ_PARAMETER_TYPE x)
    

Return the number of consecutive high-order zero bits in a 32-bit integer.

__device__ __CLZ_RETURN_TYPE [__clzll](#group__cuda__math__intrinsic__int_1gadfcfbb26591ff3b72c6903effa20b8c6)(__CLZLL_PARAMETER_TYPE x)
    

Count the number of consecutive high-order zero bits in a 64-bit integer.

__device__ int [__dp2a_hi](#group__cuda__math__intrinsic__int_1ga00191d5b570bdfe73fb2e303d7d512ec)(int srcA, int srcB, int c)
    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int [__dp2a_hi](#group__cuda__math__intrinsic__int_1ga0cd92b14a229973cb1447322e3027dce)(unsigned int srcA, unsigned int srcB, unsigned int c)
    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int [__dp2a_hi](#group__cuda__math__intrinsic__int_1ga906765cd4ccd9a17d31d5dbcc9c65bff)(ushort2 srcA, uchar4 srcB, unsigned int c)
    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input.

__device__ int [__dp2a_hi](#group__cuda__math__intrinsic__int_1gaebad383283dfdaea5748fcc39f85757e)(short2 srcA, char4 srcB, int c)
    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input.

__device__ unsigned int [__dp2a_lo](#group__cuda__math__intrinsic__int_1ga321b0e1408f8d24571a673c324fed41a)(ushort2 srcA, uchar4 srcB, unsigned int c)
    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

__device__ int [__dp2a_lo](#group__cuda__math__intrinsic__int_1ga32df0ce9fb758b7103671563694644a7)(short2 srcA, char4 srcB, int c)
    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

__device__ unsigned int [__dp2a_lo](#group__cuda__math__intrinsic__int_1ga74105ec4bb95f6124231adeaf44bee27)(unsigned int srcA, unsigned int srcB, unsigned int c)
    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input.

__device__ int [__dp2a_lo](#group__cuda__math__intrinsic__int_1gaa239a450d1919fe98fd29d8619dc9273)(int srcA, int srcB, int c)
    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input.

__device__ unsigned int [__dp4a](#group__cuda__math__intrinsic__int_1ga5a028e13382462fec36e03a8cbd0ce2b)(uchar4 srcA, uchar4 srcB, unsigned int c)
    

Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

__device__ unsigned int [__dp4a](#group__cuda__math__intrinsic__int_1ga5a68396e517a994093daf7f7a1090191)(unsigned int srcA, unsigned int srcB, unsigned int c)
    

Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate.

__device__ int [__dp4a](#group__cuda__math__intrinsic__int_1ga933213059df6da2de206771f145ac2f8)(int srcA, int srcB, int c)
    

Four-way `signed` `int8` dot product with `int32` accumulate.

__device__ int [__dp4a](#group__cuda__math__intrinsic__int_1ga9fe10d2bf9fb3886ed8de73ebd00ef3a)(char4 srcA, char4 srcB, int c)
    

Four-way `signed` `int8` dot product with `int32` accumulate.

__device__ int [__ffs](#group__cuda__math__intrinsic__int_1gaaf1eb22243e29e0b7222adee8ae7d4e4)(int x)
    

Find the position of the least significant bit set to 1 in a 32-bit integer.

__device__ int [__ffsll](#group__cuda__math__intrinsic__int_1gac42b0f1b68e9db038015d40a18d305df)(long long int x)
    

Find the position of the least significant bit set to 1 in a 64-bit integer.

__device__ unsigned [__fns](#group__cuda__math__intrinsic__int_1ga2fc8e909eb9a959dcc3262e54365bfc5)(unsigned mask, unsigned base, int offset)
    

Find the position of the n-th set to 1 bit in a 32-bit integer.

__device__ unsigned int [__funnelshift_l](#group__cuda__math__intrinsic__int_1gaf939c350eafa2f13d64e278549d3a8aa)(unsigned int lo, unsigned int hi, unsigned int shift)
    

Concatenate `hi` : `lo` , shift left by `shift` & 31 bits, return the most significant 32 bits.

__device__ unsigned int [__funnelshift_lc](#group__cuda__math__intrinsic__int_1ga6afcc1126ca2bf68f74c34812cbde57d)(unsigned int lo, unsigned int hi, unsigned int shift)
    

Concatenate `hi` : `lo` , shift left by min( `shift` , 32) bits, return the most significant 32 bits.

__device__ unsigned int [__funnelshift_r](#group__cuda__math__intrinsic__int_1ga125eeef4993d16dc8d679b239460fc34)(unsigned int lo, unsigned int hi, unsigned int shift)
    

Concatenate `hi` : `lo` , shift right by `shift` & 31 bits, return the least significant 32 bits.

__device__ unsigned int [__funnelshift_rc](#group__cuda__math__intrinsic__int_1ga66b72e69ee8799b49365b218051b1e4d)(unsigned int lo, unsigned int hi, unsigned int shift)
    

Concatenate `hi` : `lo` , shift right by min( `shift` , 32) bits, return the least significant 32 bits.

__device__ int [__hadd](#group__cuda__math__intrinsic__int_1gac030c6e3e8d0117a0d081e7397ba40dd)(int x, int y)
    

Compute average of signed input arguments, avoiding overflow in the intermediate sum.

__device__ int [__mul24](#group__cuda__math__intrinsic__int_1gaffff4cfc8958ec96329c11a797146573)(int x, int y)
    

Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.

__device__ long long int [__mul64hi](#group__cuda__math__intrinsic__int_1ga29b1d773d9393278aae147362827403b)(long long int x, long long int y)
    

Calculate the most significant 64 bits of the product of the two 64-bit integers.

__device__ int [__mulhi](#group__cuda__math__intrinsic__int_1gac3afa952add7894d6dda0931476d4882)(int x, int y)
    

Calculate the most significant 32 bits of the product of the two 32-bit integers.

__host__ __device__ unsigned short [__nv_bswap16](#group__cuda__math__intrinsic__int_1ga4641c0ff23b709195539777c655f8f68)(unsigned short x)
    

Reverse the order of bytes of the 16-bit unsigned integer.

__host__ __device__ unsigned int [__nv_bswap32](#group__cuda__math__intrinsic__int_1gabd869c83628fa594acdd07712244b10b)(unsigned int x)
    

Reverse the order of bytes of the 32-bit unsigned integer.

__host__ __device__ unsigned long long [__nv_bswap64](#group__cuda__math__intrinsic__int_1ga59d715c5cfb7a780441b4ebe6d1e5e7b)(unsigned long long x)
    

Reverse the order of bytes of the 64-bit unsigned integer.

__device__ int [__popc](#group__cuda__math__intrinsic__int_1ga43c9c7d2b9ebf202ff1ef5769989be46)(unsigned int x)
    

Count the number of bits that are set to 1 in a 32-bit integer.

__device__ int [__popcll](#group__cuda__math__intrinsic__int_1ga6bb2f1cc3495f3f2ebe1df6ec52b3aec)(unsigned long long int x)
    

Count the number of bits that are set to 1 in a 64-bit integer.

__device__ int [__rhadd](#group__cuda__math__intrinsic__int_1ga92bcad377730dc44cb65ab2c36ca6e43)(int x, int y)
    

Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int [__sad](#group__cuda__math__intrinsic__int_1gae955ffc59084b5600dd5d223fb26beb0)(int x, int y, unsigned int z)
    

Calculate \\(|x - y| + z\\) , the sum of absolute difference.

__device__ unsigned int [__uhadd](#group__cuda__math__intrinsic__int_1gae84cd786a304ddbecfc2bcdb10b06099)(unsigned int x, unsigned int y)
    

Compute average of unsigned input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int [__umul24](#group__cuda__math__intrinsic__int_1ga2b1446551e854d164e3d4aae25f94a3f)(unsigned int x, unsigned int y)
    

Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.

__device__ unsigned long long int [__umul64hi](#group__cuda__math__intrinsic__int_1gaefaf2a1bb2ad8986b2c248b74461e12e)(unsigned long long int x, unsigned long long int y)
    

Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers.

__device__ unsigned int [__umulhi](#group__cuda__math__intrinsic__int_1ga2ea43def3122a1bfb3eb8271a59f4470)(unsigned int x, unsigned int y)
    

Calculate the most significant 32 bits of the product of the two 32-bit unsigned integers.

__device__ unsigned int [__urhadd](#group__cuda__math__intrinsic__int_1ga6f364981d9731158e400bae169490b20)(unsigned int x, unsigned int y)
    

Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum.

__device__ unsigned int [__usad](#group__cuda__math__intrinsic__int_1ga58e819af40fc4a3e9e9c3a4c13cbd0e2)(unsigned int x, unsigned int y, unsigned int z)
    

Calculate \\(|x - y| + z\\) , the sum of absolute difference.

##  13.1. Functions[](#functions "Permalink to this headline")

__device__ unsigned int __brev(unsigned int x)[](#_CPPv46__brevj "Permalink to this definition")  

    

Reverse the bit order of a 32-bit unsigned integer. 

Reverses the bit order of the 32-bit unsigned integer `x`.

Returns
    

Returns the bit-reversed value of `x`. i.e. bit N of the return value corresponds to bit 31-N of `x`. 

__device__ unsigned long long int __brevll(unsigned long long int x)[](#_CPPv48__brevlly "Permalink to this definition")  

    

Reverse the bit order of a 64-bit unsigned integer. 

Reverses the bit order of the 64-bit unsigned integer `x`.

Returns
    

Returns the bit-reversed value of `x`. i.e. bit N of the return value corresponds to bit 63-N of `x`. 

__device__ unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)[](#_CPPv411__byte_permjjj "Permalink to this definition")  

    

Return selected bytes from two 32-bit unsigned integers. 

Create 8-byte source

  * uint64_t `tmp64` = ((uint64_t)`y` << 32) | `x`;


Extract selector bits

  * `selector0` = (`s` >> 0) & 0x7;

  * `selector1` = (`s` >> 4) & 0x7;

  * `selector2` = (`s` >> 8) & 0x7;

  * `selector3` = (`s` >> 12) & 0x7;


Return 4 selected bytes from 8-byte source:

  * `res`[07:00] = `tmp64`[`selector0`];

  * `res`[15:08] = `tmp64`[`selector1`];

  * `res`[23:16] = `tmp64`[`selector2`];

  * `res`[31:24] = `tmp64`[`selector3`]; 


Returns
    

Returns a 32-bit integer consisting of four bytes from eight input bytes provided in the two input integers `x` and `y`, as specified by a selector, `s`.

__device__ __CLZ_RETURN_TYPE __clz(__CLZ_PARAMETER_TYPE x)[](#_CPPv45__clz20__CLZ_PARAMETER_TYPE "Permalink to this definition")  

    

Return the number of consecutive high-order zero bits in a 32-bit integer. 

Count the number of consecutive leading zero bits, starting at the most significant bit (bit 31) of `x`.

To accomodate to ACLE builtins,

  * on ARM64 with GCC 11.4 or later as the host compiler, __CLZ_RETURN_TYPE is ‘unsigned int’ and __CLZ_PARAMETER_TYPE is ‘unsigned int’.

  * for all other cases, __CLZ_RETURN_TYPE is ‘int’ and __CLZ_PARAMETER_TYPE is ‘int’.


Returns
    

Returns a value between 0 and 32 inclusive representing the number of zero bits. 

__device__ __CLZ_RETURN_TYPE __clzll(__CLZLL_PARAMETER_TYPE x)[](#_CPPv47__clzll22__CLZLL_PARAMETER_TYPE "Permalink to this definition")  

    

Count the number of consecutive high-order zero bits in a 64-bit integer. 

Count the number of consecutive leading zero bits, starting at the most significant bit (bit 63) of `x`.

To accomodate to ACLE builtins,

  * on ARM64 with GCC 11.4 or later as the host compiler, __CLZ_RETURN_TYPE is ‘unsigned int’ and __CLZLL_PARAMETER_TYPE is ‘unsigned long int’.

  * for all other cases, __CLZ_RETURN_TYPE is ‘int’ and __CLZLL_PARAMETER_TYPE is ‘long long int’.


Returns
    

Returns a value between 0 and 64 inclusive representing the number of zero bits. 

__device__ int __dp2a_hi(int srcA, int srcB, int c)[](#_CPPv49__dp2a_hiiii "Permalink to this definition")  

    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input. 

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the upper 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`. 

__device__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c)[](#_CPPv49__dp2a_hijjj "Permalink to this definition")  

    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input. 

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the upper 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`. 

__device__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c)[](#_CPPv49__dp2a_hi7ushort26uchar4j "Permalink to this definition")  

    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the upper half of the second input. 

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the upper 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`. 

__device__ int __dp2a_hi(short2 srcA, char4 srcB, int c)[](#_CPPv49__dp2a_hi6short25char4i "Permalink to this definition")  

    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the upper half of the second input. 

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the upper 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`. 

__device__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c)[](#_CPPv49__dp2a_lo7ushort26uchar4j "Permalink to this definition")  

    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input. 

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the lower 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`. 

__device__ int __dp2a_lo(short2 srcA, char4 srcB, int c)[](#_CPPv49__dp2a_lo6short25char4i "Permalink to this definition")  

    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input. 

Takes two packed 16-bit integers from `scrA` vector and two packed 8-bit integers from the lower 16 bits of `srcB` vector, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`. 

__device__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c)[](#_CPPv49__dp2a_lojjj "Permalink to this definition")  

    

Two-way `unsigned` `int16` by `int8` dot product with `unsigned` `int32` accumulate, taking the lower half of the second input. 

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the lower 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to an unsigned 32-bit integer `c`. 

__device__ int __dp2a_lo(int srcA, int srcB, int c)[](#_CPPv49__dp2a_loiii "Permalink to this definition")  

    

Two-way `signed` `int16` by `int8` dot product with `int32` accumulate, taking the lower half of the second input. 

Extracts two packed 16-bit integers from `scrA` and two packed 8-bit integers from the lower 16 bits of `srcB`, then creates two pairwise 8x16 products and adds them together to a signed 32-bit integer `c`. 

__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c)[](#_CPPv46__dp4a6uchar46uchar4j "Permalink to this definition")  

    

Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate. 

Takes four pairs of packed byte-sized integers from `scrA` and `srcB` vectors, then creates four pairwise products and adds them together to an unsigned 32-bit integer `c`. 

__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c)[](#_CPPv46__dp4ajjj "Permalink to this definition")  

    

Four-way `unsigned` `int8` dot product with `unsigned` `int32` accumulate. 

Extracts four pairs of packed byte-sized integers from `scrA` and `srcB`, then creates four pairwise products and adds them together to an unsigned 32-bit integer `c`. 

__device__ int __dp4a(int srcA, int srcB, int c)[](#_CPPv46__dp4aiii "Permalink to this definition")  

    

Four-way `signed` `int8` dot product with `int32` accumulate. 

Extracts four pairs of packed byte-sized integers from `scrA` and `srcB`, then creates four pairwise products and adds them together to a signed 32-bit integer `c`. 

__device__ int __dp4a(char4 srcA, char4 srcB, int c)[](#_CPPv46__dp4a5char45char4i "Permalink to this definition")  

    

Four-way `signed` `int8` dot product with `int32` accumulate. 

Takes four pairs of packed byte-sized integers from `scrA` and `srcB` vectors, then creates four pairwise products and adds them together to a signed 32-bit integer `c`. 

__device__ int __ffs(int x)[](#_CPPv45__ffsi "Permalink to this definition")  

    

Find the position of the least significant bit set to 1 in a 32-bit integer. 

Find the position of the first (least significant) bit set to 1 in `x`, where the least significant bit position is 1.

Returns
    

Returns a value between 0 and 32 inclusive representing the position of the first bit set.

  * __ffs(0) returns 0. 


__device__ int __ffsll(long long int x)[](#_CPPv47__ffsllx "Permalink to this definition")  

    

Find the position of the least significant bit set to 1 in a 64-bit integer. 

Find the position of the first (least significant) bit set to 1 in `x`, where the least significant bit position is 1.

Returns
    

Returns a value between 0 and 64 inclusive representing the position of the first bit set.

  * __ffsll(0) returns 0. 


__device__ unsigned __fns(unsigned mask, unsigned base, int offset)[](#_CPPv45__fnsjji "Permalink to this definition")  

    

Find the position of the n-th set to 1 bit in a 32-bit integer. 

Given a 32-bit value `mask` and an integer value `base` (between 0 and 31), find the n-th (given by `offset`) set bit in `mask` from the `base` bit. If not found, return 0xFFFFFFFF.

See also <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns> for more information.

Returns
    

Returns a value between 0 and 32 inclusive representing the position of the n-th set bit.

  * parameter `base` must be <=31, otherwise behavior is undefined. 


__device__ unsigned int __funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)[](#_CPPv415__funnelshift_ljjj "Permalink to this definition")  

    

Concatenate `hi` : `lo`, shift left by `shift` & 31 bits, return the most significant 32 bits. 

Shift the 64-bit value formed by concatenating argument `lo` and `hi` left by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted left by the wrapped value of `shift` (`shift` & 31). The most significant 32-bits of the result are returned.

Returns
    

Returns the most significant 32 bits of the shifted 64-bit value. 

__device__ unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)[](#_CPPv416__funnelshift_lcjjj "Permalink to this definition")  

    

Concatenate `hi` : `lo`, shift left by min(`shift`, 32) bits, return the most significant 32 bits. 

Shift the 64-bit value formed by concatenating argument `lo` and `hi` left by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted left by the clamped value of `shift` (min(`shift`, 32)). The most significant 32-bits of the result are returned.

Returns
    

Returns the most significant 32 bits of the shifted 64-bit value. 

__device__ unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)[](#_CPPv415__funnelshift_rjjj "Permalink to this definition")  

    

Concatenate `hi` : `lo`, shift right by `shift` & 31 bits, return the least significant 32 bits. 

Shift the 64-bit value formed by concatenating argument `lo` and `hi` right by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted right by the wrapped value of `shift` (`shift` & 31). The least significant 32-bits of the result are returned.

Returns
    

Returns the least significant 32 bits of the shifted 64-bit value. 

__device__ unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)[](#_CPPv416__funnelshift_rcjjj "Permalink to this definition")  

    

Concatenate `hi` : `lo`, shift right by min(`shift`, 32) bits, return the least significant 32 bits. 

Shift the 64-bit value formed by concatenating argument `lo` and `hi` right by the amount specified by the argument `shift`. Argument `lo` holds bits 31:0 and argument `hi` holds bits 63:32 of the 64-bit source value. The source is shifted right by the clamped value of `shift` (min(`shift`, 32)). The least significant 32-bits of the result are returned.

Returns
    

Returns the least significant 32 bits of the shifted 64-bit value. 

__device__ int __hadd(int x, int y)[](#_CPPv46__haddii "Permalink to this definition")  

    

Compute average of signed input arguments, avoiding overflow in the intermediate sum. 

Compute average of signed input arguments `x` and `y` as ( `x` \+ `y` ) >> 1, avoiding overflow in the intermediate sum.

Returns
    

Returns a signed integer value representing the signed average value of the two inputs. 

__device__ int __mul24(int x, int y)[](#_CPPv47__mul24ii "Permalink to this definition")  

    

Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers. 

Calculate the least significant 32 bits of the product of the least significant 24 bits of `x` and `y`. The high order 8 bits of `x` and `y` are ignored.

Returns
    

Returns the least significant 32 bits of the product `x` * `y`. 

__device__ long long int __mul64hi(long long int x, long long int y)[](#_CPPv49__mul64hixx "Permalink to this definition")  

    

Calculate the most significant 64 bits of the product of the two 64-bit integers. 

Calculate the most significant 64 bits of the 128-bit product `x` * `y`, where `x` and `y` are 64-bit integers.

Returns
    

Returns the most significant 64 bits of the product `x` * `y`. 

__device__ int __mulhi(int x, int y)[](#_CPPv47__mulhiii "Permalink to this definition")  

    

Calculate the most significant 32 bits of the product of the two 32-bit integers. 

Calculate the most significant 32 bits of the 64-bit product `x` * `y`, where `x` and `y` are 32-bit integers.

Returns
    

Returns the most significant 32 bits of the product `x` * `y`. 

__host__ __device__ unsigned short __nv_bswap16(unsigned short x)[](#_CPPv412__nv_bswap16t "Permalink to this definition")  

    

Reverse the order of bytes of the 16-bit unsigned integer. 

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns
    

Returns `x` with the order of bytes reversed. 

__host__ __device__ unsigned int __nv_bswap32(unsigned int x)[](#_CPPv412__nv_bswap32j "Permalink to this definition")  

    

Reverse the order of bytes of the 32-bit unsigned integer. 

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns
    

Returns `x` with the order of bytes reversed. 

__host__ __device__ unsigned long long __nv_bswap64(unsigned long long x)[](#_CPPv412__nv_bswap64y "Permalink to this definition")  

    

Reverse the order of bytes of the 64-bit unsigned integer. 

Reverse the order of bytes of `x` . Only supported in MSVC and other host compilers which define the `__GNUC__` macro, such as GCC and CLANG.

Returns
    

Returns `x` with the order of bytes reversed. 

__device__ int __popc(unsigned int x)[](#_CPPv46__popcj "Permalink to this definition")  

    

Count the number of bits that are set to 1 in a 32-bit integer. 

Count the number of bits that are set to 1 in `x`.

Returns
    

Returns a value between 0 and 32 inclusive representing the number of set bits. 

__device__ int __popcll(unsigned long long int x)[](#_CPPv48__popclly "Permalink to this definition")  

    

Count the number of bits that are set to 1 in a 64-bit integer. 

Count the number of bits that are set to 1 in `x`.

Returns
    

Returns a value between 0 and 64 inclusive representing the number of set bits. 

__device__ int __rhadd(int x, int y)[](#_CPPv47__rhaddii "Permalink to this definition")  

    

Compute rounded average of signed input arguments, avoiding overflow in the intermediate sum. 

Compute average of signed input arguments `x` and `y` as ( `x` \+ `y` \+ 1 ) >> 1, avoiding overflow in the intermediate sum.

Returns
    

Returns a signed integer value representing the signed rounded average value of the two inputs. 

__device__ unsigned int __sad(int x, int y, unsigned int z)[](#_CPPv45__sadiij "Permalink to this definition")  

    

Calculate \\( |x - y| + z \\) , the sum of absolute difference. 

Calculate \\( |x - y| + z \\) , the 32-bit sum of the third argument `z` plus and the absolute value of the difference between the first argument, `x`, and second argument, `y`.

Inputs `x` and `y` are signed 32-bit integers, input `z` is a 32-bit unsigned integer.

Returns
    

Returns \\( |x - y| + z \\). 

__device__ unsigned int __uhadd(unsigned int x, unsigned int y)[](#_CPPv47__uhaddjj "Permalink to this definition")  

    

Compute average of unsigned input arguments, avoiding overflow in the intermediate sum. 

Compute average of unsigned input arguments `x` and `y` as ( `x` \+ `y` ) >> 1, avoiding overflow in the intermediate sum.

Returns
    

Returns an unsigned integer value representing the unsigned average value of the two inputs. 

__device__ unsigned int __umul24(unsigned int x, unsigned int y)[](#_CPPv48__umul24jj "Permalink to this definition")  

    

Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers. 

Calculate the least significant 32 bits of the product of the least significant 24 bits of `x` and `y`. The high order 8 bits of `x` and `y` are ignored.

Returns
    

Returns the least significant 32 bits of the product `x` * `y`. 

__device__ unsigned long long int __umul64hi(unsigned long long int x, unsigned long long int y)[](#_CPPv410__umul64hiyy "Permalink to this definition")  

    

Calculate the most significant 64 bits of the product of the two 64 unsigned bit integers. 

Calculate the most significant 64 bits of the 128-bit product `x` * `y`, where `x` and `y` are 64-bit unsigned integers.

Returns
    

Returns the most significant 64 bits of the product `x` * `y`. 

__device__ unsigned int __umulhi(unsigned int x, unsigned int y)[](#_CPPv48__umulhijj "Permalink to this definition")  

    

Calculate the most significant 32 bits of the product of the two 32-bit unsigned integers. 

Calculate the most significant 32 bits of the 64-bit product `x` * `y`, where `x` and `y` are 32-bit unsigned integers.

Returns
    

Returns the most significant 32 bits of the product `x` * `y`. 

__device__ unsigned int __urhadd(unsigned int x, unsigned int y)[](#_CPPv48__urhaddjj "Permalink to this definition")  

    

Compute rounded average of unsigned input arguments, avoiding overflow in the intermediate sum. 

Compute average of unsigned input arguments `x` and `y` as ( `x` \+ `y` \+ 1 ) >> 1, avoiding overflow in the intermediate sum.

Returns
    

Returns an unsigned integer value representing the unsigned rounded average value of the two inputs. 

__device__ unsigned int __usad(unsigned int x, unsigned int y, unsigned int z)[](#_CPPv46__usadjjj "Permalink to this definition")  

    

Calculate \\( |x - y| + z \\) , the sum of absolute difference. 

Calculate \\( |x - y| + z \\) , the 32-bit sum of the third argument `z` plus and the absolute value of the difference between the first argument, `x`, and second argument, `y`.

Inputs `x`, `y`, and `z` are unsigned 32-bit integers.

Returns
    

Returns \\( |x - y| + z \\).