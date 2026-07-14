# 14. SIMD Intrinsics

**Source:** [group__CUDA__MATH__INTRINSIC__SIMD.html](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)

---

#  14\. SIMD Intrinsics[](#simd-intrinsics "Permalink to this headline")

This section describes SIMD intrinsic functions that are only supported in device code. 

To use these functions, you do not need to include any additional header file in your program. 

Functions

__device__ unsigned int [__vabs2](#group__cuda__math__intrinsic__simd_1gab29295ac0d7b95d3a48a92a2b34e0b44)(unsigned int a)
    

Computes per-halfword absolute value: |a|.

__device__ unsigned int [__vabs4](#group__cuda__math__intrinsic__simd_1ga349447d947f0e63ffa1eeb2703db0847)(unsigned int a)
    

Computes per-byte absolute value: |a|.

__device__ unsigned int [__vabsdiffs2](#group__cuda__math__intrinsic__simd_1ga03b9557c20cc57ff06ae1a5cce97faa0)(unsigned int a, unsigned int b)
    

Computes per-halfword absolute difference of signed integer: |a - b|.

__device__ unsigned int [__vabsdiffs4](#group__cuda__math__intrinsic__simd_1ga5a21bd92816a35c45ce322eb14650d0a)(unsigned int a, unsigned int b)
    

Computes per-byte absolute difference of signed integer: |a - b|.

__device__ unsigned int [__vabsdiffu2](#group__cuda__math__intrinsic__simd_1gaac1486f685f7033e85b6ab0067f99815)(unsigned int a, unsigned int b)
    

Computes per-halfword absolute difference of unsigned integer: |a - b|.

__device__ unsigned int [__vabsdiffu4](#group__cuda__math__intrinsic__simd_1gad69242d176526aa536d59e7e1fee7f06)(unsigned int a, unsigned int b)
    

Computes per-byte absolute difference of unsigned integer: |a - b|.

__device__ unsigned int [__vabsss2](#group__cuda__math__intrinsic__simd_1ga098fa7e88d1bf9ab09ff7930b48895f6)(unsigned int a)
    

Computes per-halfword absolute value with signed saturation: |a|.

__device__ unsigned int [__vabsss4](#group__cuda__math__intrinsic__simd_1ga4b780280a83f8e40d68f948a365d73cf)(unsigned int a)
    

Computes per-byte absolute value with signed saturation: |a|.

__device__ unsigned int [__vadd2](#group__cuda__math__intrinsic__simd_1ga90b011f2d59f67b01a3b747f1b4387d8)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed addition, with wrap-around: a + b.

__device__ unsigned int [__vadd4](#group__cuda__math__intrinsic__simd_1ga985e19defa6381f163004ac5dd6e68e8)(unsigned int a, unsigned int b)
    

Performs per-byte (un)signed addition: a + b.

__device__ unsigned int [__vaddss2](#group__cuda__math__intrinsic__simd_1gac3a123787a00cf218837fb698261a82b)(unsigned int a, unsigned int b)
    

Performs per-halfword addition with signed saturation: a + b.

__device__ unsigned int [__vaddss4](#group__cuda__math__intrinsic__simd_1ga644b48c38da5bb44caec16eca4930297)(unsigned int a, unsigned int b)
    

Performs per-byte addition with signed saturation: a + b.

__device__ unsigned int [__vaddus2](#group__cuda__math__intrinsic__simd_1ga0945c2ee112bdfc4dacffcdcd5ee567c)(unsigned int a, unsigned int b)
    

Performs per-halfword addition with unsigned saturation: a + b.

__device__ unsigned int [__vaddus4](#group__cuda__math__intrinsic__simd_1gaae4ef43fa9f898b19dc4e43e3d3b38a0)(unsigned int a, unsigned int b)
    

Performs per-byte addition with unsigned saturation: a + b.

__device__ unsigned int [__vavgs2](#group__cuda__math__intrinsic__simd_1gaa6a2d0248863e6832914687d1b79abd7)(unsigned int a, unsigned int b)
    

Performs per-halfword signed rounded average computation.

__device__ unsigned int [__vavgs4](#group__cuda__math__intrinsic__simd_1ga6f6fa8cd3573e5e937a5b38d14e2d537)(unsigned int a, unsigned int b)
    

Computes per-byte signed rounded average.

__device__ unsigned int [__vavgu2](#group__cuda__math__intrinsic__simd_1ga39949ea8d92d8a96d91c6a77cbd23e2d)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned rounded average computation.

__device__ unsigned int [__vavgu4](#group__cuda__math__intrinsic__simd_1ga6e240de9679b7ddaddaec81cdc13de52)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned rounded average.

__device__ unsigned int [__vcmpeq2](#group__cuda__math__intrinsic__simd_1ga833153e5e2f30319d5b6306db7ebcfff)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed comparison: a == b ? 0xffff : 0.

__device__ unsigned int [__vcmpeq4](#group__cuda__math__intrinsic__simd_1gad20eb0df80985e3977ef0d2364101e05)(unsigned int a, unsigned int b)
    

Performs per-byte (un)signed comparison: a == b ? 0xff : 0.

__device__ unsigned int [__vcmpges2](#group__cuda__math__intrinsic__simd_1gab311e0b61691081df934edd1d46d44bb)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: a >= b ? 0xffff : 0.

__device__ unsigned int [__vcmpges4](#group__cuda__math__intrinsic__simd_1ga3b99d880125c9a7375b6b1df28df1fd3)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: a >= b ? 0xff : 0.

__device__ unsigned int [__vcmpgeu2](#group__cuda__math__intrinsic__simd_1ga94d245ad243307f5056b2640b450461a)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0.

__device__ unsigned int [__vcmpgeu4](#group__cuda__math__intrinsic__simd_1gabcd1b8cbe6b52d01477ea22faa420f16)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: a >= b ? 0xff : 0.

__device__ unsigned int [__vcmpgts2](#group__cuda__math__intrinsic__simd_1ga63b3f1467246e19a2bc0ab00d71abf6b)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: a > b ? 0xffff : 0.

__device__ unsigned int [__vcmpgts4](#group__cuda__math__intrinsic__simd_1ga412e4e95de0c4a96078338056197ef6a)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: a > b ? 0xff : 0.

__device__ unsigned int [__vcmpgtu2](#group__cuda__math__intrinsic__simd_1ga1cc37b85086626d986c9e1449922440b)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: a > b ? 0xffff : 0.

__device__ unsigned int [__vcmpgtu4](#group__cuda__math__intrinsic__simd_1gaccb78546a5bc327ee92f0fa6ad75d233)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: a > b ? 0xff : 0.

__device__ unsigned int [__vcmples2](#group__cuda__math__intrinsic__simd_1ga29c33e3aee2772c8977534e0c33fdf03)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: a <= b ? 0xffff : 0.

__device__ unsigned int [__vcmples4](#group__cuda__math__intrinsic__simd_1gaa29ba94e6a2400d71db039eb44067407)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: a <= b ? 0xff : 0.

__device__ unsigned int [__vcmpleu2](#group__cuda__math__intrinsic__simd_1ga14f940b7833ad41f5c09232c2ab30534)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0.

__device__ unsigned int [__vcmpleu4](#group__cuda__math__intrinsic__simd_1ga4045df2c71583305ac5df0c60bc4f371)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: a <= b ? 0xff : 0.

__device__ unsigned int [__vcmplts2](#group__cuda__math__intrinsic__simd_1ga391ba9661c65ef94a2710a59462e78ba)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: a < b ? 0xffff : 0.

__device__ unsigned int [__vcmplts4](#group__cuda__math__intrinsic__simd_1gaa93e4107e980fece166e58515c6888fd)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: a < b ? 0xff : 0.

__device__ unsigned int [__vcmpltu2](#group__cuda__math__intrinsic__simd_1ga2ccb689e812524cdc8cd498a60bebb4c)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: a < b ? 0xffff : 0.

__device__ unsigned int [__vcmpltu4](#group__cuda__math__intrinsic__simd_1gab315ad16339e8811b1ad5f66db271776)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: a < b ? 0xff : 0.

__device__ unsigned int [__vcmpne2](#group__cuda__math__intrinsic__simd_1ga7ca7014944cf78e4f1a40deeafe0c511)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0.

__device__ unsigned int [__vcmpne4](#group__cuda__math__intrinsic__simd_1ga75a2f66da3469d72be6abbcb971214cd)(unsigned int a, unsigned int b)
    

Performs per-byte (un)signed comparison: a != b ? 0xff : 0.

__device__ unsigned int [__vhaddu2](#group__cuda__math__intrinsic__simd_1gacdd77ff7f296f3b1034539823e98a8b7)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned average computation.

__device__ unsigned int [__vhaddu4](#group__cuda__math__intrinsic__simd_1gadd9c8eb1b9c654f0ec765b032c54f01b)(unsigned int a, unsigned int b)
    

Computes per-byte unsigned average.

__host__ __device__ unsigned int [__viaddmax_s16x2](#group__cuda__math__intrinsic__simd_1ga900ac993f9e89ad9ed887138970617fb)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(a + b, c)

__host__ __device__ unsigned int [__viaddmax_s16x2_relu](#group__cuda__math__intrinsic__simd_1gaa2cab252f19a1bf81113daccd2894937)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(max(a + b, c), 0)

__host__ __device__ int [__viaddmax_s32](#group__cuda__math__intrinsic__simd_1gadc03dd9067a7f1caea5cc95bb0861a62)(const int a, const int b, const int c)
    

Computes max(a + b, c)

__host__ __device__ int [__viaddmax_s32_relu](#group__cuda__math__intrinsic__simd_1ga4f7afd796c34428df6a76d43e8205c86)(const int a, const int b, const int c)
    

Computes max(max(a + b, c), 0)

__host__ __device__ unsigned int [__viaddmax_u16x2](#group__cuda__math__intrinsic__simd_1ga774f2d1d247a59a06ba6e5c7c8f266a4)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(a + b, c)

__host__ __device__ unsigned int [__viaddmax_u32](#group__cuda__math__intrinsic__simd_1gafd33057238289815b6e2f8857d6bdb5f)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Computes max(a + b, c)

__host__ __device__ unsigned int [__viaddmin_s16x2](#group__cuda__math__intrinsic__simd_1ga47adaffb0bbfdda51b90afcc83f22891)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword min(a + b, c)

__host__ __device__ unsigned int [__viaddmin_s16x2_relu](#group__cuda__math__intrinsic__simd_1gab54599ac7cf935c6df686bb597e8cde4)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(min(a + b, c), 0)

__host__ __device__ int [__viaddmin_s32](#group__cuda__math__intrinsic__simd_1ga45274b591410f27e9e134cc7ac43a42b)(const int a, const int b, const int c)
    

Computes min(a + b, c)

__host__ __device__ int [__viaddmin_s32_relu](#group__cuda__math__intrinsic__simd_1gaa48b75a99628f4f99f979334336a13dc)(const int a, const int b, const int c)
    

Computes max(min(a + b, c), 0)

__host__ __device__ unsigned int [__viaddmin_u16x2](#group__cuda__math__intrinsic__simd_1gacd5838a53956fcaf3dcfba93af621589)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword min(a + b, c)

__host__ __device__ unsigned int [__viaddmin_u32](#group__cuda__math__intrinsic__simd_1gaf451909f7fe7b42d0051b951684d6001)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Computes min(a + b, c)

__host__ __device__ unsigned int [__vibmax_s16x2](#group__cuda__math__intrinsic__simd_1ga7a01100495ded66ace5998ba9f92e9fa)(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)
    

Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

__host__ __device__ int [__vibmax_s32](#group__cuda__math__intrinsic__simd_1ga66dcb8846807ee2897a0c9f5dd09f2e7)(const int a, const int b, bool *const pred)
    

Computes max(a, b), also sets the value pointed to by pred to (a >= b).

__host__ __device__ unsigned int [__vibmax_u16x2](#group__cuda__math__intrinsic__simd_1ga811cbf97462f94781325d49e9532d45d)(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)
    

Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b).

__host__ __device__ unsigned int [__vibmax_u32](#group__cuda__math__intrinsic__simd_1ga6219a3491e93b870a7ee59fa4704d09a)(const unsigned int a, const unsigned int b, bool *const pred)
    

Computes max(a, b), also sets the value pointed to by pred to (a >= b).

__host__ __device__ unsigned int [__vibmin_s16x2](#group__cuda__math__intrinsic__simd_1ga9b75ae6e3da4c53dcedc5c252679a814)(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)
    

Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

__host__ __device__ int [__vibmin_s32](#group__cuda__math__intrinsic__simd_1ga2afff0beeb4f7089781a16bea7b7e9bf)(const int a, const int b, bool *const pred)
    

Computes min(a, b), also sets the value pointed to by pred to (a <= b).

__host__ __device__ unsigned int [__vibmin_u16x2](#group__cuda__math__intrinsic__simd_1gacdbd0f995b84e72556c9210fb8fb00c1)(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)
    

Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b).

__host__ __device__ unsigned int [__vibmin_u32](#group__cuda__math__intrinsic__simd_1ga701055ff460e3cf3d7c65e5a42b6bc3e)(const unsigned int a, const unsigned int b, bool *const pred)
    

Computes min(a, b), also sets the value pointed to by pred to (a <= b).

__host__ __device__ unsigned int [__vimax3_s16x2](#group__cuda__math__intrinsic__simd_1ga7d3c212f5bbf02e55e6a4c02f01737c7)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(max(a, b), c)

__host__ __device__ unsigned int [__vimax3_s16x2_relu](#group__cuda__math__intrinsic__simd_1gaa121d37a8fa533209999b9384f7fa3e5)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(max(max(a, b), c), 0)

__host__ __device__ int [__vimax3_s32](#group__cuda__math__intrinsic__simd_1ga59d95f5dcc48cfb22f2eb063fc80c30e)(const int a, const int b, const int c)
    

Computes max(max(a, b), c)

__host__ __device__ int [__vimax3_s32_relu](#group__cuda__math__intrinsic__simd_1ga04edce5f320e6a1e28da0ce643ac5a72)(const int a, const int b, const int c)
    

Computes max(max(max(a, b), c), 0)

__host__ __device__ unsigned int [__vimax3_u16x2](#group__cuda__math__intrinsic__simd_1gad8ed7566d823cb29b47e436cca52b444)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(max(a, b), c)

__host__ __device__ unsigned int [__vimax3_u32](#group__cuda__math__intrinsic__simd_1gab54251c6217e304ef58ea54944baa02f)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Computes max(max(a, b), c)

__host__ __device__ unsigned int [__vimax_s16x2_relu](#group__cuda__math__intrinsic__simd_1gac81a262771e2f1bd54b95e8aaff60a98)(const unsigned int a, const unsigned int b)
    

Performs per-halfword max(max(a, b), 0)

__host__ __device__ int [__vimax_s32_relu](#group__cuda__math__intrinsic__simd_1ga18a197348f6c3fdffb13e34800df97a0)(const int a, const int b)
    

Computes max(max(a, b), 0)

__host__ __device__ unsigned int [__vimin3_s16x2](#group__cuda__math__intrinsic__simd_1gabad39957c9ecebb366dda3fefb0b8205)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword min(min(a, b), c)

__host__ __device__ unsigned int [__vimin3_s16x2_relu](#group__cuda__math__intrinsic__simd_1ga5aae0ce40c11fa77ffa8c663ff7f09d2)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword max(min(min(a, b), c), 0)

__host__ __device__ int [__vimin3_s32](#group__cuda__math__intrinsic__simd_1ga8a73d21522d54d9647ee2ba99b05d1eb)(const int a, const int b, const int c)
    

Computes min(min(a, b), c)

__host__ __device__ int [__vimin3_s32_relu](#group__cuda__math__intrinsic__simd_1ga1f2ab1e2a7b6614e955c341775966a25)(const int a, const int b, const int c)
    

Computes max(min(min(a, b), c), 0)

__host__ __device__ unsigned int [__vimin3_u16x2](#group__cuda__math__intrinsic__simd_1ga559e8a980645635dc61af1638864db0c)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Performs per-halfword min(min(a, b), c)

__host__ __device__ unsigned int [__vimin3_u32](#group__cuda__math__intrinsic__simd_1gad58511f791c421b89d12d8d401843af1)(const unsigned int a, const unsigned int b, const unsigned int c)
    

Computes min(min(a, b), c)

__host__ __device__ unsigned int [__vimin_s16x2_relu](#group__cuda__math__intrinsic__simd_1gab41208ca59116cacc58828ff3fe66fb5)(const unsigned int a, const unsigned int b)
    

Performs per-halfword max(min(a, b), 0)

__host__ __device__ int [__vimin_s32_relu](#group__cuda__math__intrinsic__simd_1ga4fda6cd64c1788589bbed4f057706156)(const int a, const int b)
    

Computes max(min(a, b), 0)

__device__ unsigned int [__vmaxs2](#group__cuda__math__intrinsic__simd_1gacf8b25711ab67719039bfb4c18627810)(unsigned int a, unsigned int b)
    

Performs per-halfword signed maximum computation.

__device__ unsigned int [__vmaxs4](#group__cuda__math__intrinsic__simd_1ga5396b461a8a6509e57fd029fa9987875)(unsigned int a, unsigned int b)
    

Computes per-byte signed maximum.

__device__ unsigned int [__vmaxu2](#group__cuda__math__intrinsic__simd_1ga029cb2cae7f009f987e72b7445efd314)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned maximum computation.

__device__ unsigned int [__vmaxu4](#group__cuda__math__intrinsic__simd_1gab11d55e928c83710021d353061efce3e)(unsigned int a, unsigned int b)
    

Computes per-byte unsigned maximum.

__device__ unsigned int [__vmins2](#group__cuda__math__intrinsic__simd_1gafe8e71d1551babfe1b5c227b20882c3d)(unsigned int a, unsigned int b)
    

Performs per-halfword signed minimum computation.

__device__ unsigned int [__vmins4](#group__cuda__math__intrinsic__simd_1ga83bb6962d3dab503490f634f18b35574)(unsigned int a, unsigned int b)
    

Computes per-byte signed minimum.

__device__ unsigned int [__vminu2](#group__cuda__math__intrinsic__simd_1ga9538c62fe263f5e493b22fae7c92e159)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned minimum computation.

__device__ unsigned int [__vminu4](#group__cuda__math__intrinsic__simd_1gac5c280b9e46058d6519816f8ecba2ab7)(unsigned int a, unsigned int b)
    

Computes per-byte unsigned minimum.

__device__ unsigned int [__vneg2](#group__cuda__math__intrinsic__simd_1ga4bd9c377a7921ea666458f07d0f43c3d)(unsigned int a)
    

Computes per-halfword negation.

__device__ unsigned int [__vneg4](#group__cuda__math__intrinsic__simd_1gad95845c3d2fcea64714bea86379d55e4)(unsigned int a)
    

Performs per-byte negation.

__device__ unsigned int [__vnegss2](#group__cuda__math__intrinsic__simd_1ga3a118ee7ffe3715b52ee264b5b6f1b83)(unsigned int a)
    

Computes per-halfword negation with signed saturation.

__device__ unsigned int [__vnegss4](#group__cuda__math__intrinsic__simd_1gaac915af25c6db93b163e74a216ae6657)(unsigned int a)
    

Performs per-byte negation with signed saturation.

__device__ unsigned int [__vsads2](#group__cuda__math__intrinsic__simd_1ga6e6dbe721438c2c3ebc7ea237bc932db)(unsigned int a, unsigned int b)
    

Performs per-halfword sum of absolute difference of signed.

__device__ unsigned int [__vsads4](#group__cuda__math__intrinsic__simd_1gaca698d190c8d65d899ad86272ce973e1)(unsigned int a, unsigned int b)
    

Computes per-byte sum of abs difference of signed.

__device__ unsigned int [__vsadu2](#group__cuda__math__intrinsic__simd_1gaf85679aff1f2409690b1417bbe6fd0c9)(unsigned int a, unsigned int b)
    

Computes per-halfword sum of abs diff of unsigned.

__device__ unsigned int [__vsadu4](#group__cuda__math__intrinsic__simd_1ga8045c81708fa686e51661aa0a6125c4b)(unsigned int a, unsigned int b)
    

Computes per-byte sum of abs difference of unsigned.

__device__ unsigned int [__vseteq2](#group__cuda__math__intrinsic__simd_1ga2481597f8d86d411d61c233a222743d9)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed comparison: returns 1 if both parts compare equal.

__device__ unsigned int [__vseteq4](#group__cuda__math__intrinsic__simd_1gade3444d671cb10ef0cea1b2c2dad53d6)(unsigned int a, unsigned int b)
    

Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare equal.

__device__ unsigned int [__vsetges2](#group__cuda__math__intrinsic__simd_1ga8cbf8cba68ed549c291456121bf24e45)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: returns 1 if both parts compare greater than or equal.

__device__ unsigned int [__vsetges4](#group__cuda__math__intrinsic__simd_1ga918d94ef1c1021d7e3c15ef54754e6ea)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than or equal.

__device__ unsigned int [__vsetgeu2](#group__cuda__math__intrinsic__simd_1ga5ea027704cea5356b24a3f1dd885c9cf)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than or equal.

__device__ unsigned int [__vsetgeu4](#group__cuda__math__intrinsic__simd_1gadea75750ed4c28df52e58db4fa65f77f)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than or equal.

__device__ unsigned int [__vsetgts2](#group__cuda__math__intrinsic__simd_1ga0ba77cf0f9573009f2cf51ff74ac519a)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: returns 1 if both parts compare greater than.

__device__ unsigned int [__vsetgts4](#group__cuda__math__intrinsic__simd_1ga12e2228d0b19adbb85d2cfbe36273708)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than.

__device__ unsigned int [__vsetgtu2](#group__cuda__math__intrinsic__simd_1ga832e6dcc48c52e28a73c3d956d76e161)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than.

__device__ unsigned int [__vsetgtu4](#group__cuda__math__intrinsic__simd_1ga18ea8b5a279852513422f9a65c065a47)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than.

__device__ unsigned int [__vsetles2](#group__cuda__math__intrinsic__simd_1gafeb9bf3e45778d92d2def9bdbf5cbfaa)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: returns 1 if both parts compare less than or equal.

__device__ unsigned int [__vsetles4](#group__cuda__math__intrinsic__simd_1gaa2111e8384279c58921a9013ba698237)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than or equal.

__device__ unsigned int [__vsetleu2](#group__cuda__math__intrinsic__simd_1ga034d60c3b955af6f5c3bfd38fabcd64d)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: returns 1 if both parts compare less than or equal.

__device__ unsigned int [__vsetleu4](#group__cuda__math__intrinsic__simd_1ga078421b782b76ffabe1c8718899b6ccf)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than or equal.

__device__ unsigned int [__vsetlts2](#group__cuda__math__intrinsic__simd_1ga765d46f1039af0bd12fc69fc3898a610)(unsigned int a, unsigned int b)
    

Performs per-halfword signed comparison: returns 1 if both parts compare less than.

__device__ unsigned int [__vsetlts4](#group__cuda__math__intrinsic__simd_1ga94c54c7a8aa44471a2acceb9f1be0253)(unsigned int a, unsigned int b)
    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than.

__device__ unsigned int [__vsetltu2](#group__cuda__math__intrinsic__simd_1ga68f691e7e38122df60b5683228d1bf52)(unsigned int a, unsigned int b)
    

Performs per-halfword unsigned comparison: returns 1 if both parts compare less than.

__device__ unsigned int [__vsetltu4](#group__cuda__math__intrinsic__simd_1ga207a3283e538ea20ca23c8c96a22c607)(unsigned int a, unsigned int b)
    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than.

__device__ unsigned int [__vsetne2](#group__cuda__math__intrinsic__simd_1ga1db86324bd17ad1cf607e565d48cd445)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed comparison: returns 1 if both parts compare not equal.

__device__ unsigned int [__vsetne4](#group__cuda__math__intrinsic__simd_1ga4cff2947dee8d1670e8aecdd3c761280)(unsigned int a, unsigned int b)
    

Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare not equal.

__device__ unsigned int [__vsub2](#group__cuda__math__intrinsic__simd_1ga967e5249dcaa11d74765cfeb8e29ca5a)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed subtraction, with wrap-around: a - b.

__device__ unsigned int [__vsub4](#group__cuda__math__intrinsic__simd_1gaece9d5ae8f5a3f27141a3a814a40b5f3)(unsigned int a, unsigned int b)
    

Performs per-byte subtraction: a - b.

__device__ unsigned int [__vsubss2](#group__cuda__math__intrinsic__simd_1ga248ad036e46905b9b8872fbbc7fed781)(unsigned int a, unsigned int b)
    

Performs per-halfword (un)signed subtraction, with signed saturation: a - b.

__device__ unsigned int [__vsubss4](#group__cuda__math__intrinsic__simd_1ga95fc5c4e63356ee2ff56b0b8238f4382)(unsigned int a, unsigned int b)
    

Performs per-byte subtraction with signed saturation: a - b.

__device__ unsigned int [__vsubus2](#group__cuda__math__intrinsic__simd_1ga53bc67f19d835c021cda1ae75d700d28)(unsigned int a, unsigned int b)
    

Performs per-halfword subtraction with unsigned saturation: a - b.

__device__ unsigned int [__vsubus4](#group__cuda__math__intrinsic__simd_1ga50aaa3aafb256a1dc5dd5024b2d6a560)(unsigned int a, unsigned int b)
    

Performs per-byte subtraction with unsigned saturation: a - b.

##  14.1. Functions[](#functions "Permalink to this headline")

__device__ unsigned int __vabs2(unsigned int a)[](#_CPPv47__vabs2j "Permalink to this definition")  

    

Computes per-halfword absolute value: |a|. 

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes, then computes absolute value for each of parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabs4(unsigned int a)[](#_CPPv47__vabs4j "Permalink to this definition")  

    

Computes per-byte absolute value: |a|. 

Splits argument by bytes. Computes absolute value of each byte. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsdiffs2(unsigned int a, unsigned int b)[](#_CPPv412__vabsdiffs2jj "Permalink to this definition")  

    

Computes per-halfword absolute difference of signed integer: |a - b|. 

Splits 4 bytes of each into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsdiffs4(unsigned int a, unsigned int b)[](#_CPPv412__vabsdiffs4jj "Permalink to this definition")  

    

Computes per-byte absolute difference of signed integer: |a - b|. 

Splits 4 bytes of each into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsdiffu2(unsigned int a, unsigned int b)[](#_CPPv412__vabsdiffu2jj "Permalink to this definition")  

    

Computes per-halfword absolute difference of unsigned integer: |a - b|. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsdiffu4(unsigned int a, unsigned int b)[](#_CPPv412__vabsdiffu4jj "Permalink to this definition")  

    

Computes per-byte absolute difference of unsigned integer: |a - b|. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsss2(unsigned int a)[](#_CPPv49__vabsss2j "Permalink to this definition")  

    

Computes per-halfword absolute value with signed saturation: |a|. 

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes, then computes absolute value with signed saturation for each of parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vabsss4(unsigned int a)[](#_CPPv49__vabsss4j "Permalink to this definition")  

    

Computes per-byte absolute value with signed saturation: |a|. 

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte, then computes absolute value with signed saturation for each of parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vadd2(unsigned int a, unsigned int b)[](#_CPPv47__vadd2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed addition, with wrap-around: a + b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs unsigned addition on corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vadd4(unsigned int a, unsigned int b)[](#_CPPv47__vadd4jj "Permalink to this definition")  

    

Performs per-byte (un)signed addition: a + b. 

Splits ‘a’ into 4 bytes, then performs unsigned addition on each of these bytes with the corresponding byte from ‘b’, ignoring overflow. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vaddss2(unsigned int a, unsigned int b)[](#_CPPv49__vaddss2jj "Permalink to this definition")  

    

Performs per-halfword addition with signed saturation: a + b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs addition with signed saturation on corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vaddss4(unsigned int a, unsigned int b)[](#_CPPv49__vaddss4jj "Permalink to this definition")  

    

Performs per-byte addition with signed saturation: a + b. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte, then performs addition with signed saturation on corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vaddus2(unsigned int a, unsigned int b)[](#_CPPv49__vaddus2jj "Permalink to this definition")  

    

Performs per-halfword addition with unsigned saturation: a + b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then performs addition with unsigned saturation on corresponding parts. 

Returns
    

Returns computed value. 

__device__ unsigned int __vaddus4(unsigned int a, unsigned int b)[](#_CPPv49__vaddus4jj "Permalink to this definition")  

    

Performs per-byte addition with unsigned saturation: a + b. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte, then performs addition with unsigned saturation on corresponding parts. 

Returns
    

Returns computed value. 

__device__ unsigned int __vavgs2(unsigned int a, unsigned int b)[](#_CPPv48__vavgs2jj "Permalink to this definition")  

    

Performs per-halfword signed rounded average computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes signed rounded average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vavgs4(unsigned int a, unsigned int b)[](#_CPPv48__vavgs4jj "Permalink to this definition")  

    

Computes per-byte signed rounded average. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes signed rounded average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vavgu2(unsigned int a, unsigned int b)[](#_CPPv48__vavgu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned rounded average computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes unsigned rounded average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vavgu4(unsigned int a, unsigned int b)[](#_CPPv48__vavgu4jj "Permalink to this definition")  

    

Performs per-byte unsigned rounded average. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes unsigned rounded average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vcmpeq2(unsigned int a, unsigned int b)[](#_CPPv49__vcmpeq2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed comparison: a == b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if they are equal, and 0000 otherwise. For example __vcmpeq2(0x1234aba5, 0x1234aba6) returns 0xffff0000. 

Returns
    

Returns 0xffff computed value. 

__device__ unsigned int __vcmpeq4(unsigned int a, unsigned int b)[](#_CPPv49__vcmpeq4jj "Permalink to this definition")  

    

Performs per-byte (un)signed comparison: a == b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if they are equal, and 00 otherwise. For example __vcmpeq4(0x1234aba5, 0x1234aba6) returns 0xffffff00. 

Returns
    

Returns 0xff if a = b, else returns 0. 

__device__ unsigned int __vcmpges2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpges2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: a >= b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part >= ‘b’ part, and 0000 otherwise. For example __vcmpges2(0x1234aba5, 0x1234aba6) returns 0xffff0000. 

Returns
    

Returns 0xffff if a >= b, else returns 0. 

__device__ unsigned int __vcmpges4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpges4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: a >= b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part >= ‘b’ part, and 00 otherwise. For example __vcmpges4(0x1234aba5, 0x1234aba6) returns 0xffffff00. 

Returns
    

Returns 0xff if a >= b, else returns 0. 

__device__ unsigned int __vcmpgeu2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgeu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: a >= b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part >= ‘b’ part, and 0000 otherwise. For example __vcmpgeu2(0x1234aba5, 0x1234aba6) returns 0xffff0000. 

Returns
    

Returns 0xffff if a >= b, else returns 0. 

__device__ unsigned int __vcmpgeu4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgeu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: a >= b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part >= ‘b’ part, and 00 otherwise. For example __vcmpgeu4(0x1234aba5, 0x1234aba6) returns 0xffffff00. 

Returns
    

Returns 0xff if a >= b, else returns 0. 

__device__ unsigned int __vcmpgts2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgts2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: a > b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part > ‘b’ part, and 0000 otherwise. For example __vcmpgts2(0x1234aba5, 0x1234aba6) returns 0x00000000. 

Returns
    

Returns 0xffff if a > b, else returns 0. 

__device__ unsigned int __vcmpgts4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgts4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: a > b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part > ‘b’ part, and 00 otherwise. For example __vcmpgts4(0x1234aba5, 0x1234aba6) returns 0x00000000. 

Returns
    

Returns 0xff if a > b, else returns 0. 

__device__ unsigned int __vcmpgtu2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgtu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: a > b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part > ‘b’ part, and 0000 otherwise. For example __vcmpgtu2(0x1234aba5, 0x1234aba6) returns 0x00000000. 

Returns
    

Returns 0xffff if a > b, else returns 0. 

__device__ unsigned int __vcmpgtu4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpgtu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: a > b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part > ‘b’ part, and 00 otherwise. For example __vcmpgtu4(0x1234aba5, 0x1234aba6) returns 0x00000000. 

Returns
    

Returns 0xff if a > b, else returns 0. 

__device__ unsigned int __vcmples2(unsigned int a, unsigned int b)[](#_CPPv410__vcmples2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: a <= b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part <= ‘b’ part, and 0000 otherwise. For example __vcmples2(0x1234aba5, 0x1234aba6) returns 0xffffffff. 

Returns
    

Returns 0xffff if a <= b, else returns 0. 

__device__ unsigned int __vcmples4(unsigned int a, unsigned int b)[](#_CPPv410__vcmples4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: a <= b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part <= ‘b’ part, and 00 otherwise. For example __vcmples4(0x1234aba5, 0x1234aba6) returns 0xffffffff. 

Returns
    

Returns 0xff if a <= b, else returns 0. 

__device__ unsigned int __vcmpleu2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpleu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: a <= b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part <= ‘b’ part, and 0000 otherwise. For example __vcmpleu2(0x1234aba5, 0x1234aba6) returns 0xffffffff. 

Returns
    

Returns 0xffff if a <= b, else returns 0. 

__device__ unsigned int __vcmpleu4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpleu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: a <= b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part <= ‘b’ part, and 00 otherwise. For example __vcmpleu4(0x1234aba5, 0x1234aba6) returns 0xffffffff. 

Returns
    

Returns 0xff if a <= b, else returns 0. 

__device__ unsigned int __vcmplts2(unsigned int a, unsigned int b)[](#_CPPv410__vcmplts2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: a < b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part < ‘b’ part, and 0000 otherwise. For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff. 

Returns
    

Returns 0xffff if a < b, else returns 0. 

__device__ unsigned int __vcmplts4(unsigned int a, unsigned int b)[](#_CPPv410__vcmplts4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: a < b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part < ‘b’ part, and 00 otherwise. For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff. 

Returns
    

Returns 0xff if a < b, else returns 0. 

__device__ unsigned int __vcmpltu2(unsigned int a, unsigned int b)[](#_CPPv410__vcmpltu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: a < b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part < ‘b’ part, and 0000 otherwise. For example __vcmpltu2(0x1234aba5, 0x1234aba6) returns 0x0000ffff. 

Returns
    

Returns 0xffff if a < b, else returns 0. 

__device__ unsigned int __vcmpltu4(unsigned int a, unsigned int b)[](#_CPPv410__vcmpltu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: a < b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part < ‘b’ part, and 00 otherwise. For example __vcmpltu4(0x1234aba5, 0x1234aba6) returns 0x000000ff. 

Returns
    

Returns 0xff if a < b, else returns 0. 

__device__ unsigned int __vcmpne2(unsigned int a, unsigned int b)[](#_CPPv49__vcmpne2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed comparison: a != b ? 0xffff : 0. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts result is ffff if ‘a’ part != ‘b’ part, and 0000 otherwise. For example __vcmplts2(0x1234aba5, 0x1234aba6) returns 0x0000ffff. 

Returns
    

Returns 0xffff if a != b, else returns 0. 

__device__ unsigned int __vcmpne4(unsigned int a, unsigned int b)[](#_CPPv49__vcmpne4jj "Permalink to this definition")  

    

Performs per-byte (un)signed comparison: a != b ? 0xff : 0. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts result is ff if ‘a’ part != ‘b’ part, and 00 otherwise. For example __vcmplts4(0x1234aba5, 0x1234aba6) returns 0x000000ff. 

Returns
    

Returns 0xff if a != b, else returns 0. 

__device__ unsigned int __vhaddu2(unsigned int a, unsigned int b)[](#_CPPv49__vhaddu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned average computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes, then computes unsigned average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vhaddu4(unsigned int a, unsigned int b)[](#_CPPv49__vhaddu4jj "Permalink to this definition")  

    

Computes per-byte unsigned average. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. then computes unsigned average of corresponding parts. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmax_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv416__viaddmax_s16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword max(a + b, c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add and compare: max(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmax_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv421__viaddmax_s16x2_reluKjKjKj "Permalink to this definition")  

    

Performs per-halfword max(max(a + b, c), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add, followed by a max with relu: max(max(a_part + b_part), c_part), 0) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __viaddmax_s32(const int a, const int b, const int c)[](#_CPPv414__viaddmax_s32KiKiKi "Permalink to this definition")  

    

Computes max(a + b, c) 

Calculates the sum of signed integers `a` and `b` and takes the max with `c`. 

Returns
    

Returns computed value. 

__host__ __device__ int __viaddmax_s32_relu(const int a, const int b, const int c)[](#_CPPv419__viaddmax_s32_reluKiKiKi "Permalink to this definition")  

    

Computes max(max(a + b, c), 0) 

Calculates the sum of signed integers `a` and `b` and takes the max with `c`. If the result is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmax_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv416__viaddmax_u16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword max(a + b, c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs an add and compare: max(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmax_u32(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__viaddmax_u32KjKjKj "Permalink to this definition")  

    

Computes max(a + b, c) 

Calculates the sum of unsigned integers `a` and `b` and takes the max with `c`. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmin_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv416__viaddmin_s16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword min(a + b, c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add and compare: min(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmin_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv421__viaddmin_s16x2_reluKjKjKj "Permalink to this definition")  

    

Performs per-halfword max(min(a + b, c), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs an add, followed by a min with relu: max(min(a_part + b_part), c_part), 0) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __viaddmin_s32(const int a, const int b, const int c)[](#_CPPv414__viaddmin_s32KiKiKi "Permalink to this definition")  

    

Computes min(a + b, c) 

Calculates the sum of signed integers `a` and `b` and takes the min with `c`. 

Returns
    

Returns computed value. 

__host__ __device__ int __viaddmin_s32_relu(const int a, const int b, const int c)[](#_CPPv419__viaddmin_s32_reluKiKiKi "Permalink to this definition")  

    

Computes max(min(a + b, c), 0) 

Calculates the sum of signed integers `a` and `b` and takes the min with `c`. If the result is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmin_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv416__viaddmin_u16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword min(a + b, c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs an add and compare: min(a_part + b_part), c_part) Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __viaddmin_u32(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__viaddmin_u32KjKjKj "Permalink to this definition")  

    

Computes min(a + b, c) 

Calculates the sum of unsigned integers `a` and `b` and takes the min with `c`. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vibmax_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)[](#_CPPv414__vibmax_s16x2KjKjPCbPCb "Permalink to this definition")  

    

Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b). 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part >= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part >= b_low_part). 

Returns
    

Returns computed values. 

__host__ __device__ int __vibmax_s32(const int a, const int b, bool *const pred)[](#_CPPv412__vibmax_s32KiKiPCb "Permalink to this definition")  

    

Computes max(a, b), also sets the value pointed to by pred to (a >= b). 

Calculates the maximum of `a` and `b` of two signed ints. Also sets the value pointed to by `pred` to the value (a >= b). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vibmax_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)[](#_CPPv414__vibmax_u16x2KjKjPCbPCb "Permalink to this definition")  

    

Performs per-halfword max(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a >= b). 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part >= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part >= b_low_part). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vibmax_u32(const unsigned int a, const unsigned int b, bool *const pred)[](#_CPPv412__vibmax_u32KjKjPCb "Permalink to this definition")  

    

Computes max(a, b), also sets the value pointed to by pred to (a >= b). 

Calculates the maximum of `a` and `b` of two unsigned ints. Also sets the value pointed to by `pred` to the value (a >= b). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vibmin_s16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)[](#_CPPv414__vibmin_s16x2KjKjPCbPCb "Permalink to this definition")  

    

Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b). 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part <= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part <= b_low_part). 

Returns
    

Returns computed values. 

__host__ __device__ int __vibmin_s32(const int a, const int b, bool *const pred)[](#_CPPv412__vibmin_s32KiKiPCb "Permalink to this definition")  

    

Computes min(a, b), also sets the value pointed to by pred to (a <= b). 

Calculates the minimum of `a` and `b` of two signed ints. Also sets the value pointed to by `pred` to the value (a <= b). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vibmin_u16x2(const unsigned int a, const unsigned int b, bool *const pred_hi, bool *const pred_lo)[](#_CPPv414__vibmin_u16x2KjKjPCbPCb "Permalink to this definition")  

    

Performs per-halfword min(a, b), also sets the value pointed to by pred_hi and pred_lo to the per-halfword result of (a <= b). 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a maximum ( = max(a_part, b_part) ). Partial results are recombined and returned as unsigned int. Sets the value pointed to by `pred_hi` to the value (a_high_part <= b_high_part). Sets the value pointed to by `pred_lo` to the value (a_low_part <= b_low_part). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vibmin_u32(const unsigned int a, const unsigned int b, bool *const pred)[](#_CPPv412__vibmin_u32KjKjPCb "Permalink to this definition")  

    

Computes min(a, b), also sets the value pointed to by pred to (a <= b). 

Calculates the minimum of `a` and `b` of two unsigned ints. Also sets the value pointed to by `pred` to the value (a <= b). 

Returns
    

Returns computed values. 

__host__ __device__ unsigned int __vimax3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__vimax3_s16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword max(max(a, b), c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a 3-way max ( = max(max(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimax3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv419__vimax3_s16x2_reluKjKjKj "Permalink to this definition")  

    

Performs per-halfword max(max(max(a, b), c), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a three-way max with relu ( = max(a_part, b_part, c_part, 0) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimax3_s32(const int a, const int b, const int c)[](#_CPPv412__vimax3_s32KiKiKi "Permalink to this definition")  

    

Computes max(max(a, b), c) 

Calculates the 3-way max of signed integers `a`, `b` and `c`. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimax3_s32_relu(const int a, const int b, const int c)[](#_CPPv417__vimax3_s32_reluKiKiKi "Permalink to this definition")  

    

Computes max(max(max(a, b), c), 0) 

Calculates the maximum of three signed ints, if this is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimax3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__vimax3_u16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword max(max(a, b), c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a 3-way max ( = max(max(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimax3_u32(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv412__vimax3_u32KjKjKj "Permalink to this definition")  

    

Computes max(max(a, b), c) 

Calculates the 3-way max of unsigned integers `a`, `b` and `c`. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimax_s16x2_relu(const unsigned int a, const unsigned int b)[](#_CPPv418__vimax_s16x2_reluKjKj "Permalink to this definition")  

    

Performs per-halfword max(max(a, b), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a max with relu ( = max(a_part, b_part, 0) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimax_s32_relu(const int a, const int b)[](#_CPPv416__vimax_s32_reluKiKi "Permalink to this definition")  

    

Computes max(max(a, b), 0) 

Calculates the maximum of `a` and `b` of two signed ints, if this is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimin3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__vimin3_s16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword min(min(a, b), c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a 3-way min ( = min(min(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimin3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv419__vimin3_s16x2_reluKjKjKj "Permalink to this definition")  

    

Performs per-halfword max(min(min(a, b), c), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a three-way min with relu ( = max(min(a_part, b_part, c_part), 0) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimin3_s32(const int a, const int b, const int c)[](#_CPPv412__vimin3_s32KiKiKi "Permalink to this definition")  

    

Computes min(min(a, b), c) 

Calculates the 3-way min of signed integers `a`, `b` and `c`. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimin3_s32_relu(const int a, const int b, const int c)[](#_CPPv417__vimin3_s32_reluKiKiKi "Permalink to this definition")  

    

Computes max(min(min(a, b), c), 0) 

Calculates the minimum of three signed ints, if this is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimin3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv414__vimin3_u16x2KjKjKj "Permalink to this definition")  

    

Performs per-halfword min(min(a, b), c) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as unsigned shorts. For corresponding parts function performs a 3-way min ( = min(min(a_part, b_part), c_part) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimin3_u32(const unsigned int a, const unsigned int b, const unsigned int c)[](#_CPPv412__vimin3_u32KjKjKj "Permalink to this definition")  

    

Computes min(min(a, b), c) 

Calculates the 3-way min of unsigned integers `a`, `b` and `c`. 

Returns
    

Returns computed value. 

__host__ __device__ unsigned int __vimin_s16x2_relu(const unsigned int a, const unsigned int b)[](#_CPPv418__vimin_s16x2_reluKjKj "Permalink to this definition")  

    

Performs per-halfword max(min(a, b), 0) 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. These 2 byte parts are interpreted as signed shorts. For corresponding parts function performs a min with relu ( = max(min(a_part, b_part), 0) ). Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__host__ __device__ int __vimin_s32_relu(const int a, const int b)[](#_CPPv416__vimin_s32_reluKiKi "Permalink to this definition")  

    

Computes max(min(a, b), 0) 

Calculates the minimum of `a` and `b` of two signed ints, if this is less than `0` then `0` is returned. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmaxs2(unsigned int a, unsigned int b)[](#_CPPv48__vmaxs2jj "Permalink to this definition")  

    

Performs per-halfword signed maximum computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes signed maximum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmaxs4(unsigned int a, unsigned int b)[](#_CPPv48__vmaxs4jj "Permalink to this definition")  

    

Computes per-byte signed maximum. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes signed maximum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmaxu2(unsigned int a, unsigned int b)[](#_CPPv48__vmaxu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned maximum computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes unsigned maximum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmaxu4(unsigned int a, unsigned int b)[](#_CPPv48__vmaxu4jj "Permalink to this definition")  

    

Computes per-byte unsigned maximum. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes unsigned maximum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmins2(unsigned int a, unsigned int b)[](#_CPPv48__vmins2jj "Permalink to this definition")  

    

Performs per-halfword signed minimum computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes signed minimum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vmins4(unsigned int a, unsigned int b)[](#_CPPv48__vmins4jj "Permalink to this definition")  

    

Computes per-byte signed minimum. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes signed minimum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vminu2(unsigned int a, unsigned int b)[](#_CPPv48__vminu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned minimum computation. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes unsigned minimum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vminu4(unsigned int a, unsigned int b)[](#_CPPv48__vminu4jj "Permalink to this definition")  

    

Computes per-byte unsigned minimum. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes unsigned minimum. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vneg2(unsigned int a)[](#_CPPv47__vneg2j "Permalink to this definition")  

    

Computes per-halfword negation. 

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes. For each part function computes negation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vneg4(unsigned int a)[](#_CPPv47__vneg4j "Permalink to this definition")  

    

Performs per-byte negation. 

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte. For each part function computes negation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vnegss2(unsigned int a)[](#_CPPv49__vnegss2j "Permalink to this definition")  

    

Computes per-halfword negation with signed saturation. 

Splits 4 bytes of argument into 2 parts, each consisting of 2 bytes. For each part function computes negation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vnegss4(unsigned int a)[](#_CPPv49__vnegss4j "Permalink to this definition")  

    

Performs per-byte negation with signed saturation. 

Splits 4 bytes of argument into 4 parts, each consisting of 1 byte. For each part function computes negation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsads2(unsigned int a, unsigned int b)[](#_CPPv48__vsads2jj "Permalink to this definition")  

    

Performs per-halfword sum of absolute difference of signed. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute difference and sum it up. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsads4(unsigned int a, unsigned int b)[](#_CPPv48__vsads4jj "Permalink to this definition")  

    

Computes per-byte sum of abs difference of signed. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute difference and sum it up. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsadu2(unsigned int a, unsigned int b)[](#_CPPv48__vsadu2jj "Permalink to this definition")  

    

Computes per-halfword sum of abs diff of unsigned. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function computes absolute differences and returns sum of those differences. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsadu4(unsigned int a, unsigned int b)[](#_CPPv48__vsadu4jj "Permalink to this definition")  

    

Computes per-byte sum of abs difference of unsigned. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function computes absolute differences and returns sum of those differences. 

Returns
    

Returns computed value. 

__device__ unsigned int __vseteq2(unsigned int a, unsigned int b)[](#_CPPv49__vseteq2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed comparison: returns 1 if both parts compare equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part == ‘b’ part. If both equalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a = b, else returns 0. 

__device__ unsigned int __vseteq4(unsigned int a, unsigned int b)[](#_CPPv49__vseteq4jj "Permalink to this definition")  

    

Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare equal. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part == ‘b’ part. If all 4 equalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a = b, else returns 0. 

__device__ unsigned int __vsetges2(unsigned int a, unsigned int b)[](#_CPPv410__vsetges2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: returns 1 if both parts compare greater than or equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a >= b, else returns 0. 

__device__ unsigned int __vsetges4(unsigned int a, unsigned int b)[](#_CPPv410__vsetges4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than or equal. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a >= b, else returns 0. 

__device__ unsigned int __vsetgeu2(unsigned int a, unsigned int b)[](#_CPPv410__vsetgeu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than or equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a >= b, else returns 0. 

__device__ unsigned int __vsetgeu4(unsigned int a, unsigned int b)[](#_CPPv410__vsetgeu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than or equal. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part >= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a >= b, else returns 0. 

__device__ unsigned int __vsetgts2(unsigned int a, unsigned int b)[](#_CPPv410__vsetgts2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: returns 1 if both parts compare greater than. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a > b, else returns 0. 

__device__ unsigned int __vsetgts4(unsigned int a, unsigned int b)[](#_CPPv410__vsetgts4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare greater than. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a > b, else returns 0. 

__device__ unsigned int __vsetgtu2(unsigned int a, unsigned int b)[](#_CPPv410__vsetgtu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: returns 1 if both parts compare greater than. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a > b, else returns 0. 

__device__ unsigned int __vsetgtu4(unsigned int a, unsigned int b)[](#_CPPv410__vsetgtu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare greater than. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part > ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a > b, else returns 0. 

__device__ unsigned int __vsetles2(unsigned int a, unsigned int b)[](#_CPPv410__vsetles2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: returns 1 if both parts compare less than or equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a <= b, else returns 0. 

__device__ unsigned int __vsetles4(unsigned int a, unsigned int b)[](#_CPPv410__vsetles4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than or equal. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a <= b, else returns 0. 

__device__ unsigned int __vsetleu2(unsigned int a, unsigned int b)[](#_CPPv410__vsetleu2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: returns 1 if both parts compare less than or equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a <= b, else returns 0. 

__device__ unsigned int __vsetleu4(unsigned int a, unsigned int b)[](#_CPPv410__vsetleu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than or equal. 

Splits 4 bytes of each argument into 4 part, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a <= b, else returns 0. 

__device__ unsigned int __vsetlts2(unsigned int a, unsigned int b)[](#_CPPv410__vsetlts2jj "Permalink to this definition")  

    

Performs per-halfword signed comparison: returns 1 if both parts compare less than. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a < b, else returns 0. 

__device__ unsigned int __vsetlts4(unsigned int a, unsigned int b)[](#_CPPv410__vsetlts4jj "Permalink to this definition")  

    

Performs per-byte signed comparison: returns 1 if all 4 pairs compare less than. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a < b, else returns 0. 

__device__ unsigned int __vsetltu2(unsigned int a, unsigned int b)[](#_CPPv410__vsetltu2jj "Permalink to this definition")  

    

Performs per-halfword unsigned comparison: returns 1 if both parts compare less than. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If both inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a < b, else returns 0. 

__device__ unsigned int __vsetltu4(unsigned int a, unsigned int b)[](#_CPPv410__vsetltu4jj "Permalink to this definition")  

    

Performs per-byte unsigned comparison: returns 1 if all 4 pairs compare less than. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part <= ‘b’ part. If all 4 inequalities are satisfied, function returns 1. 

Returns
    

Returns 1 if a < b, else returns 0. 

__device__ unsigned int __vsetne2(unsigned int a, unsigned int b)[](#_CPPv49__vsetne2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed comparison: returns 1 if both parts compare not equal. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs comparison ‘a’ part != ‘b’ part. If both conditions are satisfied, function returns 1. 

Returns
    

Returns 1 if a != b, else returns 0. 

__device__ unsigned int __vsetne4(unsigned int a, unsigned int b)[](#_CPPv49__vsetne4jj "Permalink to this definition")  

    

Performs per-byte (un)signed comparison: returns 1 if all 4 pairs compare not equal. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs comparison ‘a’ part != ‘b’ part. If all 4 conditions are satisfied, function returns 1. 

Returns
    

Returns 1 if a != b, else returns 0. 

__device__ unsigned int __vsub2(unsigned int a, unsigned int b)[](#_CPPv47__vsub2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed subtraction, with wrap-around: a - b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsub4(unsigned int a, unsigned int b)[](#_CPPv47__vsub4jj "Permalink to this definition")  

    

Performs per-byte subtraction: a - b. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsubss2(unsigned int a, unsigned int b)[](#_CPPv49__vsubss2jj "Permalink to this definition")  

    

Performs per-halfword (un)signed subtraction, with signed saturation: a - b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction with signed saturation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsubss4(unsigned int a, unsigned int b)[](#_CPPv49__vsubss4jj "Permalink to this definition")  

    

Performs per-byte subtraction with signed saturation: a - b. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction with signed saturation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsubus2(unsigned int a, unsigned int b)[](#_CPPv49__vsubus2jj "Permalink to this definition")  

    

Performs per-halfword subtraction with unsigned saturation: a - b. 

Splits 4 bytes of each argument into 2 parts, each consisting of 2 bytes. For corresponding parts function performs subtraction with unsigned saturation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value. 

__device__ unsigned int __vsubus4(unsigned int a, unsigned int b)[](#_CPPv49__vsubus4jj "Permalink to this definition")  

    

Performs per-byte subtraction with unsigned saturation: a - b. 

Splits 4 bytes of each argument into 4 parts, each consisting of 1 byte. For corresponding parts function performs subtraction with unsigned saturation. Partial results are recombined and returned as unsigned int. 

Returns
    

Returns computed value.