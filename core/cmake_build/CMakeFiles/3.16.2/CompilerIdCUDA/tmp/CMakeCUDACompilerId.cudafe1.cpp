# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

# 1
# 61 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
#pragma GCC diagnostic push
# 64
#pragma GCC diagnostic ignored "-Wunused-function"
# 66 "/usr/local/cuda/bin/..//include/device_types.h"
#if 0
# 66
enum cudaRoundMode { 
# 68
cudaRoundNearest, 
# 69
cudaRoundZero, 
# 70
cudaRoundPosInf, 
# 71
cudaRoundMinInf
# 72
}; 
#endif
# 98 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 98
struct char1 { 
# 100
signed char x; 
# 101
}; 
#endif
# 103 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 103
struct uchar1 { 
# 105
unsigned char x; 
# 106
}; 
#endif
# 109 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 109
struct __attribute((aligned(2))) char2 { 
# 111
signed char x, y; 
# 112
}; 
#endif
# 114 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 114
struct __attribute((aligned(2))) uchar2 { 
# 116
unsigned char x, y; 
# 117
}; 
#endif
# 119 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 119
struct char3 { 
# 121
signed char x, y, z; 
# 122
}; 
#endif
# 124 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 124
struct uchar3 { 
# 126
unsigned char x, y, z; 
# 127
}; 
#endif
# 129 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 129
struct __attribute((aligned(4))) char4 { 
# 131
signed char x, y, z, w; 
# 132
}; 
#endif
# 134 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 134
struct __attribute((aligned(4))) uchar4 { 
# 136
unsigned char x, y, z, w; 
# 137
}; 
#endif
# 139 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 139
struct short1 { 
# 141
short x; 
# 142
}; 
#endif
# 144 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 144
struct ushort1 { 
# 146
unsigned short x; 
# 147
}; 
#endif
# 149 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 149
struct __attribute((aligned(4))) short2 { 
# 151
short x, y; 
# 152
}; 
#endif
# 154 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 154
struct __attribute((aligned(4))) ushort2 { 
# 156
unsigned short x, y; 
# 157
}; 
#endif
# 159 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 159
struct short3 { 
# 161
short x, y, z; 
# 162
}; 
#endif
# 164 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 164
struct ushort3 { 
# 166
unsigned short x, y, z; 
# 167
}; 
#endif
# 169 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 169
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 170 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 170
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 172 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 172
struct int1 { 
# 174
int x; 
# 175
}; 
#endif
# 177 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 177
struct uint1 { 
# 179
unsigned x; 
# 180
}; 
#endif
# 182 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 182
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 183 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 183
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 185 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 185
struct int3 { 
# 187
int x, y, z; 
# 188
}; 
#endif
# 190 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 190
struct uint3 { 
# 192
unsigned x, y, z; 
# 193
}; 
#endif
# 195 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 195
struct __attribute((aligned(16))) int4 { 
# 197
int x, y, z, w; 
# 198
}; 
#endif
# 200 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 200
struct __attribute((aligned(16))) uint4 { 
# 202
unsigned x, y, z, w; 
# 203
}; 
#endif
# 205 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 205
struct long1 { 
# 207
long x; 
# 208
}; 
#endif
# 210 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 210
struct ulong1 { 
# 212
unsigned long x; 
# 213
}; 
#endif
# 220 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 220
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 222
long x, y; 
# 223
}; 
#endif
# 225 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 225
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 227
unsigned long x, y; 
# 228
}; 
#endif
# 232 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 232
struct long3 { 
# 234
long x, y, z; 
# 235
}; 
#endif
# 237 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 237
struct ulong3 { 
# 239
unsigned long x, y, z; 
# 240
}; 
#endif
# 242 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 242
struct __attribute((aligned(16))) long4 { 
# 244
long x, y, z, w; 
# 245
}; 
#endif
# 247 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 247
struct __attribute((aligned(16))) ulong4 { 
# 249
unsigned long x, y, z, w; 
# 250
}; 
#endif
# 252 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 252
struct float1 { 
# 254
float x; 
# 255
}; 
#endif
# 274 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 274
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 279 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 279
struct float3 { 
# 281
float x, y, z; 
# 282
}; 
#endif
# 284 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 284
struct __attribute((aligned(16))) float4 { 
# 286
float x, y, z, w; 
# 287
}; 
#endif
# 289 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 289
struct longlong1 { 
# 291
long long x; 
# 292
}; 
#endif
# 294 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 294
struct ulonglong1 { 
# 296
unsigned long long x; 
# 297
}; 
#endif
# 299 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 299
struct __attribute((aligned(16))) longlong2 { 
# 301
long long x, y; 
# 302
}; 
#endif
# 304 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 304
struct __attribute((aligned(16))) ulonglong2 { 
# 306
unsigned long long x, y; 
# 307
}; 
#endif
# 309 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 309
struct longlong3 { 
# 311
long long x, y, z; 
# 312
}; 
#endif
# 314 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 314
struct ulonglong3 { 
# 316
unsigned long long x, y, z; 
# 317
}; 
#endif
# 319 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 319
struct __attribute((aligned(16))) longlong4 { 
# 321
long long x, y, z, w; 
# 322
}; 
#endif
# 324 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 324
struct __attribute((aligned(16))) ulonglong4 { 
# 326
unsigned long long x, y, z, w; 
# 327
}; 
#endif
# 329 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 329
struct double1 { 
# 331
double x; 
# 332
}; 
#endif
# 334 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 334
struct __attribute((aligned(16))) double2 { 
# 336
double x, y; 
# 337
}; 
#endif
# 339 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 339
struct double3 { 
# 341
double x, y, z; 
# 342
}; 
#endif
# 344 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 344
struct __attribute((aligned(16))) double4 { 
# 346
double x, y, z, w; 
# 347
}; 
#endif
# 361 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef char1 
# 361
char1; 
#endif
# 362 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uchar1 
# 362
uchar1; 
#endif
# 363 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef char2 
# 363
char2; 
#endif
# 364 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uchar2 
# 364
uchar2; 
#endif
# 365 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef char3 
# 365
char3; 
#endif
# 366 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uchar3 
# 366
uchar3; 
#endif
# 367 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef char4 
# 367
char4; 
#endif
# 368 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uchar4 
# 368
uchar4; 
#endif
# 369 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef short1 
# 369
short1; 
#endif
# 370 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ushort1 
# 370
ushort1; 
#endif
# 371 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef short2 
# 371
short2; 
#endif
# 372 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ushort2 
# 372
ushort2; 
#endif
# 373 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef short3 
# 373
short3; 
#endif
# 374 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ushort3 
# 374
ushort3; 
#endif
# 375 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef short4 
# 375
short4; 
#endif
# 376 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ushort4 
# 376
ushort4; 
#endif
# 377 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef int1 
# 377
int1; 
#endif
# 378 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uint1 
# 378
uint1; 
#endif
# 379 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef int2 
# 379
int2; 
#endif
# 380 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uint2 
# 380
uint2; 
#endif
# 381 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef int3 
# 381
int3; 
#endif
# 382 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uint3 
# 382
uint3; 
#endif
# 383 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef int4 
# 383
int4; 
#endif
# 384 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef uint4 
# 384
uint4; 
#endif
# 385 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef long1 
# 385
long1; 
#endif
# 386 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulong1 
# 386
ulong1; 
#endif
# 387 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef long2 
# 387
long2; 
#endif
# 388 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulong2 
# 388
ulong2; 
#endif
# 389 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef long3 
# 389
long3; 
#endif
# 390 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulong3 
# 390
ulong3; 
#endif
# 391 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef long4 
# 391
long4; 
#endif
# 392 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulong4 
# 392
ulong4; 
#endif
# 393 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef float1 
# 393
float1; 
#endif
# 394 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef float2 
# 394
float2; 
#endif
# 395 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef float3 
# 395
float3; 
#endif
# 396 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef float4 
# 396
float4; 
#endif
# 397 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef longlong1 
# 397
longlong1; 
#endif
# 398 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulonglong1 
# 398
ulonglong1; 
#endif
# 399 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef longlong2 
# 399
longlong2; 
#endif
# 400 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulonglong2 
# 400
ulonglong2; 
#endif
# 401 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef longlong3 
# 401
longlong3; 
#endif
# 402 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulonglong3 
# 402
ulonglong3; 
#endif
# 403 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef longlong4 
# 403
longlong4; 
#endif
# 404 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef ulonglong4 
# 404
ulonglong4; 
#endif
# 405 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef double1 
# 405
double1; 
#endif
# 406 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef double2 
# 406
double2; 
#endif
# 407 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef double3 
# 407
double3; 
#endif
# 408 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef double4 
# 408
double4; 
#endif
# 416 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
# 416
struct dim3 { 
# 418
unsigned x, y, z; 
# 424
}; 
#endif
# 426 "/usr/local/cuda/bin/..//include/vector_types.h"
#if 0
typedef dim3 
# 426
dim3; 
#endif
# 149 "/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h" 3
typedef long ptrdiff_t; 
# 216 "/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h" 3
typedef unsigned long size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
# 189 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 189
enum cudaError { 
# 196
cudaSuccess, 
# 202
cudaErrorMissingConfiguration, 
# 208
cudaErrorMemoryAllocation, 
# 214
cudaErrorInitializationError, 
# 223 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorLaunchFailure, 
# 232 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorPriorLaunchFailure, 
# 243 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorLaunchTimeout, 
# 252 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorLaunchOutOfResources, 
# 258
cudaErrorInvalidDeviceFunction, 
# 267 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorInvalidConfiguration, 
# 273
cudaErrorInvalidDevice, 
# 279
cudaErrorInvalidValue, 
# 285
cudaErrorInvalidPitchValue, 
# 291
cudaErrorInvalidSymbol, 
# 296
cudaErrorMapBufferObjectFailed, 
# 301
cudaErrorUnmapBufferObjectFailed, 
# 307
cudaErrorInvalidHostPointer, 
# 313
cudaErrorInvalidDevicePointer, 
# 319
cudaErrorInvalidTexture, 
# 325
cudaErrorInvalidTextureBinding, 
# 332
cudaErrorInvalidChannelDescriptor, 
# 338
cudaErrorInvalidMemcpyDirection, 
# 348 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorAddressOfConstant, 
# 357 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorTextureFetchFailed, 
# 366 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorTextureNotBound, 
# 375 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorSynchronizationError, 
# 381
cudaErrorInvalidFilterSetting, 
# 387
cudaErrorInvalidNormSetting, 
# 395
cudaErrorMixedDeviceExecution, 
# 402
cudaErrorCudartUnloading, 
# 407
cudaErrorUnknown, 
# 415
cudaErrorNotYetImplemented, 
# 424 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorMemoryValueTooLarge, 
# 431
cudaErrorInvalidResourceHandle, 
# 439
cudaErrorNotReady, 
# 446
cudaErrorInsufficientDriver, 
# 459 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorSetOnActiveProcess, 
# 465
cudaErrorInvalidSurface, 
# 471
cudaErrorNoDevice, 
# 477
cudaErrorECCUncorrectable, 
# 482
cudaErrorSharedObjectSymbolNotFound, 
# 487
cudaErrorSharedObjectInitFailed, 
# 493
cudaErrorUnsupportedLimit, 
# 499
cudaErrorDuplicateVariableName, 
# 505
cudaErrorDuplicateTextureName, 
# 511
cudaErrorDuplicateSurfaceName, 
# 521 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorDevicesUnavailable, 
# 526
cudaErrorInvalidKernelImage, 
# 534
cudaErrorNoKernelImageForDevice, 
# 547 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorIncompatibleDriverContext, 
# 554
cudaErrorPeerAccessAlreadyEnabled, 
# 561
cudaErrorPeerAccessNotEnabled, 
# 567
cudaErrorDeviceAlreadyInUse = 54, 
# 574
cudaErrorProfilerDisabled, 
# 582
cudaErrorProfilerNotInitialized, 
# 589
cudaErrorProfilerAlreadyStarted, 
# 596
cudaErrorProfilerAlreadyStopped, 
# 603
cudaErrorAssert, 
# 610
cudaErrorTooManyPeers, 
# 616
cudaErrorHostMemoryAlreadyRegistered, 
# 622
cudaErrorHostMemoryNotRegistered, 
# 627
cudaErrorOperatingSystem, 
# 633
cudaErrorPeerAccessUnsupported, 
# 640
cudaErrorLaunchMaxDepthExceeded, 
# 648
cudaErrorLaunchFileScopedTex, 
# 656
cudaErrorLaunchFileScopedSurf, 
# 671 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorSyncDepthExceeded, 
# 683 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorLaunchPendingCountExceeded, 
# 688
cudaErrorNotPermitted, 
# 694
cudaErrorNotSupported, 
# 703 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorHardwareStackError, 
# 711
cudaErrorIllegalInstruction, 
# 720 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorMisalignedAddress, 
# 731 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorInvalidAddressSpace, 
# 739
cudaErrorInvalidPc, 
# 747
cudaErrorIllegalAddress, 
# 753
cudaErrorInvalidPtx, 
# 758
cudaErrorInvalidGraphicsContext, 
# 764
cudaErrorNvlinkUncorrectable, 
# 771
cudaErrorJitCompilerNotFound, 
# 780 "/usr/local/cuda/bin/..//include/driver_types.h"
cudaErrorCooperativeLaunchTooLarge, 
# 787
cudaErrorSystemNotReady, 
# 793
cudaErrorIllegalState, 
# 798
cudaErrorStartupFailure = 127, 
# 803
cudaErrorStreamCaptureUnsupported = 900, 
# 809
cudaErrorStreamCaptureInvalidated, 
# 815
cudaErrorStreamCaptureMerge, 
# 820
cudaErrorStreamCaptureUnmatched, 
# 826
cudaErrorStreamCaptureUnjoined, 
# 833
cudaErrorStreamCaptureIsolation, 
# 839
cudaErrorStreamCaptureImplicit, 
# 845
cudaErrorCapturedEvent, 
# 853
cudaErrorApiFailureBase = 10000
# 854
}; 
#endif
# 859 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 859
enum cudaChannelFormatKind { 
# 861
cudaChannelFormatKindSigned, 
# 862
cudaChannelFormatKindUnsigned, 
# 863
cudaChannelFormatKindFloat, 
# 864
cudaChannelFormatKindNone
# 865
}; 
#endif
# 870 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 870
struct cudaChannelFormatDesc { 
# 872
int x; 
# 873
int y; 
# 874
int z; 
# 875
int w; 
# 876
cudaChannelFormatKind f; 
# 877
}; 
#endif
# 882 "/usr/local/cuda/bin/..//include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 887
typedef const cudaArray *cudaArray_const_t; 
# 889
struct cudaArray; 
# 894
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 899
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 901
struct cudaMipmappedArray; 
# 906
#if 0
# 906
enum cudaMemoryType { 
# 908
cudaMemoryTypeUnregistered, 
# 909
cudaMemoryTypeHost, 
# 910
cudaMemoryTypeDevice, 
# 911
cudaMemoryTypeManaged
# 912
}; 
#endif
# 917 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 917
enum cudaMemcpyKind { 
# 919
cudaMemcpyHostToHost, 
# 920
cudaMemcpyHostToDevice, 
# 921
cudaMemcpyDeviceToHost, 
# 922
cudaMemcpyDeviceToDevice, 
# 923
cudaMemcpyDefault
# 924
}; 
#endif
# 931 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 931
struct cudaPitchedPtr { 
# 933
void *ptr; 
# 934
size_t pitch; 
# 935
size_t xsize; 
# 936
size_t ysize; 
# 937
}; 
#endif
# 944 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 944
struct cudaExtent { 
# 946
size_t width; 
# 947
size_t height; 
# 948
size_t depth; 
# 949
}; 
#endif
# 956 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 956
struct cudaPos { 
# 958
size_t x; 
# 959
size_t y; 
# 960
size_t z; 
# 961
}; 
#endif
# 966 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 966
struct cudaMemcpy3DParms { 
# 968
cudaArray_t srcArray; 
# 969
cudaPos srcPos; 
# 970
cudaPitchedPtr srcPtr; 
# 972
cudaArray_t dstArray; 
# 973
cudaPos dstPos; 
# 974
cudaPitchedPtr dstPtr; 
# 976
cudaExtent extent; 
# 977
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 978
}; 
#endif
# 983 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 983
struct cudaMemcpy3DPeerParms { 
# 985
cudaArray_t srcArray; 
# 986
cudaPos srcPos; 
# 987
cudaPitchedPtr srcPtr; 
# 988
int srcDevice; 
# 990
cudaArray_t dstArray; 
# 991
cudaPos dstPos; 
# 992
cudaPitchedPtr dstPtr; 
# 993
int dstDevice; 
# 995
cudaExtent extent; 
# 996
}; 
#endif
# 1001 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1001
struct cudaMemsetParams { 
# 1002
void *dst; 
# 1003
size_t pitch; 
# 1004
unsigned value; 
# 1005
unsigned elementSize; 
# 1006
size_t width; 
# 1007
size_t height; 
# 1008
}; 
#endif
# 1020 "/usr/local/cuda/bin/..//include/driver_types.h"
typedef void (*cudaHostFn_t)(void * userData); 
# 1025
#if 0
# 1025
struct cudaHostNodeParams { 
# 1026
cudaHostFn_t fn; 
# 1027
void *userData; 
# 1028
}; 
#endif
# 1033 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1033
enum cudaStreamCaptureStatus { 
# 1034
cudaStreamCaptureStatusNone, 
# 1035
cudaStreamCaptureStatusActive, 
# 1036
cudaStreamCaptureStatusInvalidated
# 1038
}; 
#endif
# 1043 "/usr/local/cuda/bin/..//include/driver_types.h"
struct cudaGraphicsResource; 
# 1048
#if 0
# 1048
enum cudaGraphicsRegisterFlags { 
# 1050
cudaGraphicsRegisterFlagsNone, 
# 1051
cudaGraphicsRegisterFlagsReadOnly, 
# 1052
cudaGraphicsRegisterFlagsWriteDiscard, 
# 1053
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 1054
cudaGraphicsRegisterFlagsTextureGather = 8
# 1055
}; 
#endif
# 1060 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1060
enum cudaGraphicsMapFlags { 
# 1062
cudaGraphicsMapFlagsNone, 
# 1063
cudaGraphicsMapFlagsReadOnly, 
# 1064
cudaGraphicsMapFlagsWriteDiscard
# 1065
}; 
#endif
# 1070 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1070
enum cudaGraphicsCubeFace { 
# 1072
cudaGraphicsCubeFacePositiveX, 
# 1073
cudaGraphicsCubeFaceNegativeX, 
# 1074
cudaGraphicsCubeFacePositiveY, 
# 1075
cudaGraphicsCubeFaceNegativeY, 
# 1076
cudaGraphicsCubeFacePositiveZ, 
# 1077
cudaGraphicsCubeFaceNegativeZ
# 1078
}; 
#endif
# 1083 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1083
enum cudaResourceType { 
# 1085
cudaResourceTypeArray, 
# 1086
cudaResourceTypeMipmappedArray, 
# 1087
cudaResourceTypeLinear, 
# 1088
cudaResourceTypePitch2D
# 1089
}; 
#endif
# 1094 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1094
enum cudaResourceViewFormat { 
# 1096
cudaResViewFormatNone, 
# 1097
cudaResViewFormatUnsignedChar1, 
# 1098
cudaResViewFormatUnsignedChar2, 
# 1099
cudaResViewFormatUnsignedChar4, 
# 1100
cudaResViewFormatSignedChar1, 
# 1101
cudaResViewFormatSignedChar2, 
# 1102
cudaResViewFormatSignedChar4, 
# 1103
cudaResViewFormatUnsignedShort1, 
# 1104
cudaResViewFormatUnsignedShort2, 
# 1105
cudaResViewFormatUnsignedShort4, 
# 1106
cudaResViewFormatSignedShort1, 
# 1107
cudaResViewFormatSignedShort2, 
# 1108
cudaResViewFormatSignedShort4, 
# 1109
cudaResViewFormatUnsignedInt1, 
# 1110
cudaResViewFormatUnsignedInt2, 
# 1111
cudaResViewFormatUnsignedInt4, 
# 1112
cudaResViewFormatSignedInt1, 
# 1113
cudaResViewFormatSignedInt2, 
# 1114
cudaResViewFormatSignedInt4, 
# 1115
cudaResViewFormatHalf1, 
# 1116
cudaResViewFormatHalf2, 
# 1117
cudaResViewFormatHalf4, 
# 1118
cudaResViewFormatFloat1, 
# 1119
cudaResViewFormatFloat2, 
# 1120
cudaResViewFormatFloat4, 
# 1121
cudaResViewFormatUnsignedBlockCompressed1, 
# 1122
cudaResViewFormatUnsignedBlockCompressed2, 
# 1123
cudaResViewFormatUnsignedBlockCompressed3, 
# 1124
cudaResViewFormatUnsignedBlockCompressed4, 
# 1125
cudaResViewFormatSignedBlockCompressed4, 
# 1126
cudaResViewFormatUnsignedBlockCompressed5, 
# 1127
cudaResViewFormatSignedBlockCompressed5, 
# 1128
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1129
cudaResViewFormatSignedBlockCompressed6H, 
# 1130
cudaResViewFormatUnsignedBlockCompressed7
# 1131
}; 
#endif
# 1136 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1136
struct cudaResourceDesc { 
# 1137
cudaResourceType resType; 
# 1139
union { 
# 1140
struct { 
# 1141
cudaArray_t array; 
# 1142
} array; 
# 1143
struct { 
# 1144
cudaMipmappedArray_t mipmap; 
# 1145
} mipmap; 
# 1146
struct { 
# 1147
void *devPtr; 
# 1148
cudaChannelFormatDesc desc; 
# 1149
size_t sizeInBytes; 
# 1150
} linear; 
# 1151
struct { 
# 1152
void *devPtr; 
# 1153
cudaChannelFormatDesc desc; 
# 1154
size_t width; 
# 1155
size_t height; 
# 1156
size_t pitchInBytes; 
# 1157
} pitch2D; 
# 1158
} res; 
# 1159
}; 
#endif
# 1164 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1164
struct cudaResourceViewDesc { 
# 1166
cudaResourceViewFormat format; 
# 1167
size_t width; 
# 1168
size_t height; 
# 1169
size_t depth; 
# 1170
unsigned firstMipmapLevel; 
# 1171
unsigned lastMipmapLevel; 
# 1172
unsigned firstLayer; 
# 1173
unsigned lastLayer; 
# 1174
}; 
#endif
# 1179 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1179
struct cudaPointerAttributes { 
# 1189 "/usr/local/cuda/bin/..//include/driver_types.h"
__attribute((deprecated)) cudaMemoryType memoryType; 
# 1195
cudaMemoryType type; 
# 1206 "/usr/local/cuda/bin/..//include/driver_types.h"
int device; 
# 1212
void *devicePointer; 
# 1221 "/usr/local/cuda/bin/..//include/driver_types.h"
void *hostPointer; 
# 1228
__attribute((deprecated)) int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1229
}; 
#endif
# 1234 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1234
struct cudaFuncAttributes { 
# 1241
size_t sharedSizeBytes; 
# 1247
size_t constSizeBytes; 
# 1252
size_t localSizeBytes; 
# 1259
int maxThreadsPerBlock; 
# 1264
int numRegs; 
# 1271
int ptxVersion; 
# 1278
int binaryVersion; 
# 1284
int cacheModeCA; 
# 1291
int maxDynamicSharedSizeBytes; 
# 1298
int preferredShmemCarveout; 
# 1299
}; 
#endif
# 1304 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1304
enum cudaFuncAttribute { 
# 1306
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1307
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1308
cudaFuncAttributeMax
# 1309
}; 
#endif
# 1314 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1314
enum cudaFuncCache { 
# 1316
cudaFuncCachePreferNone, 
# 1317
cudaFuncCachePreferShared, 
# 1318
cudaFuncCachePreferL1, 
# 1319
cudaFuncCachePreferEqual
# 1320
}; 
#endif
# 1326 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1326
enum cudaSharedMemConfig { 
# 1328
cudaSharedMemBankSizeDefault, 
# 1329
cudaSharedMemBankSizeFourByte, 
# 1330
cudaSharedMemBankSizeEightByte
# 1331
}; 
#endif
# 1336 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1336
enum cudaSharedCarveout { 
# 1337
cudaSharedmemCarveoutDefault = (-1), 
# 1338
cudaSharedmemCarveoutMaxShared = 100, 
# 1339
cudaSharedmemCarveoutMaxL1 = 0
# 1340
}; 
#endif
# 1345 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1345
enum cudaComputeMode { 
# 1347
cudaComputeModeDefault, 
# 1348
cudaComputeModeExclusive, 
# 1349
cudaComputeModeProhibited, 
# 1350
cudaComputeModeExclusiveProcess
# 1351
}; 
#endif
# 1356 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1356
enum cudaLimit { 
# 1358
cudaLimitStackSize, 
# 1359
cudaLimitPrintfFifoSize, 
# 1360
cudaLimitMallocHeapSize, 
# 1361
cudaLimitDevRuntimeSyncDepth, 
# 1362
cudaLimitDevRuntimePendingLaunchCount, 
# 1363
cudaLimitMaxL2FetchGranularity
# 1364
}; 
#endif
# 1369 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1369
enum cudaMemoryAdvise { 
# 1371
cudaMemAdviseSetReadMostly = 1, 
# 1372
cudaMemAdviseUnsetReadMostly, 
# 1373
cudaMemAdviseSetPreferredLocation, 
# 1374
cudaMemAdviseUnsetPreferredLocation, 
# 1375
cudaMemAdviseSetAccessedBy, 
# 1376
cudaMemAdviseUnsetAccessedBy
# 1377
}; 
#endif
# 1382 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1382
enum cudaMemRangeAttribute { 
# 1384
cudaMemRangeAttributeReadMostly = 1, 
# 1385
cudaMemRangeAttributePreferredLocation, 
# 1386
cudaMemRangeAttributeAccessedBy, 
# 1387
cudaMemRangeAttributeLastPrefetchLocation
# 1388
}; 
#endif
# 1393 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1393
enum cudaOutputMode { 
# 1395
cudaKeyValuePair, 
# 1396
cudaCSV
# 1397
}; 
#endif
# 1402 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1402
enum cudaDeviceAttr { 
# 1404
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1405
cudaDevAttrMaxBlockDimX, 
# 1406
cudaDevAttrMaxBlockDimY, 
# 1407
cudaDevAttrMaxBlockDimZ, 
# 1408
cudaDevAttrMaxGridDimX, 
# 1409
cudaDevAttrMaxGridDimY, 
# 1410
cudaDevAttrMaxGridDimZ, 
# 1411
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1412
cudaDevAttrTotalConstantMemory, 
# 1413
cudaDevAttrWarpSize, 
# 1414
cudaDevAttrMaxPitch, 
# 1415
cudaDevAttrMaxRegistersPerBlock, 
# 1416
cudaDevAttrClockRate, 
# 1417
cudaDevAttrTextureAlignment, 
# 1418
cudaDevAttrGpuOverlap, 
# 1419
cudaDevAttrMultiProcessorCount, 
# 1420
cudaDevAttrKernelExecTimeout, 
# 1421
cudaDevAttrIntegrated, 
# 1422
cudaDevAttrCanMapHostMemory, 
# 1423
cudaDevAttrComputeMode, 
# 1424
cudaDevAttrMaxTexture1DWidth, 
# 1425
cudaDevAttrMaxTexture2DWidth, 
# 1426
cudaDevAttrMaxTexture2DHeight, 
# 1427
cudaDevAttrMaxTexture3DWidth, 
# 1428
cudaDevAttrMaxTexture3DHeight, 
# 1429
cudaDevAttrMaxTexture3DDepth, 
# 1430
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1431
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1432
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1433
cudaDevAttrSurfaceAlignment, 
# 1434
cudaDevAttrConcurrentKernels, 
# 1435
cudaDevAttrEccEnabled, 
# 1436
cudaDevAttrPciBusId, 
# 1437
cudaDevAttrPciDeviceId, 
# 1438
cudaDevAttrTccDriver, 
# 1439
cudaDevAttrMemoryClockRate, 
# 1440
cudaDevAttrGlobalMemoryBusWidth, 
# 1441
cudaDevAttrL2CacheSize, 
# 1442
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1443
cudaDevAttrAsyncEngineCount, 
# 1444
cudaDevAttrUnifiedAddressing, 
# 1445
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1446
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1447
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1448
cudaDevAttrMaxTexture2DGatherHeight, 
# 1449
cudaDevAttrMaxTexture3DWidthAlt, 
# 1450
cudaDevAttrMaxTexture3DHeightAlt, 
# 1451
cudaDevAttrMaxTexture3DDepthAlt, 
# 1452
cudaDevAttrPciDomainId, 
# 1453
cudaDevAttrTexturePitchAlignment, 
# 1454
cudaDevAttrMaxTextureCubemapWidth, 
# 1455
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1456
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1457
cudaDevAttrMaxSurface1DWidth, 
# 1458
cudaDevAttrMaxSurface2DWidth, 
# 1459
cudaDevAttrMaxSurface2DHeight, 
# 1460
cudaDevAttrMaxSurface3DWidth, 
# 1461
cudaDevAttrMaxSurface3DHeight, 
# 1462
cudaDevAttrMaxSurface3DDepth, 
# 1463
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1464
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1465
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1466
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1467
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1468
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1469
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1470
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1471
cudaDevAttrMaxTexture1DLinearWidth, 
# 1472
cudaDevAttrMaxTexture2DLinearWidth, 
# 1473
cudaDevAttrMaxTexture2DLinearHeight, 
# 1474
cudaDevAttrMaxTexture2DLinearPitch, 
# 1475
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1476
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1477
cudaDevAttrComputeCapabilityMajor, 
# 1478
cudaDevAttrComputeCapabilityMinor, 
# 1479
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1480
cudaDevAttrStreamPrioritiesSupported, 
# 1481
cudaDevAttrGlobalL1CacheSupported, 
# 1482
cudaDevAttrLocalL1CacheSupported, 
# 1483
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1484
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1485
cudaDevAttrManagedMemory, 
# 1486
cudaDevAttrIsMultiGpuBoard, 
# 1487
cudaDevAttrMultiGpuBoardGroupID, 
# 1488
cudaDevAttrHostNativeAtomicSupported, 
# 1489
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1490
cudaDevAttrPageableMemoryAccess, 
# 1491
cudaDevAttrConcurrentManagedAccess, 
# 1492
cudaDevAttrComputePreemptionSupported, 
# 1493
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1494
cudaDevAttrReserved92, 
# 1495
cudaDevAttrReserved93, 
# 1496
cudaDevAttrReserved94, 
# 1497
cudaDevAttrCooperativeLaunch, 
# 1498
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1499
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
# 1500
cudaDevAttrCanFlushRemoteWrites, 
# 1501
cudaDevAttrHostRegisterSupported, 
# 1502
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
# 1503
cudaDevAttrDirectManagedMemAccessFromHost
# 1504
}; 
#endif
# 1510 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1510
enum cudaDeviceP2PAttr { 
# 1511
cudaDevP2PAttrPerformanceRank = 1, 
# 1512
cudaDevP2PAttrAccessSupported, 
# 1513
cudaDevP2PAttrNativeAtomicSupported, 
# 1514
cudaDevP2PAttrCudaArrayAccessSupported
# 1515
}; 
#endif
# 1522 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1522
struct CUuuid_st { 
# 1523
char bytes[16]; 
# 1524
}; 
#endif
# 1525 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef CUuuid_st 
# 1525
CUuuid; 
#endif
# 1527 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef CUuuid_st 
# 1527
cudaUUID_t; 
#endif
# 1532 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1532
struct cudaDeviceProp { 
# 1534
char name[256]; 
# 1535
cudaUUID_t uuid; 
# 1536
char luid[8]; 
# 1537
unsigned luidDeviceNodeMask; 
# 1538
size_t totalGlobalMem; 
# 1539
size_t sharedMemPerBlock; 
# 1540
int regsPerBlock; 
# 1541
int warpSize; 
# 1542
size_t memPitch; 
# 1543
int maxThreadsPerBlock; 
# 1544
int maxThreadsDim[3]; 
# 1545
int maxGridSize[3]; 
# 1546
int clockRate; 
# 1547
size_t totalConstMem; 
# 1548
int major; 
# 1549
int minor; 
# 1550
size_t textureAlignment; 
# 1551
size_t texturePitchAlignment; 
# 1552
int deviceOverlap; 
# 1553
int multiProcessorCount; 
# 1554
int kernelExecTimeoutEnabled; 
# 1555
int integrated; 
# 1556
int canMapHostMemory; 
# 1557
int computeMode; 
# 1558
int maxTexture1D; 
# 1559
int maxTexture1DMipmap; 
# 1560
int maxTexture1DLinear; 
# 1561
int maxTexture2D[2]; 
# 1562
int maxTexture2DMipmap[2]; 
# 1563
int maxTexture2DLinear[3]; 
# 1564
int maxTexture2DGather[2]; 
# 1565
int maxTexture3D[3]; 
# 1566
int maxTexture3DAlt[3]; 
# 1567
int maxTextureCubemap; 
# 1568
int maxTexture1DLayered[2]; 
# 1569
int maxTexture2DLayered[3]; 
# 1570
int maxTextureCubemapLayered[2]; 
# 1571
int maxSurface1D; 
# 1572
int maxSurface2D[2]; 
# 1573
int maxSurface3D[3]; 
# 1574
int maxSurface1DLayered[2]; 
# 1575
int maxSurface2DLayered[3]; 
# 1576
int maxSurfaceCubemap; 
# 1577
int maxSurfaceCubemapLayered[2]; 
# 1578
size_t surfaceAlignment; 
# 1579
int concurrentKernels; 
# 1580
int ECCEnabled; 
# 1581
int pciBusID; 
# 1582
int pciDeviceID; 
# 1583
int pciDomainID; 
# 1584
int tccDriver; 
# 1585
int asyncEngineCount; 
# 1586
int unifiedAddressing; 
# 1587
int memoryClockRate; 
# 1588
int memoryBusWidth; 
# 1589
int l2CacheSize; 
# 1590
int maxThreadsPerMultiProcessor; 
# 1591
int streamPrioritiesSupported; 
# 1592
int globalL1CacheSupported; 
# 1593
int localL1CacheSupported; 
# 1594
size_t sharedMemPerMultiprocessor; 
# 1595
int regsPerMultiprocessor; 
# 1596
int managedMemory; 
# 1597
int isMultiGpuBoard; 
# 1598
int multiGpuBoardGroupID; 
# 1599
int hostNativeAtomicSupported; 
# 1600
int singleToDoublePrecisionPerfRatio; 
# 1601
int pageableMemoryAccess; 
# 1602
int concurrentManagedAccess; 
# 1603
int computePreemptionSupported; 
# 1604
int canUseHostPointerForRegisteredMem; 
# 1605
int cooperativeLaunch; 
# 1606
int cooperativeMultiDeviceLaunch; 
# 1607
size_t sharedMemPerBlockOptin; 
# 1608
int pageableMemoryAccessUsesHostPageTables; 
# 1609
int directManagedMemAccessFromHost; 
# 1610
}; 
#endif
# 1703 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef 
# 1700
struct cudaIpcEventHandle_st { 
# 1702
char reserved[64]; 
# 1703
} cudaIpcEventHandle_t; 
#endif
# 1711 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef 
# 1708
struct cudaIpcMemHandle_st { 
# 1710
char reserved[64]; 
# 1711
} cudaIpcMemHandle_t; 
#endif
# 1716 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1716
enum cudaExternalMemoryHandleType { 
# 1720
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
# 1724
cudaExternalMemoryHandleTypeOpaqueWin32, 
# 1728
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
# 1732
cudaExternalMemoryHandleTypeD3D12Heap, 
# 1736
cudaExternalMemoryHandleTypeD3D12Resource
# 1737
}; 
#endif
# 1747 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1747
struct cudaExternalMemoryHandleDesc { 
# 1751
cudaExternalMemoryHandleType type; 
# 1752
union { 
# 1758
int fd; 
# 1770 "/usr/local/cuda/bin/..//include/driver_types.h"
struct { 
# 1774
void *handle; 
# 1779
const void *name; 
# 1780
} win32; 
# 1781
} handle; 
# 1785
unsigned long long size; 
# 1789
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1790
}; 
#endif
# 1795 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1795
struct cudaExternalMemoryBufferDesc { 
# 1799
unsigned long long offset; 
# 1803
unsigned long long size; 
# 1807
unsigned flags; 
# 1808
}; 
#endif
# 1813 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1813
struct cudaExternalMemoryMipmappedArrayDesc { 
# 1818
unsigned long long offset; 
# 1822
cudaChannelFormatDesc formatDesc; 
# 1826
cudaExtent extent; 
# 1831
unsigned flags; 
# 1835
unsigned numLevels; 
# 1836
}; 
#endif
# 1841 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1841
enum cudaExternalSemaphoreHandleType { 
# 1845
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
# 1849
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
# 1853
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
# 1857
cudaExternalSemaphoreHandleTypeD3D12Fence
# 1858
}; 
#endif
# 1863 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1863
struct cudaExternalSemaphoreHandleDesc { 
# 1867
cudaExternalSemaphoreHandleType type; 
# 1868
union { 
# 1873
int fd; 
# 1884 "/usr/local/cuda/bin/..//include/driver_types.h"
struct { 
# 1888
void *handle; 
# 1893
const void *name; 
# 1894
} win32; 
# 1895
} handle; 
# 1899
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1900
}; 
#endif
# 1905 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1905
struct cudaExternalSemaphoreSignalParams { 
# 1906
union { 
# 1910
struct { 
# 1914
unsigned long long value; 
# 1915
} fence; 
# 1916
} params; 
# 1920
unsigned flags; 
# 1921
}; 
#endif
# 1926 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1926
struct cudaExternalSemaphoreWaitParams { 
# 1927
union { 
# 1931
struct { 
# 1935
unsigned long long value; 
# 1936
} fence; 
# 1937
} params; 
# 1941
unsigned flags; 
# 1942
}; 
#endif
# 1954 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef cudaError 
# 1954
cudaError_t; 
#endif
# 1959 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 1959
cudaStream_t; 
#endif
# 1964 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 1964
cudaEvent_t; 
#endif
# 1969 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 1969
cudaGraphicsResource_t; 
#endif
# 1974 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 1974
cudaOutputMode_t; 
#endif
# 1979 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
# 1979
cudaExternalMemory_t; 
#endif
# 1984 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
# 1984
cudaExternalSemaphore_t; 
#endif
# 1989 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUgraph_st *
# 1989
cudaGraph_t; 
#endif
# 1994 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
typedef struct CUgraphNode_st *
# 1994
cudaGraphNode_t; 
#endif
# 1999 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 1999
enum cudaCGScope { 
# 2000
cudaCGScopeInvalid, 
# 2001
cudaCGScopeGrid, 
# 2002
cudaCGScopeMultiGrid
# 2003
}; 
#endif
# 2008 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 2008
struct cudaLaunchParams { 
# 2010
void *func; 
# 2011
dim3 gridDim; 
# 2012
dim3 blockDim; 
# 2013
void **args; 
# 2014
size_t sharedMem; 
# 2015
cudaStream_t stream; 
# 2016
}; 
#endif
# 2021 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 2021
struct cudaKernelNodeParams { 
# 2022
void *func; 
# 2023
dim3 gridDim; 
# 2024
dim3 blockDim; 
# 2025
unsigned sharedMemBytes; 
# 2026
void **kernelParams; 
# 2027
void **extra; 
# 2028
}; 
#endif
# 2033 "/usr/local/cuda/bin/..//include/driver_types.h"
#if 0
# 2033
enum cudaGraphNodeType { 
# 2034
cudaGraphNodeTypeKernel, 
# 2035
cudaGraphNodeTypeMemcpy, 
# 2036
cudaGraphNodeTypeMemset, 
# 2037
cudaGraphNodeTypeHost, 
# 2038
cudaGraphNodeTypeGraph, 
# 2039
cudaGraphNodeTypeEmpty, 
# 2040
cudaGraphNodeTypeCount
# 2041
}; 
#endif
# 2046 "/usr/local/cuda/bin/..//include/driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
# 84 "/usr/local/cuda/bin/..//include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/usr/local/cuda/bin/..//include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/usr/local/cuda/bin/..//include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/usr/local/cuda/bin/..//include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 155
int __cudaReserved[15]; 
# 156
}; 
#endif
# 161 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
# 161
struct cudaTextureDesc { 
# 166
cudaTextureAddressMode addressMode[3]; 
# 170
cudaTextureFilterMode filterMode; 
# 174
cudaTextureReadMode readMode; 
# 178
int sRGB; 
# 182
float borderColor[4]; 
# 186
int normalizedCoords; 
# 190
unsigned maxAnisotropy; 
# 194
cudaTextureFilterMode mipmapFilterMode; 
# 198
float mipmapLevelBias; 
# 202
float minMipmapLevelClamp; 
# 206
float maxMipmapLevelClamp; 
# 207
}; 
#endif
# 212 "/usr/local/cuda/bin/..//include/texture_types.h"
#if 0
typedef unsigned long long 
# 212
cudaTextureObject_t; 
#endif
# 70 "/usr/local/cuda/bin/..//include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_32F = 0, 
# 59
CUDA_C_32F = 4, 
# 60
CUDA_R_64F = 1, 
# 61
CUDA_C_64F = 5, 
# 62
CUDA_R_8I = 3, 
# 63
CUDA_C_8I = 7, 
# 64
CUDA_R_8U, 
# 65
CUDA_C_8U, 
# 66
CUDA_R_32I, 
# 67
CUDA_C_32I, 
# 68
CUDA_R_32U, 
# 69
CUDA_C_32U
# 70
} cudaDataType; 
# 78
typedef 
# 73
enum libraryPropertyType_t { 
# 75
MAJOR_VERSION, 
# 76
MINOR_VERSION, 
# 77
PATCH_LEVEL
# 78
} libraryPropertyType; 
# 121 "/usr/local/cuda/bin/..//include/cuda_device_runtime_api.h"
extern "C" {
# 123
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 124
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 125
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 126
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 127
extern cudaError_t cudaDeviceSynchronize(); 
# 128
extern cudaError_t cudaGetLastError(); 
# 129
extern cudaError_t cudaPeekAtLastError(); 
# 130
extern const char *cudaGetErrorString(cudaError_t error); 
# 131
extern const char *cudaGetErrorName(cudaError_t error); 
# 132
extern cudaError_t cudaGetDeviceCount(int * count); 
# 133
extern cudaError_t cudaGetDevice(int * device); 
# 134
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 135
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 136
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 137
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 138
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 139
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 140
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 141
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 142
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 143
extern cudaError_t cudaFree(void * devPtr); 
# 144
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 145
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 146
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 147
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 148
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 149
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 150
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 151
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 152
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 153
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 154
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 155
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 156
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 157
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 178 "/usr/local/cuda/bin/..//include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 206 "/usr/local/cuda/bin/..//include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 207
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 208
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 226 "/usr/local/cuda/bin/..//include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 227
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 230
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 231
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 233
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 234
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 235
__attribute__((unused)) extern cudaError_t cudaCGSynchronizeGrid(unsigned long long handle, unsigned flags); 
# 236
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 237
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 238
}
# 240
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 241
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 242
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 243
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 245 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern "C" {
# 280 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceReset(); 
# 301 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSynchronize(); 
# 380 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 413 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 446 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 483 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 527 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 558 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 602 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 629 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 659 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 706 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 746 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 789 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 844 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 879 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 921 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadExit(); 
# 947 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSynchronize(); 
# 996 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 1029 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1065 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1112 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1171 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(); 
# 1217 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaPeekAtLastError(); 
# 1233 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern const char *cudaGetErrorName(cudaError_t error); 
# 1249 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t error); 
# 1277 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1548 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1735 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1775 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 1796 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 1833 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaSetDevice(int device); 
# 1854 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetDevice(int * device); 
# 1885 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 1951 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 1997 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 2037 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 2069 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2115 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2142 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2167 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2198 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2224 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 2232
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2299 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2323 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2348 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2431 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2460 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamBeginCapture(cudaStream_t stream); 
# 2484 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
# 2523 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
# 2561 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 2598 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 2637 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 2668 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 2698 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 2725 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 2768 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 2903 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
# 2955 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
# 3009 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
# 3032 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
# 3126 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
# 3165 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3208 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
# 3230 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
# 3295 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 3352 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 3451 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 3501 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 3557 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 3593 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 3630 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 3656 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 3682 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
__attribute((deprecated)) extern cudaError_t cudaSetDoubleForHost(double * d); 
# 3748 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
# 3803 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 3847 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 3899 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0); 
# 3930 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset); 
# 3973 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaLaunch(const void * func); 
# 4095 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 4126 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 4159 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 4202 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 4248 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 4277 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void * devPtr); 
# 4300 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFreeHost(void * ptr); 
# 4323 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 4346 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 4412 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 4496 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 4519 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaHostUnregister(void * ptr); 
# 4564 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 4586 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 4625 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 4764 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 4903 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 4932 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 5037 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 5068 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 5186 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 5212 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 5234 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 5260 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 5303 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 5338 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 5379 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 5419 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 5460 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 5508 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 5557 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 5606 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 5653 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 5696 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 5739 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 5795 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5830 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 5879 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5927 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5989 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6046 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6102 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6153 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6204 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 6233 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 6267 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 6311 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 6347 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 6388 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 6439 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 6467 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 6494 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 6564 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 6680 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 6739 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 6778 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 6944 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 6985 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 7027 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 7049 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 7112 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 7147 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 7186 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 7221 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 7253 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 7291 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 7320 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 7362 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 7398 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 7451 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 7508 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 7544 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 7582 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 7606 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 7633 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 7661 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 7704 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 7727 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 7957 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 7976 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 7996 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 8016 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 8037 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 8080 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 8099 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 8118 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 8152 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 8177 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 8224 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
# 8321 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
# 8354 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
# 8379 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
# 8423 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
# 8446 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
# 8469 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
# 8511 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
# 8534 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
# 8557 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
# 8596 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
# 8619 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
# 8642 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
# 8680 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
# 8704 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
# 8741 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, cudaGraphNode_t * pDependencies, size_t numDependencies); 
# 8768 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
# 8796 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
# 8827 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
# 8858 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
# 8889 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
# 8923 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
# 8954 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
# 8986 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
# 9017 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t numDependencies); 
# 9048 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t numDependencies); 
# 9074 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node); 
# 9110 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
# 9135 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
# 9156 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
# 9176 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
extern cudaError_t cudaGraphDestroy(cudaGraph_t graph); 
# 9181
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 9432 "/usr/local/cuda/bin/..//include/cuda_runtime_api.h"
}
# 104 "/usr/local/cuda/bin/..//include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 105
{ 
# 106
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 107
} 
# 109
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 110
{ 
# 111
int e = (((int)sizeof(unsigned short)) * 8); 
# 113
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 114
} 
# 116
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 117
{ 
# 118
int e = (((int)sizeof(unsigned short)) * 8); 
# 120
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 121
} 
# 123
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 124
{ 
# 125
int e = (((int)sizeof(unsigned short)) * 8); 
# 127
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 128
} 
# 130
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 131
{ 
# 132
int e = (((int)sizeof(unsigned short)) * 8); 
# 134
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 135
} 
# 137
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 138
{ 
# 139
int e = (((int)sizeof(char)) * 8); 
# 144
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 146
} 
# 148
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 149
{ 
# 150
int e = (((int)sizeof(signed char)) * 8); 
# 152
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 153
} 
# 155
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 156
{ 
# 157
int e = (((int)sizeof(unsigned char)) * 8); 
# 159
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 160
} 
# 162
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 163
{ 
# 164
int e = (((int)sizeof(signed char)) * 8); 
# 166
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 167
} 
# 169
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 170
{ 
# 171
int e = (((int)sizeof(unsigned char)) * 8); 
# 173
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 174
} 
# 176
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 177
{ 
# 178
int e = (((int)sizeof(signed char)) * 8); 
# 180
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 181
} 
# 183
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 184
{ 
# 185
int e = (((int)sizeof(unsigned char)) * 8); 
# 187
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 188
} 
# 190
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 191
{ 
# 192
int e = (((int)sizeof(signed char)) * 8); 
# 194
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 195
} 
# 197
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 198
{ 
# 199
int e = (((int)sizeof(unsigned char)) * 8); 
# 201
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 202
} 
# 204
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 205
{ 
# 206
int e = (((int)sizeof(short)) * 8); 
# 208
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 209
} 
# 211
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 212
{ 
# 213
int e = (((int)sizeof(unsigned short)) * 8); 
# 215
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 216
} 
# 218
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 219
{ 
# 220
int e = (((int)sizeof(short)) * 8); 
# 222
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 223
} 
# 225
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 226
{ 
# 227
int e = (((int)sizeof(unsigned short)) * 8); 
# 229
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 230
} 
# 232
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 233
{ 
# 234
int e = (((int)sizeof(short)) * 8); 
# 236
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 237
} 
# 239
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 240
{ 
# 241
int e = (((int)sizeof(unsigned short)) * 8); 
# 243
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 244
} 
# 246
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 247
{ 
# 248
int e = (((int)sizeof(short)) * 8); 
# 250
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 251
} 
# 253
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 254
{ 
# 255
int e = (((int)sizeof(unsigned short)) * 8); 
# 257
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 258
} 
# 260
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 261
{ 
# 262
int e = (((int)sizeof(int)) * 8); 
# 264
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 265
} 
# 267
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 268
{ 
# 269
int e = (((int)sizeof(unsigned)) * 8); 
# 271
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 272
} 
# 274
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 275
{ 
# 276
int e = (((int)sizeof(int)) * 8); 
# 278
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 279
} 
# 281
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 282
{ 
# 283
int e = (((int)sizeof(unsigned)) * 8); 
# 285
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 286
} 
# 288
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 289
{ 
# 290
int e = (((int)sizeof(int)) * 8); 
# 292
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 293
} 
# 295
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 296
{ 
# 297
int e = (((int)sizeof(unsigned)) * 8); 
# 299
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 300
} 
# 302
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 303
{ 
# 304
int e = (((int)sizeof(int)) * 8); 
# 306
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 307
} 
# 309
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 310
{ 
# 311
int e = (((int)sizeof(unsigned)) * 8); 
# 313
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 314
} 
# 376 "/usr/local/cuda/bin/..//include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 377
{ 
# 378
int e = (((int)sizeof(float)) * 8); 
# 380
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 381
} 
# 383
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 384
{ 
# 385
int e = (((int)sizeof(float)) * 8); 
# 387
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 388
} 
# 390
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 391
{ 
# 392
int e = (((int)sizeof(float)) * 8); 
# 394
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 395
} 
# 397
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 398
{ 
# 399
int e = (((int)sizeof(float)) * 8); 
# 401
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 402
} 
# 79 "/usr/local/cuda/bin/..//include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/usr/local/cuda/bin/..//include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/usr/local/cuda/bin/..//include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 73 "/usr/local/cuda/bin/..//include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 75
static inline uchar1 make_uchar1(unsigned char x); 
# 77
static inline char2 make_char2(signed char x, signed char y); 
# 79
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 81
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 83
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 85
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 87
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 89
static inline short1 make_short1(short x); 
# 91
static inline ushort1 make_ushort1(unsigned short x); 
# 93
static inline short2 make_short2(short x, short y); 
# 95
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 97
static inline short3 make_short3(short x, short y, short z); 
# 99
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 101
static inline short4 make_short4(short x, short y, short z, short w); 
# 103
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 105
static inline int1 make_int1(int x); 
# 107
static inline uint1 make_uint1(unsigned x); 
# 109
static inline int2 make_int2(int x, int y); 
# 111
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 113
static inline int3 make_int3(int x, int y, int z); 
# 115
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 117
static inline int4 make_int4(int x, int y, int z, int w); 
# 119
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 121
static inline long1 make_long1(long x); 
# 123
static inline ulong1 make_ulong1(unsigned long x); 
# 125
static inline long2 make_long2(long x, long y); 
# 127
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 129
static inline long3 make_long3(long x, long y, long z); 
# 131
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 133
static inline long4 make_long4(long x, long y, long z, long w); 
# 135
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 137
static inline float1 make_float1(float x); 
# 139
static inline float2 make_float2(float x, float y); 
# 141
static inline float3 make_float3(float x, float y, float z); 
# 143
static inline float4 make_float4(float x, float y, float z, float w); 
# 145
static inline longlong1 make_longlong1(long long x); 
# 147
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 149
static inline longlong2 make_longlong2(long long x, long long y); 
# 151
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 153
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 155
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 157
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 159
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 161
static inline double1 make_double1(double x); 
# 163
static inline double2 make_double2(double x, double y); 
# 165
static inline double3 make_double3(double x, double y, double z); 
# 167
static inline double4 make_double4(double x, double y, double z, double w); 
# 73 "/usr/local/cuda/bin/..//include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 74
{ 
# 75
char1 t; (t.x) = x; return t; 
# 76
} 
# 78
static inline uchar1 make_uchar1(unsigned char x) 
# 79
{ 
# 80
uchar1 t; (t.x) = x; return t; 
# 81
} 
# 83
static inline char2 make_char2(signed char x, signed char y) 
# 84
{ 
# 85
char2 t; (t.x) = x; (t.y) = y; return t; 
# 86
} 
# 88
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 89
{ 
# 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 91
} 
# 93
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 94
{ 
# 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 96
} 
# 98
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 99
{ 
# 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 101
} 
# 103
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 104
{ 
# 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 106
} 
# 108
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 109
{ 
# 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 111
} 
# 113
static inline short1 make_short1(short x) 
# 114
{ 
# 115
short1 t; (t.x) = x; return t; 
# 116
} 
# 118
static inline ushort1 make_ushort1(unsigned short x) 
# 119
{ 
# 120
ushort1 t; (t.x) = x; return t; 
# 121
} 
# 123
static inline short2 make_short2(short x, short y) 
# 124
{ 
# 125
short2 t; (t.x) = x; (t.y) = y; return t; 
# 126
} 
# 128
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 129
{ 
# 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 131
} 
# 133
static inline short3 make_short3(short x, short y, short z) 
# 134
{ 
# 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 136
} 
# 138
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 139
{ 
# 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 141
} 
# 143
static inline short4 make_short4(short x, short y, short z, short w) 
# 144
{ 
# 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 146
} 
# 148
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 149
{ 
# 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 151
} 
# 153
static inline int1 make_int1(int x) 
# 154
{ 
# 155
int1 t; (t.x) = x; return t; 
# 156
} 
# 158
static inline uint1 make_uint1(unsigned x) 
# 159
{ 
# 160
uint1 t; (t.x) = x; return t; 
# 161
} 
# 163
static inline int2 make_int2(int x, int y) 
# 164
{ 
# 165
int2 t; (t.x) = x; (t.y) = y; return t; 
# 166
} 
# 168
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 169
{ 
# 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 171
} 
# 173
static inline int3 make_int3(int x, int y, int z) 
# 174
{ 
# 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 176
} 
# 178
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 179
{ 
# 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 181
} 
# 183
static inline int4 make_int4(int x, int y, int z, int w) 
# 184
{ 
# 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 186
} 
# 188
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 189
{ 
# 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 191
} 
# 193
static inline long1 make_long1(long x) 
# 194
{ 
# 195
long1 t; (t.x) = x; return t; 
# 196
} 
# 198
static inline ulong1 make_ulong1(unsigned long x) 
# 199
{ 
# 200
ulong1 t; (t.x) = x; return t; 
# 201
} 
# 203
static inline long2 make_long2(long x, long y) 
# 204
{ 
# 205
long2 t; (t.x) = x; (t.y) = y; return t; 
# 206
} 
# 208
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 209
{ 
# 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 211
} 
# 213
static inline long3 make_long3(long x, long y, long z) 
# 214
{ 
# 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 216
} 
# 218
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 219
{ 
# 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 221
} 
# 223
static inline long4 make_long4(long x, long y, long z, long w) 
# 224
{ 
# 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 226
} 
# 228
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 229
{ 
# 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 231
} 
# 233
static inline float1 make_float1(float x) 
# 234
{ 
# 235
float1 t; (t.x) = x; return t; 
# 236
} 
# 238
static inline float2 make_float2(float x, float y) 
# 239
{ 
# 240
float2 t; (t.x) = x; (t.y) = y; return t; 
# 241
} 
# 243
static inline float3 make_float3(float x, float y, float z) 
# 244
{ 
# 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 246
} 
# 248
static inline float4 make_float4(float x, float y, float z, float w) 
# 249
{ 
# 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 251
} 
# 253
static inline longlong1 make_longlong1(long long x) 
# 254
{ 
# 255
longlong1 t; (t.x) = x; return t; 
# 256
} 
# 258
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 259
{ 
# 260
ulonglong1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline longlong2 make_longlong2(long long x, long long y) 
# 264
{ 
# 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 268
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 269
{ 
# 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 271
} 
# 273
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 274
{ 
# 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 276
} 
# 278
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 279
{ 
# 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 281
} 
# 283
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 284
{ 
# 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 286
} 
# 288
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 289
{ 
# 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 291
} 
# 293
static inline double1 make_double1(double x) 
# 294
{ 
# 295
double1 t; (t.x) = x; return t; 
# 296
} 
# 298
static inline double2 make_double2(double x, double y) 
# 299
{ 
# 300
double2 t; (t.x) = x; (t.y) = y; return t; 
# 301
} 
# 303
static inline double3 make_double3(double x, double y, double z) 
# 304
{ 
# 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 306
} 
# 308
static inline double4 make_double4(double x, double y, double z, double w) 
# 309
{ 
# 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 311
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 90 "/usr/include/string.h" 3
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 171
extern char *strdup(const char * __s) throw()
# 172
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 179
extern char *strndup(const char * __string, size_t __n) throw()
# 180
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 209 "/usr/include/string.h" 3
extern "C++" {
# 211
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 212
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 213
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 214
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 229 "/usr/include/string.h" 3
}
# 236
extern "C++" {
# 238
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 239
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 240
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 241
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 256 "/usr/include/string.h" 3
}
# 267
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 268
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 269
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 270
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 280
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 281
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 284
extern size_t strspn(const char * __s, const char * __accept) throw()
# 285
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 288
extern "C++" {
# 290
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 291
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 292
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 293
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 308 "/usr/include/string.h" 3
}
# 315
extern "C++" {
# 317
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 318
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 319
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 320
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 335 "/usr/include/string.h" 3
}
# 343
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 344
 __attribute((__nonnull__(2))); 
# 349
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 352
 __attribute((__nonnull__(2, 3))); 
# 354
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 356
 __attribute((__nonnull__(2, 3))); 
# 362
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 363
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 364
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 366
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 377 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 379
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 383
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 385
 __attribute((__nonnull__(1, 2))); 
# 386
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 388
 __attribute((__nonnull__(1, 2))); 
# 394
extern size_t strlen(const char * __s) throw()
# 395
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 401
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 402
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 408
extern char *strerror(int __errnum) throw(); 
# 433 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 434
 __attribute((__nonnull__(2))); 
# 440
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 446
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 450
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 454
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 457
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 458
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 462
extern "C++" {
# 464
extern char *index(char * __s, int __c) throw() __asm__("index")
# 465
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 466
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 467
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 482 "/usr/include/string.h" 3
}
# 490
extern "C++" {
# 492
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 493
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 494
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 495
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 510 "/usr/include/string.h" 3
}
# 518
extern int ffs(int __i) throw() __attribute((const)); 
# 523
extern int ffsl(long __l) throw() __attribute((const)); 
# 524
__extension__ extern int ffsll(long long __ll) throw()
# 525
 __attribute((const)); 
# 529
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 530
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 533
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 534
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 540
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 542
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 544
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 546
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 552
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 554
 __attribute((__nonnull__(1, 2))); 
# 559
extern char *strsignal(int __sig) throw(); 
# 562
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 563
 __attribute((__nonnull__(1, 2))); 
# 564
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 565
 __attribute((__nonnull__(1, 2))); 
# 569
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 571
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 579
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 580
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 583
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 586
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 594
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 595
 __attribute((__nonnull__(1))); 
# 596
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 597
 __attribute((__nonnull__(1))); 
# 658 "/usr/include/string.h" 3
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 124 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned long __dev_t; 
# 125
typedef unsigned __uid_t; 
# 126
typedef unsigned __gid_t; 
# 127
typedef unsigned long __ino_t; 
# 128
typedef unsigned long __ino64_t; 
# 129
typedef unsigned __mode_t; 
# 130
typedef unsigned long __nlink_t; 
# 131
typedef long __off_t; 
# 132
typedef long __off64_t; 
# 133
typedef int __pid_t; 
# 134
typedef struct { int __val[2]; } __fsid_t; 
# 135
typedef long __clock_t; 
# 136
typedef unsigned long __rlim_t; 
# 137
typedef unsigned long __rlim64_t; 
# 138
typedef unsigned __id_t; 
# 139
typedef long __time_t; 
# 140
typedef unsigned __useconds_t; 
# 141
typedef long __suseconds_t; 
# 143
typedef int __daddr_t; 
# 144
typedef int __key_t; 
# 147
typedef int __clockid_t; 
# 150
typedef void *__timer_t; 
# 153
typedef long __blksize_t; 
# 158
typedef long __blkcnt_t; 
# 159
typedef long __blkcnt64_t; 
# 162
typedef unsigned long __fsblkcnt_t; 
# 163
typedef unsigned long __fsblkcnt64_t; 
# 166
typedef unsigned long __fsfilcnt_t; 
# 167
typedef unsigned long __fsfilcnt64_t; 
# 170
typedef long __fsword_t; 
# 172
typedef long __ssize_t; 
# 175
typedef long __syscall_slong_t; 
# 177
typedef unsigned long __syscall_ulong_t; 
# 181
typedef __off64_t __loff_t; 
# 182
typedef __quad_t *__qaddr_t; 
# 183
typedef char *__caddr_t; 
# 186
typedef long __intptr_t; 
# 189
typedef unsigned __socklen_t; 
# 30 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/x86_64-linux-gnu/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75 "/usr/include/time.h" 3
typedef __time_t time_t; 
# 91 "/usr/include/time.h" 3
typedef __clockid_t clockid_t; 
# 103 "/usr/include/time.h" 3
typedef __timer_t timer_t; 
# 120 "/usr/include/time.h" 3
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189 "/usr/include/time.h" 3
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403 "/usr/include/time.h" 3
extern int getdate_err; 
# 412 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 426 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 80 "/usr/local/cuda/bin/..//include/crt/common_functions.h"
extern "C" {
# 83
extern clock_t clock() throw(); 
# 88 "/usr/local/cuda/bin/..//include/crt/common_functions.h"
extern void *memset(void *, int, size_t) throw(); 
# 89 "/usr/local/cuda/bin/..//include/crt/common_functions.h"
extern void *memcpy(void *, const void *, size_t) throw(); 
# 91 "/usr/local/cuda/bin/..//include/crt/common_functions.h"
}
# 108 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern "C" {
# 192 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int abs(int) throw(); 
# 193 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long labs(long) throw(); 
# 194 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long long llabs(long long) throw(); 
# 244 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fabs(double x) throw(); 
# 285 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fabsf(float x) throw(); 
# 289 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern inline int min(int, int); 
# 291
extern inline unsigned umin(unsigned, unsigned); 
# 292
extern inline long long llmin(long long, long long); 
# 293
extern inline unsigned long long ullmin(unsigned long long, unsigned long long); 
# 314 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fminf(float x, float y) throw(); 
# 334 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fmin(double x, double y) throw(); 
# 341 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern inline int max(int, int); 
# 343
extern inline unsigned umax(unsigned, unsigned); 
# 344
extern inline long long llmax(long long, long long); 
# 345
extern inline unsigned long long ullmax(unsigned long long, unsigned long long); 
# 366 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fmaxf(float x, float y) throw(); 
# 386 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fmax(double, double) throw(); 
# 430 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double sin(double x) throw(); 
# 463 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cos(double x) throw(); 
# 482 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 498 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 543 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double tan(double x) throw(); 
# 612 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double sqrt(double x) throw(); 
# 684 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rsqrt(double x); 
# 754 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rsqrtf(float x); 
# 810 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double log2(double x) throw(); 
# 835 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double exp2(double x) throw(); 
# 860 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float exp2f(float x) throw(); 
# 887 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 910 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float exp10f(float x) throw(); 
# 956 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double expm1(double x) throw(); 
# 1001 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float expm1f(float x) throw(); 
# 1056 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float log2f(float x) throw(); 
# 1110 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double log10(double x) throw(); 
# 1181 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double log(double x) throw(); 
# 1275 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double log1p(double x) throw(); 
# 1372 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float log1pf(float x) throw(); 
# 1447 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double floor(double x) throw(); 
# 1486 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double exp(double x) throw(); 
# 1517 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cosh(double x) throw(); 
# 1547 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double sinh(double x) throw(); 
# 1577 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double tanh(double x) throw(); 
# 1612 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double acosh(double x) throw(); 
# 1650 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float acoshf(float x) throw(); 
# 1666 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double asinh(double x) throw(); 
# 1682 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float asinhf(float x) throw(); 
# 1736 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double atanh(double x) throw(); 
# 1790 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float atanhf(float x) throw(); 
# 1849 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double ldexp(double x, int exp) throw(); 
# 1905 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float ldexpf(float x, int exp) throw(); 
# 1957 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double logb(double x) throw(); 
# 2012 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float logbf(float x) throw(); 
# 2042 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int ilogb(double x) throw(); 
# 2072 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int ilogbf(float x) throw(); 
# 2148 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double scalbn(double x, int n) throw(); 
# 2224 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float scalbnf(float x, int n) throw(); 
# 2300 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double scalbln(double x, long n) throw(); 
# 2376 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float scalblnf(float x, long n) throw(); 
# 2454 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double frexp(double x, int * nptr) throw(); 
# 2529 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float frexpf(float x, int * nptr) throw(); 
# 2543 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double round(double x) throw(); 
# 2560 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float roundf(float x) throw(); 
# 2578 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long lround(double x) throw(); 
# 2596 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long lroundf(float x) throw(); 
# 2614 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long long llround(double x) throw(); 
# 2632 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long long llroundf(float x) throw(); 
# 2684 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 2701 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long lrint(double x) throw(); 
# 2718 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long lrintf(float x) throw(); 
# 2735 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long long llrint(double x) throw(); 
# 2752 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern long long llrintf(float x) throw(); 
# 2805 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double nearbyint(double x) throw(); 
# 2858 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float nearbyintf(float x) throw(); 
# 2920 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double ceil(double x) throw(); 
# 2932 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double trunc(double x) throw(); 
# 2947 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float truncf(float x) throw(); 
# 2973 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fdim(double x, double y) throw(); 
# 2999 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fdimf(float x, float y) throw(); 
# 3035 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double atan2(double y, double x) throw(); 
# 3066 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double atan(double x) throw(); 
# 3089 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double acos(double x) throw(); 
# 3121 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double asin(double x) throw(); 
# 3167 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 3219 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rhypot(double x, double y) throw(); 
# 3265 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float hypotf(float x, float y) throw(); 
# 3317 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rhypotf(float x, float y) throw(); 
# 3361 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double norm3d(double a, double b, double c) throw(); 
# 3412 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rnorm3d(double a, double b, double c) throw(); 
# 3461 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double norm4d(double a, double b, double c, double d) throw(); 
# 3517 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rnorm4d(double a, double b, double c, double d) throw(); 
# 3562 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double norm(int dim, const double * t) throw(); 
# 3613 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rnorm(int dim, const double * t) throw(); 
# 3665 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rnormf(int dim, const float * a) throw(); 
# 3709 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float normf(int dim, const float * a) throw(); 
# 3754 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float norm3df(float a, float b, float c) throw(); 
# 3805 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rnorm3df(float a, float b, float c) throw(); 
# 3854 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float norm4df(float a, float b, float c, float d) throw(); 
# 3910 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rnorm4df(float a, float b, float c, float d) throw(); 
# 3997 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cbrt(double x) throw(); 
# 4083 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float cbrtf(float x) throw(); 
# 4138 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double rcbrt(double x); 
# 4188 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float rcbrtf(float x); 
# 4248 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double sinpi(double x); 
# 4308 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float sinpif(float x); 
# 4360 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cospi(double x); 
# 4412 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float cospif(float x); 
# 4442 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern void sincospi(double x, double * sptr, double * cptr); 
# 4472 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern void sincospif(float x, float * sptr, float * cptr); 
# 4784 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double pow(double x, double y) throw(); 
# 4840 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double modf(double x, double * iptr) throw(); 
# 4899 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fmod(double x, double y) throw(); 
# 4985 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double remainder(double x, double y) throw(); 
# 5075 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float remainderf(float x, float y) throw(); 
# 5129 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double remquo(double x, double y, int * quo) throw(); 
# 5183 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float remquof(float x, float y, int * quo) throw(); 
# 5224 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double j0(double x) throw(); 
# 5266 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float j0f(float x) throw(); 
# 5327 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double j1(double x) throw(); 
# 5388 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float j1f(float x) throw(); 
# 5431 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double jn(int n, double x) throw(); 
# 5474 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float jnf(int n, float x) throw(); 
# 5526 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double y0(double x) throw(); 
# 5578 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float y0f(float x) throw(); 
# 5630 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double y1(double x) throw(); 
# 5682 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float y1f(float x) throw(); 
# 5735 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double yn(int n, double x) throw(); 
# 5788 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float ynf(int n, float x) throw(); 
# 5815 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cyl_bessel_i0(double x) throw(); 
# 5841 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float cyl_bessel_i0f(float x) throw(); 
# 5868 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double cyl_bessel_i1(double x) throw(); 
# 5894 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float cyl_bessel_i1f(float x) throw(); 
# 5977 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double erf(double x) throw(); 
# 6059 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float erff(float x) throw(); 
# 6123 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double erfinv(double y); 
# 6180 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float erfinvf(float y); 
# 6219 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double erfc(double x) throw(); 
# 6257 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float erfcf(float x) throw(); 
# 6385 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double lgamma(double x) throw(); 
# 6448 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double erfcinv(double y); 
# 6504 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float erfcinvf(float y); 
# 6562 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double normcdfinv(double y); 
# 6620 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float normcdfinvf(float y); 
# 6663 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double normcdf(double y); 
# 6706 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float normcdff(float y); 
# 6781 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double erfcx(double x); 
# 6856 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float erfcxf(float x); 
# 6990 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float lgammaf(float x) throw(); 
# 7099 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double tgamma(double x) throw(); 
# 7208 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float tgammaf(float x) throw(); 
# 7221 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double copysign(double x, double y) throw(); 
# 7234 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float copysignf(float x, float y) throw(); 
# 7271 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double nextafter(double x, double y) throw(); 
# 7308 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float nextafterf(float x, float y) throw(); 
# 7324 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double nan(const char * tagp) throw(); 
# 7340 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float nanf(const char * tagp) throw(); 
# 7347 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isinff(float) throw(); 
# 7348 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isnanf(float) throw(); 
# 7358 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 7359 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __finitef(float) throw(); 
# 7360 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __signbit(double) throw(); 
# 7361 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isnan(double) throw(); 
# 7362 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isinf(double) throw(); 
# 7365 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __signbitf(float) throw(); 
# 7524 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern double fma(double x, double y, double z) throw(); 
# 7682 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fmaf(float x, float y, float z) throw(); 
# 7693 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __signbitl(long double) throw(); 
# 7699 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __finitel(long double) throw(); 
# 7700 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isinfl(long double) throw(); 
# 7701 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern int __isnanl(long double) throw(); 
# 7751 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 7791 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float asinf(float x) throw(); 
# 7831 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float atanf(float x) throw(); 
# 7864 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float atan2f(float y, float x) throw(); 
# 7888 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float cosf(float x) throw(); 
# 7930 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float sinf(float x) throw(); 
# 7972 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float tanf(float x) throw(); 
# 7996 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float coshf(float x) throw(); 
# 8037 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float sinhf(float x) throw(); 
# 8067 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float tanhf(float x) throw(); 
# 8118 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float logf(float x) throw(); 
# 8168 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float expf(float x) throw(); 
# 8219 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float log10f(float x) throw(); 
# 8274 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float modff(float x, float * iptr) throw(); 
# 8582 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float powf(float x, float y) throw(); 
# 8651 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float sqrtf(float x) throw(); 
# 8710 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float ceilf(float x) throw(); 
# 8782 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float floorf(float x) throw(); 
# 8841 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern float fmodf(float x, float y) throw(); 
# 8856 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
}
# 28 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/x86_64-linux-gnu/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 122
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 128
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 131
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 134
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 141
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 144
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 153
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 156
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 162
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 169
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 178
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 181
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 184
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 187
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 192
extern int __isinf(double __value) throw() __attribute((const)); 
# 195
extern int __finite(double __value) throw() __attribute((const)); 
# 204
extern inline int isinf(double __value) throw() __attribute((const)); 
# 208
extern int finite(double __value) throw() __attribute((const)); 
# 211
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 215
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 221
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 228
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnan(double __value) throw() __attribute((const)); 
# 241
extern inline int isnan(double __value) throw() __attribute((const)); 
# 247
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 248
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 249
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 250
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 251
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 252
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 259
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 260
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 261
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 268
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 274
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 281
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 289
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 292
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 294
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 298
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 302
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 306
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 311
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 315
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 319
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 323
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 328
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 335
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 337
__extension__ extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 341
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 343
__extension__ extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 347
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 350
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 353
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 357
extern int __fpclassify(double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbit(double __value) throw()
# 362
 __attribute((const)); 
# 366
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 375
extern int __issignaling(double __value) throw()
# 376
 __attribute((const)); 
# 383
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern float exp10f(float __x) throw(); 
# 122
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 128
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 131
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 134
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 141
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 144
extern float log2f(float __x) throw(); 
# 153
extern float powf(float __x, float __y) throw(); 
# 156
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 162
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 169
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 178
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 181
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 184
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 187
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 192
extern int __isinff(float __value) throw() __attribute((const)); 
# 195
extern int __finitef(float __value) throw() __attribute((const)); 
# 204
extern int isinff(float __value) throw() __attribute((const)); 
# 208
extern int finitef(float __value) throw() __attribute((const)); 
# 211
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 215
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 221
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 228
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanf(float __value) throw() __attribute((const)); 
# 241
extern int isnanf(float __value) throw() __attribute((const)); 
# 247
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 248
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 249
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 250
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 251
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 252
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 259
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 260
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 261
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 268
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 274
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 281
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 289
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 292
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 294
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 298
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 302
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 306
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 311
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 315
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 319
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 323
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 328
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 335
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 337
__extension__ extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 341
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 343
__extension__ extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 347
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 350
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 353
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyf(float __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitf(float __value) throw()
# 362
 __attribute((const)); 
# 366
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 375
extern int __issignalingf(float __value) throw()
# 376
 __attribute((const)); 
# 383
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 122
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 128
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 131
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 134
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 141
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 144
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 153
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 156
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 162
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 169
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 178
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 181
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 184
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 187
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 192
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 195
extern int __finitel(long double __value) throw() __attribute((const)); 
# 204
extern int isinfl(long double __value) throw() __attribute((const)); 
# 208
extern int finitel(long double __value) throw() __attribute((const)); 
# 211
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 215
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 221
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 228
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 241
extern int isnanl(long double __value) throw() __attribute((const)); 
# 247
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 248
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 249
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 250
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 251
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 252
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 259
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 260
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 261
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 268
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 274
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 281
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 289
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 292
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 294
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 298
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 302
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 306
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 311
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 315
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 319
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 323
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 328
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 335
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 337
__extension__ extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 341
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 343
__extension__ extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 347
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 350
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 353
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyl(long double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitl(long double __value) throw()
# 362
 __attribute((const)); 
# 366
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 375
extern int __issignalingl(long double __value) throw()
# 376
 __attribute((const)); 
# 383
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 168 "/usr/include/math.h" 3
extern int signgam; 
# 210 "/usr/include/math.h" 3
enum { 
# 211
FP_NAN, 
# 214
FP_INFINITE, 
# 217
FP_ZERO, 
# 220
FP_SUBNORMAL, 
# 223
FP_NORMAL
# 226
}; 
# 354 "/usr/include/math.h" 3
typedef 
# 348
enum { 
# 349
_IEEE_ = (-1), 
# 350
_SVID_ = 0, 
# 351
_XOPEN_, 
# 352
_POSIX_, 
# 353
_ISOC_
# 354
} _LIB_VERSION_TYPE; 
# 359
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 370 "/usr/include/math.h" 3
struct __exception { 
# 375
int type; 
# 376
char *name; 
# 377
double arg1; 
# 378
double arg2; 
# 379
double retval; 
# 380
}; 
# 383
extern int matherr(__exception * __exc) throw(); 
# 534 "/usr/include/math.h" 3
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 55 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
typedef 
# 51
enum { 
# 52
P_ALL, 
# 53
P_PID, 
# 54
P_PGID
# 55
} idtype_t; 
# 45 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 305 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 22 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 30
typedef 
# 28
struct { 
# 29
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 30
} __sigset_t; 
# 37 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef long __fd_mask; 
# 75 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" {
# 106 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
}
# 24 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" {
# 27
__extension__ extern unsigned gnu_dev_major(unsigned long long __dev) throw()
# 28
 __attribute((const)); 
# 30
__extension__ extern unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 31
 __attribute((const)); 
# 33
__extension__ extern unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 35
 __attribute((const)); 
# 58 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
}
# 228 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 128 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef 
# 91 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 125 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __data; 
# 126
char __size[40]; 
# 127
long __align; 
# 128
} pthread_mutex_t; 
# 134
typedef 
# 131
union { 
# 132
char __size[4]; 
# 133
int __align; 
# 134
} pthread_mutexattr_t; 
# 154
typedef 
# 140
union { 
# 142
struct { 
# 143
int __lock; 
# 144
unsigned __futex; 
# 145
__extension__ unsigned long long __total_seq; 
# 146
__extension__ unsigned long long __wakeup_seq; 
# 147
__extension__ unsigned long long __woken_seq; 
# 148
void *__mutex; 
# 149
unsigned __nwaiters; 
# 150
unsigned __broadcast_seq; 
# 151
} __data; 
# 152
char __size[48]; 
# 153
__extension__ long long __align; 
# 154
} pthread_cond_t; 
# 160
typedef 
# 157
union { 
# 158
char __size[4]; 
# 159
int __align; 
# 160
} pthread_condattr_t; 
# 164
typedef unsigned pthread_key_t; 
# 168
typedef int pthread_once_t; 
# 222 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef 
# 175 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union { 
# 178
struct { 
# 179
int __lock; 
# 180
unsigned __nr_readers; 
# 181
unsigned __readers_wakeup; 
# 182
unsigned __writer_wakeup; 
# 183
unsigned __nr_readers_queued; 
# 184
unsigned __nr_writers_queued; 
# 185
int __writer; 
# 186
int __shared; 
# 187
signed char __rwelision; 
# 192
unsigned char __pad1[7]; 
# 195
unsigned long __pad2; 
# 198
unsigned __flags; 
# 200
} __data; 
# 220 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[56]; 
# 221
long __align; 
# 222
} pthread_rwlock_t; 
# 228
typedef 
# 225
union { 
# 226
char __size[8]; 
# 227
long __align; 
# 228
} pthread_rwlockattr_t; 
# 234
typedef volatile int pthread_spinlock_t; 
# 243
typedef 
# 240
union { 
# 241
char __size[32]; 
# 242
long __align; 
# 243
} pthread_barrier_t; 
# 249
typedef 
# 246
union { 
# 247
char __size[4]; 
# 248
int __align; 
# 249
} pthread_barrierattr_t; 
# 273 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
__extension__ unsigned long long __a; 
# 420
}; 
# 423
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 424
 __attribute((__nonnull__(1, 2))); 
# 425
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 427
 __attribute((__nonnull__(1, 2))); 
# 430
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 432
 __attribute((__nonnull__(1, 2))); 
# 433
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 436
 __attribute((__nonnull__(1, 2))); 
# 439
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 441
 __attribute((__nonnull__(1, 2))); 
# 442
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 445
 __attribute((__nonnull__(1, 2))); 
# 448
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 449
 __attribute((__nonnull__(2))); 
# 451
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 454
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 456
 __attribute((__nonnull__(1, 2))); 
# 466
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 468
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 469
 __attribute((__malloc__)); 
# 480
extern void *realloc(void * __ptr, size_t __size) throw()
# 481
 __attribute((__warn_unused_result__)); 
# 483
extern void free(void * __ptr) throw(); 
# 488
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 498 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 503
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 504
 __attribute((__nonnull__(1))); 
# 509
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 510
 __attribute((__malloc__)) __attribute((__alloc_size__(2))); 
# 515
extern void abort() throw() __attribute((__noreturn__)); 
# 519
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 524
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 525
 __attribute((__nonnull__(1))); 
# 535
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 536
 __attribute((__nonnull__(1))); 
# 543
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 549
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 557
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 564
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 570
extern char *secure_getenv(const char * __name) throw()
# 571
 __attribute((__nonnull__(1))); 
# 578
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 584
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 585
 __attribute((__nonnull__(2))); 
# 588
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 595
extern int clearenv() throw(); 
# 606 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 764
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 765
 __attribute((__nonnull__(1, 4))); 
# 767
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 769
 __attribute((__nonnull__(1, 4))); 
# 774
extern int abs(int __x) throw() __attribute((const)); 
# 775
extern long labs(long __x) throw() __attribute((const)); 
# 779
__extension__ extern long long llabs(long long __x) throw()
# 780
 __attribute((const)); 
# 788
extern div_t div(int __numer, int __denom) throw()
# 789
 __attribute((const)); 
# 790
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 791
 __attribute((const)); 
# 796
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 798
 __attribute((const)); 
# 811 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 812
 __attribute((__nonnull__(3, 4))); 
# 817
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 818
 __attribute((__nonnull__(3, 4))); 
# 823
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 824
 __attribute((__nonnull__(3))); 
# 829
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 831
 __attribute((__nonnull__(3, 4))); 
# 832
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 834
 __attribute((__nonnull__(3, 4))); 
# 835
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 836
 __attribute((__nonnull__(3))); 
# 841
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 843
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 846
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 852
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 855
 __attribute((__nonnull__(3, 4, 5))); 
# 862
extern int mblen(const char * __s, size_t __n) throw(); 
# 865
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 869
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 873
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 876
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 887
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 898 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 901
 __attribute((__nonnull__(1, 2, 3))); 
# 907
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 915
extern int posix_openpt(int __oflag); 
# 923
extern int grantpt(int __fd) throw(); 
# 927
extern int unlockpt(int __fd) throw(); 
# 932
extern char *ptsname(int __fd) throw(); 
# 939
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 940
 __attribute((__nonnull__(2))); 
# 943
extern int getpt(); 
# 950
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 951
 __attribute((__nonnull__(1))); 
# 967 "/usr/include/stdlib.h" 3
}
# 194 "/usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h" 3
namespace std { 
# 196
typedef unsigned long size_t; 
# 197
typedef long ptrdiff_t; 
# 202
}
# 216 "/usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h" 3
namespace std { 
# 218
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 219
}
# 220
namespace __gnu_cxx { 
# 222
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 223
}
# 68 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 72
template< class _Iterator, class _Container> class __normal_iterator; 
# 76
}
# 78
namespace std __attribute((__visibility__("default"))) { 
# 82
struct __true_type { }; 
# 83
struct __false_type { }; 
# 85
template< bool > 
# 86
struct __truth_type { 
# 87
typedef __false_type __type; }; 
# 90
template<> struct __truth_type< true>  { 
# 91
typedef __true_type __type; }; 
# 95
template< class _Sp, class _Tp> 
# 96
struct __traitor { 
# 98
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 99
typedef typename __truth_type< __value> ::__type __type; 
# 100
}; 
# 103
template< class , class > 
# 104
struct __are_same { 
# 106
enum { __value}; 
# 107
typedef __false_type __type; 
# 108
}; 
# 110
template< class _Tp> 
# 111
struct __are_same< _Tp, _Tp>  { 
# 113
enum { __value = 1}; 
# 114
typedef __true_type __type; 
# 115
}; 
# 118
template< class _Tp> 
# 119
struct __is_void { 
# 121
enum { __value}; 
# 122
typedef __false_type __type; 
# 123
}; 
# 126
template<> struct __is_void< void>  { 
# 128
enum { __value = 1}; 
# 129
typedef __true_type __type; 
# 130
}; 
# 135
template< class _Tp> 
# 136
struct __is_integer { 
# 138
enum { __value}; 
# 139
typedef __false_type __type; 
# 140
}; 
# 147
template<> struct __is_integer< bool>  { 
# 149
enum { __value = 1}; 
# 150
typedef __true_type __type; 
# 151
}; 
# 154
template<> struct __is_integer< char>  { 
# 156
enum { __value = 1}; 
# 157
typedef __true_type __type; 
# 158
}; 
# 161
template<> struct __is_integer< signed char>  { 
# 163
enum { __value = 1}; 
# 164
typedef __true_type __type; 
# 165
}; 
# 168
template<> struct __is_integer< unsigned char>  { 
# 170
enum { __value = 1}; 
# 171
typedef __true_type __type; 
# 172
}; 
# 176
template<> struct __is_integer< wchar_t>  { 
# 178
enum { __value = 1}; 
# 179
typedef __true_type __type; 
# 180
}; 
# 200 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short>  { 
# 202
enum { __value = 1}; 
# 203
typedef __true_type __type; 
# 204
}; 
# 207
template<> struct __is_integer< unsigned short>  { 
# 209
enum { __value = 1}; 
# 210
typedef __true_type __type; 
# 211
}; 
# 214
template<> struct __is_integer< int>  { 
# 216
enum { __value = 1}; 
# 217
typedef __true_type __type; 
# 218
}; 
# 221
template<> struct __is_integer< unsigned>  { 
# 223
enum { __value = 1}; 
# 224
typedef __true_type __type; 
# 225
}; 
# 228
template<> struct __is_integer< long>  { 
# 230
enum { __value = 1}; 
# 231
typedef __true_type __type; 
# 232
}; 
# 235
template<> struct __is_integer< unsigned long>  { 
# 237
enum { __value = 1}; 
# 238
typedef __true_type __type; 
# 239
}; 
# 242
template<> struct __is_integer< long long>  { 
# 244
enum { __value = 1}; 
# 245
typedef __true_type __type; 
# 246
}; 
# 249
template<> struct __is_integer< unsigned long long>  { 
# 251
enum { __value = 1}; 
# 252
typedef __true_type __type; 
# 253
}; 
# 270 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128_t>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< __uint128_t>  { enum { __value = 1}; typedef __true_type __type; }; 
# 287 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 288
struct __is_floating { 
# 290
enum { __value}; 
# 291
typedef __false_type __type; 
# 292
}; 
# 296
template<> struct __is_floating< float>  { 
# 298
enum { __value = 1}; 
# 299
typedef __true_type __type; 
# 300
}; 
# 303
template<> struct __is_floating< double>  { 
# 305
enum { __value = 1}; 
# 306
typedef __true_type __type; 
# 307
}; 
# 310
template<> struct __is_floating< long double>  { 
# 312
enum { __value = 1}; 
# 313
typedef __true_type __type; 
# 314
}; 
# 319
template< class _Tp> 
# 320
struct __is_pointer { 
# 322
enum { __value}; 
# 323
typedef __false_type __type; 
# 324
}; 
# 326
template< class _Tp> 
# 327
struct __is_pointer< _Tp *>  { 
# 329
enum { __value = 1}; 
# 330
typedef __true_type __type; 
# 331
}; 
# 336
template< class _Tp> 
# 337
struct __is_normal_iterator { 
# 339
enum { __value}; 
# 340
typedef __false_type __type; 
# 341
}; 
# 343
template< class _Iterator, class _Container> 
# 344
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> >  { 
# 347
enum { __value = 1}; 
# 348
typedef __true_type __type; 
# 349
}; 
# 354
template< class _Tp> 
# 355
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 357
}; 
# 362
template< class _Tp> 
# 363
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 365
}; 
# 370
template< class _Tp> 
# 371
struct __is_char { 
# 373
enum { __value}; 
# 374
typedef __false_type __type; 
# 375
}; 
# 378
template<> struct __is_char< char>  { 
# 380
enum { __value = 1}; 
# 381
typedef __true_type __type; 
# 382
}; 
# 386
template<> struct __is_char< wchar_t>  { 
# 388
enum { __value = 1}; 
# 389
typedef __true_type __type; 
# 390
}; 
# 393
template< class _Tp> 
# 394
struct __is_byte { 
# 396
enum { __value}; 
# 397
typedef __false_type __type; 
# 398
}; 
# 401
template<> struct __is_byte< char>  { 
# 403
enum { __value = 1}; 
# 404
typedef __true_type __type; 
# 405
}; 
# 408
template<> struct __is_byte< signed char>  { 
# 410
enum { __value = 1}; 
# 411
typedef __true_type __type; 
# 412
}; 
# 415
template<> struct __is_byte< unsigned char>  { 
# 417
enum { __value = 1}; 
# 418
typedef __true_type __type; 
# 419
}; 
# 424
template< class _Tp> 
# 425
struct __is_move_iterator { 
# 427
enum { __value}; 
# 428
typedef __false_type __type; 
# 429
}; 
# 444 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
}
# 37 "/usr/include/c++/5/ext/type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 42
template< bool , class > 
# 43
struct __enable_if { 
# 44
}; 
# 46
template< class _Tp> 
# 47
struct __enable_if< true, _Tp>  { 
# 48
typedef _Tp __type; }; 
# 52
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 53
struct __conditional_type { 
# 54
typedef _Iftrue __type; }; 
# 56
template< class _Iftrue, class _Iffalse> 
# 57
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 58
typedef _Iffalse __type; }; 
# 62
template< class _Tp> 
# 63
struct __add_unsigned { 
# 66
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 69
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 70
}; 
# 73
template<> struct __add_unsigned< char>  { 
# 74
typedef unsigned char __type; }; 
# 77
template<> struct __add_unsigned< signed char>  { 
# 78
typedef unsigned char __type; }; 
# 81
template<> struct __add_unsigned< short>  { 
# 82
typedef unsigned short __type; }; 
# 85
template<> struct __add_unsigned< int>  { 
# 86
typedef unsigned __type; }; 
# 89
template<> struct __add_unsigned< long>  { 
# 90
typedef unsigned long __type; }; 
# 93
template<> struct __add_unsigned< long long>  { 
# 94
typedef unsigned long long __type; }; 
# 98
template<> struct __add_unsigned< bool> ; 
# 101
template<> struct __add_unsigned< wchar_t> ; 
# 105
template< class _Tp> 
# 106
struct __remove_unsigned { 
# 109
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 112
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 113
}; 
# 116
template<> struct __remove_unsigned< char>  { 
# 117
typedef signed char __type; }; 
# 120
template<> struct __remove_unsigned< unsigned char>  { 
# 121
typedef signed char __type; }; 
# 124
template<> struct __remove_unsigned< unsigned short>  { 
# 125
typedef short __type; }; 
# 128
template<> struct __remove_unsigned< unsigned>  { 
# 129
typedef int __type; }; 
# 132
template<> struct __remove_unsigned< unsigned long>  { 
# 133
typedef long __type; }; 
# 136
template<> struct __remove_unsigned< unsigned long long>  { 
# 137
typedef long long __type; }; 
# 141
template<> struct __remove_unsigned< bool> ; 
# 144
template<> struct __remove_unsigned< wchar_t> ; 
# 148
template< class _Type> inline bool 
# 150
__is_null_pointer(_Type *__ptr) 
# 151
{ return __ptr == 0; } 
# 153
template< class _Type> inline bool 
# 155
__is_null_pointer(_Type) 
# 156
{ return false; } 
# 165 "/usr/include/c++/5/ext/type_traits.h" 3
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 166
struct __promote { 
# 167
typedef double __type; }; 
# 172
template< class _Tp> 
# 173
struct __promote< _Tp, false>  { 
# 174
}; 
# 177
template<> struct __promote< long double>  { 
# 178
typedef long double __type; }; 
# 181
template<> struct __promote< double>  { 
# 182
typedef double __type; }; 
# 185
template<> struct __promote< float>  { 
# 186
typedef float __type; }; 
# 188
template< class _Tp, class _Up, class 
# 189
_Tp2 = typename __promote< _Tp> ::__type, class 
# 190
_Up2 = typename __promote< _Up> ::__type> 
# 191
struct __promote_2 { 
# 193
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 194
}; 
# 196
template< class _Tp, class _Up, class _Vp, class 
# 197
_Tp2 = typename __promote< _Tp> ::__type, class 
# 198
_Up2 = typename __promote< _Up> ::__type, class 
# 199
_Vp2 = typename __promote< _Vp> ::__type> 
# 200
struct __promote_3 { 
# 202
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 203
}; 
# 205
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 206
_Tp2 = typename __promote< _Tp> ::__type, class 
# 207
_Up2 = typename __promote< _Up> ::__type, class 
# 208
_Vp2 = typename __promote< _Vp> ::__type, class 
# 209
_Wp2 = typename __promote< _Wp> ::__type> 
# 210
struct __promote_4 { 
# 212
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 213
}; 
# 216
}
# 75 "/usr/include/c++/5/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 81
inline double abs(double __x) 
# 82
{ return __builtin_fabs(__x); } 
# 87
inline float abs(float __x) 
# 88
{ return __builtin_fabsf(__x); } 
# 91
inline long double abs(long double __x) 
# 92
{ return __builtin_fabsl(__x); } 
# 95
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
abs(_Tp __x) 
# 100
{ return __builtin_fabs(__x); } 
# 102
using ::acos;
# 106
inline float acos(float __x) 
# 107
{ return __builtin_acosf(__x); } 
# 110
inline long double acos(long double __x) 
# 111
{ return __builtin_acosl(__x); } 
# 114
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
acos(_Tp __x) 
# 119
{ return __builtin_acos(__x); } 
# 121
using ::asin;
# 125
inline float asin(float __x) 
# 126
{ return __builtin_asinf(__x); } 
# 129
inline long double asin(long double __x) 
# 130
{ return __builtin_asinl(__x); } 
# 133
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
asin(_Tp __x) 
# 138
{ return __builtin_asin(__x); } 
# 140
using ::atan;
# 144
inline float atan(float __x) 
# 145
{ return __builtin_atanf(__x); } 
# 148
inline long double atan(long double __x) 
# 149
{ return __builtin_atanl(__x); } 
# 152
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 156
atan(_Tp __x) 
# 157
{ return __builtin_atan(__x); } 
# 159
using ::atan2;
# 163
inline float atan2(float __y, float __x) 
# 164
{ return __builtin_atan2f(__y, __x); } 
# 167
inline long double atan2(long double __y, long double __x) 
# 168
{ return __builtin_atan2l(__y, __x); } 
# 171
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 174
atan2(_Tp __y, _Up __x) 
# 175
{ 
# 176
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 177
return atan2((__type)__y, (__type)__x); 
# 178
} 
# 180
using ::ceil;
# 184
inline float ceil(float __x) 
# 185
{ return __builtin_ceilf(__x); } 
# 188
inline long double ceil(long double __x) 
# 189
{ return __builtin_ceill(__x); } 
# 192
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
ceil(_Tp __x) 
# 197
{ return __builtin_ceil(__x); } 
# 199
using ::cos;
# 203
inline float cos(float __x) 
# 204
{ return __builtin_cosf(__x); } 
# 207
inline long double cos(long double __x) 
# 208
{ return __builtin_cosl(__x); } 
# 211
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cos(_Tp __x) 
# 216
{ return __builtin_cos(__x); } 
# 218
using ::cosh;
# 222
inline float cosh(float __x) 
# 223
{ return __builtin_coshf(__x); } 
# 226
inline long double cosh(long double __x) 
# 227
{ return __builtin_coshl(__x); } 
# 230
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
cosh(_Tp __x) 
# 235
{ return __builtin_cosh(__x); } 
# 237
using ::exp;
# 241
inline float exp(float __x) 
# 242
{ return __builtin_expf(__x); } 
# 245
inline long double exp(long double __x) 
# 246
{ return __builtin_expl(__x); } 
# 249
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
exp(_Tp __x) 
# 254
{ return __builtin_exp(__x); } 
# 256
using ::fabs;
# 260
inline float fabs(float __x) 
# 261
{ return __builtin_fabsf(__x); } 
# 264
inline long double fabs(long double __x) 
# 265
{ return __builtin_fabsl(__x); } 
# 268
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
fabs(_Tp __x) 
# 273
{ return __builtin_fabs(__x); } 
# 275
using ::floor;
# 279
inline float floor(float __x) 
# 280
{ return __builtin_floorf(__x); } 
# 283
inline long double floor(long double __x) 
# 284
{ return __builtin_floorl(__x); } 
# 287
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 291
floor(_Tp __x) 
# 292
{ return __builtin_floor(__x); } 
# 294
using ::fmod;
# 298
inline float fmod(float __x, float __y) 
# 299
{ return __builtin_fmodf(__x, __y); } 
# 302
inline long double fmod(long double __x, long double __y) 
# 303
{ return __builtin_fmodl(__x, __y); } 
# 306
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 309
fmod(_Tp __x, _Up __y) 
# 310
{ 
# 311
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 312
return fmod((__type)__x, (__type)__y); 
# 313
} 
# 315
using ::frexp;
# 319
inline float frexp(float __x, int *__exp) 
# 320
{ return __builtin_frexpf(__x, __exp); } 
# 323
inline long double frexp(long double __x, int *__exp) 
# 324
{ return __builtin_frexpl(__x, __exp); } 
# 327
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
frexp(_Tp __x, int *__exp) 
# 332
{ return __builtin_frexp(__x, __exp); } 
# 334
using ::ldexp;
# 338
inline float ldexp(float __x, int __exp) 
# 339
{ return __builtin_ldexpf(__x, __exp); } 
# 342
inline long double ldexp(long double __x, int __exp) 
# 343
{ return __builtin_ldexpl(__x, __exp); } 
# 346
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
ldexp(_Tp __x, int __exp) 
# 351
{ return __builtin_ldexp(__x, __exp); } 
# 353
using ::log;
# 357
inline float log(float __x) 
# 358
{ return __builtin_logf(__x); } 
# 361
inline long double log(long double __x) 
# 362
{ return __builtin_logl(__x); } 
# 365
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log(_Tp __x) 
# 370
{ return __builtin_log(__x); } 
# 372
using ::log10;
# 376
inline float log10(float __x) 
# 377
{ return __builtin_log10f(__x); } 
# 380
inline long double log10(long double __x) 
# 381
{ return __builtin_log10l(__x); } 
# 384
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 388
log10(_Tp __x) 
# 389
{ return __builtin_log10(__x); } 
# 391
using ::modf;
# 395
inline float modf(float __x, float *__iptr) 
# 396
{ return __builtin_modff(__x, __iptr); } 
# 399
inline long double modf(long double __x, long double *__iptr) 
# 400
{ return __builtin_modfl(__x, __iptr); } 
# 403
using ::pow;
# 407
inline float pow(float __x, float __y) 
# 408
{ return __builtin_powf(__x, __y); } 
# 411
inline long double pow(long double __x, long double __y) 
# 412
{ return __builtin_powl(__x, __y); } 
# 418
inline double pow(double __x, int __i) 
# 419
{ return __builtin_powi(__x, __i); } 
# 422
inline float pow(float __x, int __n) 
# 423
{ return __builtin_powif(__x, __n); } 
# 426
inline long double pow(long double __x, int __n) 
# 427
{ return __builtin_powil(__x, __n); } 
# 431
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 434
pow(_Tp __x, _Up __y) 
# 435
{ 
# 436
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 437
return pow((__type)__x, (__type)__y); 
# 438
} 
# 440
using ::sin;
# 444
inline float sin(float __x) 
# 445
{ return __builtin_sinf(__x); } 
# 448
inline long double sin(long double __x) 
# 449
{ return __builtin_sinl(__x); } 
# 452
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sin(_Tp __x) 
# 457
{ return __builtin_sin(__x); } 
# 459
using ::sinh;
# 463
inline float sinh(float __x) 
# 464
{ return __builtin_sinhf(__x); } 
# 467
inline long double sinh(long double __x) 
# 468
{ return __builtin_sinhl(__x); } 
# 471
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sinh(_Tp __x) 
# 476
{ return __builtin_sinh(__x); } 
# 478
using ::sqrt;
# 482
inline float sqrt(float __x) 
# 483
{ return __builtin_sqrtf(__x); } 
# 486
inline long double sqrt(long double __x) 
# 487
{ return __builtin_sqrtl(__x); } 
# 490
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
sqrt(_Tp __x) 
# 495
{ return __builtin_sqrt(__x); } 
# 497
using ::tan;
# 501
inline float tan(float __x) 
# 502
{ return __builtin_tanf(__x); } 
# 505
inline long double tan(long double __x) 
# 506
{ return __builtin_tanl(__x); } 
# 509
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tan(_Tp __x) 
# 514
{ return __builtin_tan(__x); } 
# 516
using ::tanh;
# 520
inline float tanh(float __x) 
# 521
{ return __builtin_tanhf(__x); } 
# 524
inline long double tanh(long double __x) 
# 525
{ return __builtin_tanhl(__x); } 
# 528
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 532
tanh(_Tp __x) 
# 533
{ return __builtin_tanh(__x); } 
# 536
}
# 555 "/usr/include/c++/5/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 831 "/usr/include/c++/5/cmath" 3
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 834
fpclassify(_Tp __f) 
# 835
{ 
# 836
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 837
return __builtin_fpclassify(0, 1, 4, 3, 2, (__type)__f); 
# 839
} 
# 841
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 844
isfinite(_Tp __f) 
# 845
{ 
# 846
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 847
return __builtin_isfinite((__type)__f); 
# 848
} 
# 850
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 853
isinf(_Tp __f) 
# 854
{ 
# 855
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 856
return __builtin_isinf((__type)__f); 
# 857
} 
# 859
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 862
isnan(_Tp __f) 
# 863
{ 
# 864
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 865
return __builtin_isnan((__type)__f); 
# 866
} 
# 868
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 871
isnormal(_Tp __f) 
# 872
{ 
# 873
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 874
return __builtin_isnormal((__type)__f); 
# 875
} 
# 877
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 880
signbit(_Tp __f) 
# 881
{ 
# 882
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 883
return (sizeof(__type) == sizeof(float)) ? __builtin_signbitf((__type)__f) : ((sizeof(__type) == sizeof(double)) ? __builtin_signbit((__type)__f) : __builtin_signbitl((__type)__f)); 
# 888
} 
# 890
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 893
isgreater(_Tp __f1, _Tp __f2) 
# 894
{ 
# 895
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 896
return __builtin_isgreater((__type)__f1, (__type)__f2); 
# 897
} 
# 899
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 902
isgreaterequal(_Tp __f1, _Tp __f2) 
# 903
{ 
# 904
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 905
return __builtin_isgreaterequal((__type)__f1, (__type)__f2); 
# 906
} 
# 908
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 911
isless(_Tp __f1, _Tp __f2) 
# 912
{ 
# 913
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 914
return __builtin_isless((__type)__f1, (__type)__f2); 
# 915
} 
# 917
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 920
islessequal(_Tp __f1, _Tp __f2) 
# 921
{ 
# 922
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 923
return __builtin_islessequal((__type)__f1, (__type)__f2); 
# 924
} 
# 926
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 929
islessgreater(_Tp __f1, _Tp __f2) 
# 930
{ 
# 931
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 932
return __builtin_islessgreater((__type)__f1, (__type)__f2); 
# 933
} 
# 935
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 938
isunordered(_Tp __f1, _Tp __f2) 
# 939
{ 
# 940
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 941
return __builtin_isunordered((__type)__f1, (__type)__f2); 
# 942
} 
# 947
}
# 114 "/usr/include/c++/5/cstdlib" 3
namespace std __attribute((__visibility__("default"))) { 
# 118
using ::div_t;
# 119
using ::ldiv_t;
# 121
using ::abort;
# 122
using ::abs;
# 123
using ::atexit;
# 129
using ::atof;
# 130
using ::atoi;
# 131
using ::atol;
# 132
using ::bsearch;
# 133
using ::calloc;
# 134
using ::div;
# 135
using ::exit;
# 136
using ::free;
# 137
using ::getenv;
# 138
using ::labs;
# 139
using ::ldiv;
# 140
using ::malloc;
# 142
using ::mblen;
# 143
using ::mbstowcs;
# 144
using ::mbtowc;
# 146
using ::qsort;
# 152
using ::rand;
# 153
using ::realloc;
# 154
using ::srand;
# 155
using ::strtod;
# 156
using ::strtol;
# 157
using ::strtoul;
# 158
using ::system;
# 160
using ::wcstombs;
# 161
using ::wctomb;
# 166
inline long abs(long __i) { return __builtin_labs(__i); } 
# 169
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 174
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 179
inline __int128_t abs(__int128_t __x) { return (__x >= (0)) ? __x : (-__x); } 
# 196 "/usr/include/c++/5/cstdlib" 3
}
# 209 "/usr/include/c++/5/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 214
using ::lldiv_t;
# 220
using ::_Exit;
# 224
using ::llabs;
# 227
inline lldiv_t div(long long __n, long long __d) 
# 228
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 230
using ::lldiv;
# 241 "/usr/include/c++/5/cstdlib" 3
using ::atoll;
# 242
using ::strtoll;
# 243
using ::strtoull;
# 245
using ::strtof;
# 246
using ::strtold;
# 249
}
# 251
namespace std { 
# 254
using __gnu_cxx::lldiv_t;
# 256
using __gnu_cxx::_Exit;
# 258
using __gnu_cxx::llabs;
# 259
using __gnu_cxx::div;
# 260
using __gnu_cxx::lldiv;
# 262
using __gnu_cxx::atoll;
# 263
using __gnu_cxx::strtof;
# 264
using __gnu_cxx::strtoll;
# 265
using __gnu_cxx::strtoull;
# 266
using __gnu_cxx::strtold;
# 267
}
# 8970 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
__attribute((always_inline)) inline int signbit(float x); 
# 8974
__attribute((always_inline)) inline int signbit(double x); 
# 8976
__attribute((always_inline)) inline int signbit(long double x); 
# 8978
__attribute((always_inline)) inline int isfinite(float x); 
# 8982
__attribute((always_inline)) inline int isfinite(double x); 
# 8984
__attribute((always_inline)) inline int isfinite(long double x); 
# 8986
__attribute((always_inline)) inline int isnan(float x); 
# 8993
extern "C" __attribute((always_inline)) inline int isnan(double x) throw(); 
# 8995
__attribute((always_inline)) inline int isnan(long double x); 
# 8997
__attribute((always_inline)) inline int isinf(float x); 
# 9004
extern "C" __attribute((always_inline)) inline int isinf(double x) throw(); 
# 9006
__attribute((always_inline)) inline int isinf(long double x); 
# 9063 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
namespace std { 
# 9065
template< class T> extern T __pow_helper(T, int); 
# 9066
template< class T> extern T __cmath_power(T, unsigned); 
# 9067
}
# 9069
using std::abs;
# 9070
using std::fabs;
# 9071
using std::ceil;
# 9072
using std::floor;
# 9073
using std::sqrt;
# 9075
using std::pow;
# 9077
using std::log;
# 9078
using std::log10;
# 9079
using std::fmod;
# 9080
using std::modf;
# 9081
using std::exp;
# 9082
using std::frexp;
# 9083
using std::ldexp;
# 9084
using std::asin;
# 9085
using std::sin;
# 9086
using std::sinh;
# 9087
using std::acos;
# 9088
using std::cos;
# 9089
using std::cosh;
# 9090
using std::atan;
# 9091
using std::atan2;
# 9092
using std::tan;
# 9093
using std::tanh;
# 9458 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
namespace std { 
# 9467 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern inline long long abs(long long); 
# 9477 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern inline long abs(long); 
# 9478
extern inline float abs(float); 
# 9479
extern inline double abs(double); 
# 9480
extern inline float fabs(float); 
# 9481
extern inline float ceil(float); 
# 9482
extern inline float floor(float); 
# 9483
extern inline float sqrt(float); 
# 9484
extern inline float pow(float, float); 
# 9493 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
extern inline float pow(float, int); 
# 9494
extern inline double pow(double, int); 
# 9499
extern inline float log(float); 
# 9500
extern inline float log10(float); 
# 9501
extern inline float fmod(float, float); 
# 9502
extern inline float modf(float, float *); 
# 9503
extern inline float exp(float); 
# 9504
extern inline float frexp(float, int *); 
# 9505
extern inline float ldexp(float, int); 
# 9506
extern inline float asin(float); 
# 9507
extern inline float sin(float); 
# 9508
extern inline float sinh(float); 
# 9509
extern inline float acos(float); 
# 9510
extern inline float cos(float); 
# 9511
extern inline float cosh(float); 
# 9512
extern inline float atan(float); 
# 9513
extern inline float atan2(float, float); 
# 9514
extern inline float tan(float); 
# 9515
extern inline float tanh(float); 
# 9589 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
}
# 9732 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
static inline float logb(float a); 
# 9734
static inline int ilogb(float a); 
# 9736
static inline float scalbn(float a, int b); 
# 9738
static inline float scalbln(float a, long b); 
# 9740
static inline float exp2(float a); 
# 9742
static inline float expm1(float a); 
# 9744
static inline float log2(float a); 
# 9746
static inline float log1p(float a); 
# 9748
static inline float acosh(float a); 
# 9750
static inline float asinh(float a); 
# 9752
static inline float atanh(float a); 
# 9754
static inline float hypot(float a, float b); 
# 9756
static inline float norm3d(float a, float b, float c); 
# 9758
static inline float norm4d(float a, float b, float c, float d); 
# 9760
static inline float cbrt(float a); 
# 9762
static inline float erf(float a); 
# 9764
static inline float erfc(float a); 
# 9766
static inline float lgamma(float a); 
# 9768
static inline float tgamma(float a); 
# 9770
static inline float copysign(float a, float b); 
# 9772
static inline float nextafter(float a, float b); 
# 9774
static inline float remainder(float a, float b); 
# 9776
static inline float remquo(float a, float b, int * quo); 
# 9778
static inline float round(float a); 
# 9780
static inline long lround(float a); 
# 9782
static inline long long llround(float a); 
# 9784
static inline float trunc(float a); 
# 9786
static inline float rint(float a); 
# 9788
static inline long lrint(float a); 
# 9790
static inline long long llrint(float a); 
# 9792
static inline float nearbyint(float a); 
# 9794
static inline float fdim(float a, float b); 
# 9796
static inline float fma(float a, float b, float c); 
# 9798
static inline float fmax(float a, float b); 
# 9800
static inline float fmin(float a, float b); 
# 9841 "/usr/local/cuda/bin/..//include/crt/math_functions.h"
static inline float exp10(float a); 
# 9843
static inline float rsqrt(float a); 
# 9845
static inline float rcbrt(float a); 
# 9847
static inline float sinpi(float a); 
# 9849
static inline float cospi(float a); 
# 9851
static inline void sincospi(float a, float * sptr, float * cptr); 
# 9853
static inline void sincos(float a, float * sptr, float * cptr); 
# 9855
static inline float j0(float a); 
# 9857
static inline float j1(float a); 
# 9859
static inline float jn(int n, float a); 
# 9861
static inline float y0(float a); 
# 9863
static inline float y1(float a); 
# 9865
static inline float yn(int n, float a); 
# 9867
static inline float cyl_bessel_i0(float a); 
# 9869
static inline float cyl_bessel_i1(float a); 
# 9871
static inline float erfinv(float a); 
# 9873
static inline float erfcinv(float a); 
# 9875
static inline float normcdfinv(float a); 
# 9877
static inline float normcdf(float a); 
# 9879
static inline float erfcx(float a); 
# 9881
static inline double copysign(double a, float b); 
# 9883
static inline double copysign(float a, double b); 
# 9885
static inline unsigned min(unsigned a, unsigned b); 
# 9887
static inline unsigned min(int a, unsigned b); 
# 9889
static inline unsigned min(unsigned a, int b); 
# 9891
static inline long min(long a, long b); 
# 9893
static inline unsigned long min(unsigned long a, unsigned long b); 
# 9895
static inline unsigned long min(long a, unsigned long b); 
# 9897
static inline unsigned long min(unsigned long a, long b); 
# 9899
static inline long long min(long long a, long long b); 
# 9901
static inline unsigned long long min(unsigned long long a, unsigned long long b); 
# 9903
static inline unsigned long long min(long long a, unsigned long long b); 
# 9905
static inline unsigned long long min(unsigned long long a, long long b); 
# 9907
static inline float min(float a, float b); 
# 9909
static inline double min(double a, double b); 
# 9911
static inline double min(float a, double b); 
# 9913
static inline double min(double a, float b); 
# 9915
static inline unsigned max(unsigned a, unsigned b); 
# 9917
static inline unsigned max(int a, unsigned b); 
# 9919
static inline unsigned max(unsigned a, int b); 
# 9921
static inline long max(long a, long b); 
# 9923
static inline unsigned long max(unsigned long a, unsigned long b); 
# 9925
static inline unsigned long max(long a, unsigned long b); 
# 9927
static inline unsigned long max(unsigned long a, long b); 
# 9929
static inline long long max(long long a, long long b); 
# 9931
static inline unsigned long long max(unsigned long long a, unsigned long long b); 
# 9933
static inline unsigned long long max(long long a, unsigned long long b); 
# 9935
static inline unsigned long long max(unsigned long long a, long long b); 
# 9937
static inline float max(float a, float b); 
# 9939
static inline double max(double a, double b); 
# 9941
static inline double max(float a, double b); 
# 9943
static inline double max(double a, float b); 
# 316 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
__attribute((always_inline)) inline int signbit(float x) { return __signbitf(x); } 
# 320
__attribute((always_inline)) inline int signbit(double x) { return __signbit(x); } 
# 322
__attribute((always_inline)) inline int signbit(long double x) { return __signbitl(x); } 
# 333 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(float x) { return __finitef(x); } 
# 348 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(double x) { return __finite(x); } 
# 361 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(long double x) { return __finitel(x); } 
# 364
__attribute((always_inline)) inline int isnan(float x) { return __isnanf(x); } 
# 368
__attribute((always_inline)) inline int isnan(double x) throw() { return __isnan(x); } 
# 370
__attribute((always_inline)) inline int isnan(long double x) { return __isnanl(x); } 
# 372
__attribute((always_inline)) inline int isinf(float x) { return __isinff(x); } 
# 376
__attribute((always_inline)) inline int isinf(double x) throw() { return __isinf(x); } 
# 378
__attribute((always_inline)) inline int isinf(long double x) { return __isinfl(x); } 
# 584 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
static inline float logb(float a) 
# 585
{ 
# 586
return logbf(a); 
# 587
} 
# 589
static inline int ilogb(float a) 
# 590
{ 
# 591
return ilogbf(a); 
# 592
} 
# 594
static inline float scalbn(float a, int b) 
# 595
{ 
# 596
return scalbnf(a, b); 
# 597
} 
# 599
static inline float scalbln(float a, long b) 
# 600
{ 
# 601
return scalblnf(a, b); 
# 602
} 
# 604
static inline float exp2(float a) 
# 605
{ 
# 606
return exp2f(a); 
# 607
} 
# 609
static inline float expm1(float a) 
# 610
{ 
# 611
return expm1f(a); 
# 612
} 
# 614
static inline float log2(float a) 
# 615
{ 
# 616
return log2f(a); 
# 617
} 
# 619
static inline float log1p(float a) 
# 620
{ 
# 621
return log1pf(a); 
# 622
} 
# 624
static inline float acosh(float a) 
# 625
{ 
# 626
return acoshf(a); 
# 627
} 
# 629
static inline float asinh(float a) 
# 630
{ 
# 631
return asinhf(a); 
# 632
} 
# 634
static inline float atanh(float a) 
# 635
{ 
# 636
return atanhf(a); 
# 637
} 
# 639
static inline float hypot(float a, float b) 
# 640
{ 
# 641
return hypotf(a, b); 
# 642
} 
# 644
static inline float norm3d(float a, float b, float c) 
# 645
{ 
# 646
return norm3df(a, b, c); 
# 647
} 
# 649
static inline float norm4d(float a, float b, float c, float d) 
# 650
{ 
# 651
return norm4df(a, b, c, d); 
# 652
} 
# 654
static inline float cbrt(float a) 
# 655
{ 
# 656
return cbrtf(a); 
# 657
} 
# 659
static inline float erf(float a) 
# 660
{ 
# 661
return erff(a); 
# 662
} 
# 664
static inline float erfc(float a) 
# 665
{ 
# 666
return erfcf(a); 
# 667
} 
# 669
static inline float lgamma(float a) 
# 670
{ 
# 671
return lgammaf(a); 
# 672
} 
# 674
static inline float tgamma(float a) 
# 675
{ 
# 676
return tgammaf(a); 
# 677
} 
# 679
static inline float copysign(float a, float b) 
# 680
{ 
# 681
return copysignf(a, b); 
# 682
} 
# 684
static inline float nextafter(float a, float b) 
# 685
{ 
# 686
return nextafterf(a, b); 
# 687
} 
# 689
static inline float remainder(float a, float b) 
# 690
{ 
# 691
return remainderf(a, b); 
# 692
} 
# 694
static inline float remquo(float a, float b, int *quo) 
# 695
{ 
# 696
return remquof(a, b, quo); 
# 697
} 
# 699
static inline float round(float a) 
# 700
{ 
# 701
return roundf(a); 
# 702
} 
# 704
static inline long lround(float a) 
# 705
{ 
# 706
return lroundf(a); 
# 707
} 
# 709
static inline long long llround(float a) 
# 710
{ 
# 711
return llroundf(a); 
# 712
} 
# 714
static inline float trunc(float a) 
# 715
{ 
# 716
return truncf(a); 
# 717
} 
# 719
static inline float rint(float a) 
# 720
{ 
# 721
return rintf(a); 
# 722
} 
# 724
static inline long lrint(float a) 
# 725
{ 
# 726
return lrintf(a); 
# 727
} 
# 729
static inline long long llrint(float a) 
# 730
{ 
# 731
return llrintf(a); 
# 732
} 
# 734
static inline float nearbyint(float a) 
# 735
{ 
# 736
return nearbyintf(a); 
# 737
} 
# 739
static inline float fdim(float a, float b) 
# 740
{ 
# 741
return fdimf(a, b); 
# 742
} 
# 744
static inline float fma(float a, float b, float c) 
# 745
{ 
# 746
return fmaf(a, b, c); 
# 747
} 
# 749
static inline float fmax(float a, float b) 
# 750
{ 
# 751
return fmaxf(a, b); 
# 752
} 
# 754
static inline float fmin(float a, float b) 
# 755
{ 
# 756
return fminf(a, b); 
# 757
} 
# 765
static inline float exp10(float a) 
# 766
{ 
# 767
return exp10f(a); 
# 768
} 
# 770
static inline float rsqrt(float a) 
# 771
{ 
# 772
return rsqrtf(a); 
# 773
} 
# 775
static inline float rcbrt(float a) 
# 776
{ 
# 777
return rcbrtf(a); 
# 778
} 
# 780
static inline float sinpi(float a) 
# 781
{ 
# 782
return sinpif(a); 
# 783
} 
# 785
static inline float cospi(float a) 
# 786
{ 
# 787
return cospif(a); 
# 788
} 
# 790
static inline void sincospi(float a, float *sptr, float *cptr) 
# 791
{ 
# 792
sincospif(a, sptr, cptr); 
# 793
} 
# 795
static inline void sincos(float a, float *sptr, float *cptr) 
# 796
{ 
# 797
sincosf(a, sptr, cptr); 
# 798
} 
# 800
static inline float j0(float a) 
# 801
{ 
# 802
return j0f(a); 
# 803
} 
# 805
static inline float j1(float a) 
# 806
{ 
# 807
return j1f(a); 
# 808
} 
# 810
static inline float jn(int n, float a) 
# 811
{ 
# 812
return jnf(n, a); 
# 813
} 
# 815
static inline float y0(float a) 
# 816
{ 
# 817
return y0f(a); 
# 818
} 
# 820
static inline float y1(float a) 
# 821
{ 
# 822
return y1f(a); 
# 823
} 
# 825
static inline float yn(int n, float a) 
# 826
{ 
# 827
return ynf(n, a); 
# 828
} 
# 830
static inline float cyl_bessel_i0(float a) 
# 831
{ 
# 832
return cyl_bessel_i0f(a); 
# 833
} 
# 835
static inline float cyl_bessel_i1(float a) 
# 836
{ 
# 837
return cyl_bessel_i1f(a); 
# 838
} 
# 840
static inline float erfinv(float a) 
# 841
{ 
# 842
return erfinvf(a); 
# 843
} 
# 845
static inline float erfcinv(float a) 
# 846
{ 
# 847
return erfcinvf(a); 
# 848
} 
# 850
static inline float normcdfinv(float a) 
# 851
{ 
# 852
return normcdfinvf(a); 
# 853
} 
# 855
static inline float normcdf(float a) 
# 856
{ 
# 857
return normcdff(a); 
# 858
} 
# 860
static inline float erfcx(float a) 
# 861
{ 
# 862
return erfcxf(a); 
# 863
} 
# 865
static inline double copysign(double a, float b) 
# 866
{ 
# 867
return copysign(a, (double)b); 
# 868
} 
# 870
static inline double copysign(float a, double b) 
# 871
{ 
# 872
return copysign((double)a, b); 
# 873
} 
# 875
static inline unsigned min(unsigned a, unsigned b) 
# 876
{ 
# 877
return umin(a, b); 
# 878
} 
# 880
static inline unsigned min(int a, unsigned b) 
# 881
{ 
# 882
return umin((unsigned)a, b); 
# 883
} 
# 885
static inline unsigned min(unsigned a, int b) 
# 886
{ 
# 887
return umin(a, (unsigned)b); 
# 888
} 
# 890
static inline long min(long a, long b) 
# 891
{ 
# 897
if (sizeof(long) == sizeof(int)) { 
# 901
return (long)min((int)a, (int)b); 
# 902
} else { 
# 903
return (long)llmin((long long)a, (long long)b); 
# 904
}  
# 905
} 
# 907
static inline unsigned long min(unsigned long a, unsigned long b) 
# 908
{ 
# 912
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 916
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 917
} else { 
# 918
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 919
}  
# 920
} 
# 922
static inline unsigned long min(long a, unsigned long b) 
# 923
{ 
# 927
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 931
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 932
} else { 
# 933
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 934
}  
# 935
} 
# 937
static inline unsigned long min(unsigned long a, long b) 
# 938
{ 
# 942
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 946
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 947
} else { 
# 948
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 949
}  
# 950
} 
# 952
static inline long long min(long long a, long long b) 
# 953
{ 
# 954
return llmin(a, b); 
# 955
} 
# 957
static inline unsigned long long min(unsigned long long a, unsigned long long b) 
# 958
{ 
# 959
return ullmin(a, b); 
# 960
} 
# 962
static inline unsigned long long min(long long a, unsigned long long b) 
# 963
{ 
# 964
return ullmin((unsigned long long)a, b); 
# 965
} 
# 967
static inline unsigned long long min(unsigned long long a, long long b) 
# 968
{ 
# 969
return ullmin(a, (unsigned long long)b); 
# 970
} 
# 972
static inline float min(float a, float b) 
# 973
{ 
# 974
return fminf(a, b); 
# 975
} 
# 977
static inline double min(double a, double b) 
# 978
{ 
# 979
return fmin(a, b); 
# 980
} 
# 982
static inline double min(float a, double b) 
# 983
{ 
# 984
return fmin((double)a, b); 
# 985
} 
# 987
static inline double min(double a, float b) 
# 988
{ 
# 989
return fmin(a, (double)b); 
# 990
} 
# 992
static inline unsigned max(unsigned a, unsigned b) 
# 993
{ 
# 994
return umax(a, b); 
# 995
} 
# 997
static inline unsigned max(int a, unsigned b) 
# 998
{ 
# 999
return umax((unsigned)a, b); 
# 1000
} 
# 1002
static inline unsigned max(unsigned a, int b) 
# 1003
{ 
# 1004
return umax(a, (unsigned)b); 
# 1005
} 
# 1007
static inline long max(long a, long b) 
# 1008
{ 
# 1013
if (sizeof(long) == sizeof(int)) { 
# 1017
return (long)max((int)a, (int)b); 
# 1018
} else { 
# 1019
return (long)llmax((long long)a, (long long)b); 
# 1020
}  
# 1021
} 
# 1023
static inline unsigned long max(unsigned long a, unsigned long b) 
# 1024
{ 
# 1028
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1032
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1033
} else { 
# 1034
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1035
}  
# 1036
} 
# 1038
static inline unsigned long max(long a, unsigned long b) 
# 1039
{ 
# 1043
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1047
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1048
} else { 
# 1049
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1050
}  
# 1051
} 
# 1053
static inline unsigned long max(unsigned long a, long b) 
# 1054
{ 
# 1058
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1062
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1063
} else { 
# 1064
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1065
}  
# 1066
} 
# 1068
static inline long long max(long long a, long long b) 
# 1069
{ 
# 1070
return llmax(a, b); 
# 1071
} 
# 1073
static inline unsigned long long max(unsigned long long a, unsigned long long b) 
# 1074
{ 
# 1075
return ullmax(a, b); 
# 1076
} 
# 1078
static inline unsigned long long max(long long a, unsigned long long b) 
# 1079
{ 
# 1080
return ullmax((unsigned long long)a, b); 
# 1081
} 
# 1083
static inline unsigned long long max(unsigned long long a, long long b) 
# 1084
{ 
# 1085
return ullmax(a, (unsigned long long)b); 
# 1086
} 
# 1088
static inline float max(float a, float b) 
# 1089
{ 
# 1090
return fmaxf(a, b); 
# 1091
} 
# 1093
static inline double max(double a, double b) 
# 1094
{ 
# 1095
return fmax(a, b); 
# 1096
} 
# 1098
static inline double max(float a, double b) 
# 1099
{ 
# 1100
return fmax((double)a, b); 
# 1101
} 
# 1103
static inline double max(double a, float b) 
# 1104
{ 
# 1105
return fmax(a, (double)b); 
# 1106
} 
# 1117 "/usr/local/cuda/bin/..//include/crt/math_functions.hpp"
inline int min(int a, int b) 
# 1118
{ 
# 1119
return (a < b) ? a : b; 
# 1120
} 
# 1122
inline unsigned umin(unsigned a, unsigned b) 
# 1123
{ 
# 1124
return (a < b) ? a : b; 
# 1125
} 
# 1127
inline long long llmin(long long a, long long b) 
# 1128
{ 
# 1129
return (a < b) ? a : b; 
# 1130
} 
# 1132
inline unsigned long long ullmin(unsigned long long a, unsigned long long 
# 1133
b) 
# 1134
{ 
# 1135
return (a < b) ? a : b; 
# 1136
} 
# 1138
inline int max(int a, int b) 
# 1139
{ 
# 1140
return (a > b) ? a : b; 
# 1141
} 
# 1143
inline unsigned umax(unsigned a, unsigned b) 
# 1144
{ 
# 1145
return (a > b) ? a : b; 
# 1146
} 
# 1148
inline long long llmax(long long a, long long b) 
# 1149
{ 
# 1150
return (a > b) ? a : b; 
# 1151
} 
# 1153
inline unsigned long long ullmax(unsigned long long a, unsigned long long 
# 1154
b) 
# 1155
{ 
# 1156
return (a > b) ? a : b; 
# 1157
} 
# 74 "/usr/local/cuda/bin/..//include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 75
struct surface : public surfaceReference { 
# 78
surface() 
# 79
{ 
# 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 81
} 
# 83
surface(cudaChannelFormatDesc desc) 
# 84
{ 
# 85
(channelDesc) = desc; 
# 86
} 
# 88
}; 
# 90
template< int dim> 
# 91
struct surface< void, dim>  : public surfaceReference { 
# 94
surface() 
# 95
{ 
# 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 97
} 
# 99
}; 
# 74 "/usr/local/cuda/bin/..//include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 75
struct texture : public textureReference { 
# 78
texture(int norm = 0, cudaTextureFilterMode 
# 79
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 80
aMode = cudaAddressModeClamp) 
# 81
{ 
# 82
(normalized) = norm; 
# 83
(filterMode) = fMode; 
# 84
((addressMode)[0]) = aMode; 
# 85
((addressMode)[1]) = aMode; 
# 86
((addressMode)[2]) = aMode; 
# 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 88
(sRGB) = 0; 
# 89
} 
# 91
texture(int norm, cudaTextureFilterMode 
# 92
fMode, cudaTextureAddressMode 
# 93
aMode, cudaChannelFormatDesc 
# 94
desc) 
# 95
{ 
# 96
(normalized) = norm; 
# 97
(filterMode) = fMode; 
# 98
((addressMode)[0]) = aMode; 
# 99
((addressMode)[1]) = aMode; 
# 100
((addressMode)[2]) = aMode; 
# 101
(channelDesc) = desc; 
# 102
(sRGB) = 0; 
# 103
} 
# 105
}; 
# 89 "/usr/local/cuda/bin/..//include/crt/device_functions.h"
extern "C" {
# 3211 "/usr/local/cuda/bin/..//include/crt/device_functions.h"
}
# 3219
__attribute__((unused)) static inline int mulhi(int a, int b); 
# 3221
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b); 
# 3223
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b); 
# 3225
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b); 
# 3227
__attribute__((unused)) static inline long long mul64hi(long long a, long long b); 
# 3229
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b); 
# 3231
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b); 
# 3233
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b); 
# 3235
__attribute__((unused)) static inline int float_as_int(float a); 
# 3237
__attribute__((unused)) static inline float int_as_float(int a); 
# 3239
__attribute__((unused)) static inline unsigned float_as_uint(float a); 
# 3241
__attribute__((unused)) static inline float uint_as_float(unsigned a); 
# 3243
__attribute__((unused)) static inline float saturate(float a); 
# 3245
__attribute__((unused)) static inline int mul24(int a, int b); 
# 3247
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b); 
# 3249
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
# 3251
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
# 3253
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
# 3255
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 90 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(int a, int b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __mulhi(a, b); 
# 93
} 
#endif
# 95 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, b); 
# 98
} 
#endif
# 100 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __umulhi((unsigned)a, b); 
# 103
} 
#endif
# 105 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umulhi(a, (unsigned)b); 
# 108
} 
#endif
# 110 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __mul64hi(a, b); 
# 113
} 
#endif
# 115 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, b); 
# 118
} 
#endif
# 120 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b) 
# 121
{int volatile ___ = 1;(void)a;(void)b;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __umul64hi((unsigned long long)a, b); 
# 123
} 
#endif
# 125 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b) 
# 126
{int volatile ___ = 1;(void)a;(void)b;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __umul64hi(a, (unsigned long long)b); 
# 128
} 
#endif
# 130 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_int(a); 
# 133
} 
#endif
# 135 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(int a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __int_as_float(a); 
# 138
} 
#endif
# 140 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(float a) 
# 141
{int volatile ___ = 1;(void)a;
# 143
::exit(___);}
#if 0
# 141
{ 
# 142
return __float_as_uint(a); 
# 143
} 
#endif
# 145 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(unsigned a) 
# 146
{int volatile ___ = 1;(void)a;
# 148
::exit(___);}
#if 0
# 146
{ 
# 147
return __uint_as_float(a); 
# 148
} 
#endif
# 149 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline float saturate(float a) 
# 150
{int volatile ___ = 1;(void)a;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __saturatef(a); 
# 152
} 
#endif
# 154 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mul24(int a, int b) 
# 155
{int volatile ___ = 1;(void)a;(void)b;
# 157
::exit(___);}
#if 0
# 155
{ 
# 156
return __mul24(a, b); 
# 157
} 
#endif
# 159 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b) 
# 160
{int volatile ___ = 1;(void)a;(void)b;
# 162
::exit(___);}
#if 0
# 160
{ 
# 161
return __umul24(a, b); 
# 162
} 
#endif
# 164 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode) 
# 165
{int volatile ___ = 1;(void)a;(void)mode;
# 170
::exit(___);}
#if 0
# 165
{ 
# 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 170
} 
#endif
# 172 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode) 
# 173
{int volatile ___ = 1;(void)a;(void)mode;
# 178
::exit(___);}
#if 0
# 173
{ 
# 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 178
} 
#endif
# 180 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode) 
# 181
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 181
{ 
# 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 186
} 
#endif
# 188 "/usr/local/cuda/bin/..//include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode) 
# 189
{int volatile ___ = 1;(void)a;(void)mode;
# 194
::exit(___);}
#if 0
# 189
{ 
# 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 194
} 
#endif
# 106 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 120 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 120
{ } 
#endif
# 122 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 132
{ } 
#endif
# 134 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 138
{ } 
#endif
# 140 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 144
{ } 
#endif
# 146 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 146
{ } 
#endif
# 171 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
extern "C" {
# 180
}
# 189 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 193
{ } 
#endif
# 195 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/usr/local/cuda/bin/..//include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 197
{ } 
#endif
# 87 "/usr/local/cuda/bin/..//include/crt/device_double_functions.h"
extern "C" {
# 1139 "/usr/local/cuda/bin/..//include/crt/device_double_functions.h"
}
# 1147
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1149
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1151
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1153
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1157
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1159
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1161
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1163
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1165
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1167
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1169
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1171
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 93 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 94
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 99
::exit(___);}
#if 0
# 94
{ 
# 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 99
} 
#endif
# 101 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 102
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 107
::exit(___);}
#if 0
# 102
{ 
# 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 107
} 
#endif
# 109 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 110
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 115
::exit(___);}
#if 0
# 110
{ 
# 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 115
} 
#endif
# 117 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 118
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 123
::exit(___);}
#if 0
# 118
{ 
# 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 123
} 
#endif
# 125 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 126
{int volatile ___ = 1;(void)a;(void)mode;
# 131
::exit(___);}
#if 0
# 126
{ 
# 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 131
} 
#endif
# 133 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 134
{int volatile ___ = 1;(void)a;(void)mode;
# 139
::exit(___);}
#if 0
# 134
{ 
# 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 139
} 
#endif
# 141 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 142
{int volatile ___ = 1;(void)a;(void)mode;
# 147
::exit(___);}
#if 0
# 142
{ 
# 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 147
} 
#endif
# 149 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 150
{int volatile ___ = 1;(void)a;(void)mode;
# 155
::exit(___);}
#if 0
# 150
{ 
# 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 155
} 
#endif
# 157 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 158
{int volatile ___ = 1;(void)a;(void)mode;
# 163
::exit(___);}
#if 0
# 158
{ 
# 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 163
} 
#endif
# 165 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 166
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 166
{ 
# 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 171
} 
#endif
# 173 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 178 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 181
::exit(___);}
#if 0
# 179
{ 
# 180
return (double)a; 
# 181
} 
#endif
# 183 "/usr/local/cuda/bin/..//include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 184
{int volatile ___ = 1;(void)a;(void)mode;
# 186
::exit(___);}
#if 0
# 184
{ 
# 185
return (double)a; 
# 186
} 
#endif
# 89 "/usr/local/cuda/bin/..//include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 89
{ } 
#endif
# 100 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 102
{ } 
#endif
# 104 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 104
{ } 
#endif
# 106 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 106
{ } 
#endif
# 108 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 108
{ } 
#endif
# 110 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 110
{ } 
#endif
# 112 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 112
{ } 
#endif
# 114 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 114
{ } 
#endif
# 116 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 116
{ } 
#endif
# 118 "/usr/local/cuda/bin/..//include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 118
{ } 
#endif
# 303 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 303
{ } 
#endif
# 306 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 306
{ } 
#endif
# 309 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 309
{ } 
#endif
# 312 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 312
{ } 
#endif
# 315 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 315
{ } 
#endif
# 318 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 318
{ } 
#endif
# 321 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 321
{ } 
#endif
# 324 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 324
{ } 
#endif
# 327 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 327
{ } 
#endif
# 330 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 330
{ } 
#endif
# 333 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 333
{ } 
#endif
# 336 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 336
{ } 
#endif
# 339 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 339
{ } 
#endif
# 342 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 342
{ } 
#endif
# 345 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 345
{ } 
#endif
# 348 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 348
{ } 
#endif
# 351 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 351
{ } 
#endif
# 354 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 354
{ } 
#endif
# 357 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 357
{ } 
#endif
# 360 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 360
{ } 
#endif
# 363 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 363
{ } 
#endif
# 366 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 366
{ } 
#endif
# 369 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 369
{ } 
#endif
# 372 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 372
{ } 
#endif
# 375 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 375
{ } 
#endif
# 378 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 378
{ } 
#endif
# 381 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 381
{ } 
#endif
# 384 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 384
{ } 
#endif
# 387 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 387
{ } 
#endif
# 390 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 390
{ } 
#endif
# 393 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 393
{ } 
#endif
# 396 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 396
{ } 
#endif
# 399 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 399
{ } 
#endif
# 402 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 402
{ } 
#endif
# 405 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 405
{ } 
#endif
# 408 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 408
{ } 
#endif
# 411 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 411
{ } 
#endif
# 414 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 414
{ } 
#endif
# 417 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 417
{ } 
#endif
# 420 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 420
{ } 
#endif
# 423 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 423
{ } 
#endif
# 426 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 426
{ } 
#endif
# 429 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 429
{ } 
#endif
# 432 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 432
{ } 
#endif
# 435 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 435
{ } 
#endif
# 438 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 439
{ } 
#endif
# 442 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 443
{ } 
#endif
# 446 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 447
compare, unsigned long long 
# 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 448
{ } 
#endif
# 451 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 452
compare, unsigned long long 
# 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 453
{ } 
#endif
# 456 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 456
{ } 
#endif
# 459 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 459
{ } 
#endif
# 462 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 462
{ } 
#endif
# 465 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 465
{ } 
#endif
# 468 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 468
{ } 
#endif
# 471 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 471
{ } 
#endif
# 474 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 474
{ } 
#endif
# 477 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 477
{ } 
#endif
# 480 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 480
{ } 
#endif
# 483 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 483
{ } 
#endif
# 486 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 486
{ } 
#endif
# 489 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 489
{ } 
#endif
# 492 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 492
{ } 
#endif
# 495 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 495
{ } 
#endif
# 498 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 498
{ } 
#endif
# 501 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 501
{ } 
#endif
# 504 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 504
{ } 
#endif
# 507 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 507
{ } 
#endif
# 510 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 510
{ } 
#endif
# 513 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 513
{ } 
#endif
# 516 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 516
{ } 
#endif
# 519 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 519
{ } 
#endif
# 522 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 522
{ } 
#endif
# 525 "/usr/local/cuda/bin/..//include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 525
{ } 
#endif
# 90 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
extern "C" {
# 1475 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
}
# 1482
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1482
{ } 
#endif
# 1484 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1484
{ } 
#endif
# 1486 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1486
{ } 
#endif
# 1488 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1488
{ } 
#endif
# 1495 "/usr/local/cuda/bin/..//include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1495
{ } 
#endif
# 105 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 113
{ } 
#endif
# 121 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 121
{ } 
#endif
# 122 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 122
{ } 
#endif
# 124 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 125 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 125
{ } 
#endif
# 127 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 128
{ } 
#endif
# 130 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 131
{ } 
#endif
# 133 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 134
{ } 
#endif
# 136 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 140
{ } 
#endif
# 142 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 143
{ } 
#endif
# 145 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 146
{ } 
#endif
# 148 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 152
{ } 
#endif
# 154 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 155
{ } 
#endif
# 158 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 158
{ } 
#endif
# 159 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 159
{ } 
#endif
# 161 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 162
{ } 
#endif
# 164 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 165
{ } 
#endif
# 167 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 168
{ } 
#endif
# 170 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 171
{ } 
#endif
# 173 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 174
{ } 
#endif
# 176 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 177
{ } 
#endif
# 179 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 180
{ } 
#endif
# 182 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 183
{ } 
#endif
# 185 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 186
{ } 
#endif
# 188 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 189
{ } 
#endif
# 191 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 192
{ } 
#endif
# 196 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 196
{ } 
#endif
# 197 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 197
{ } 
#endif
# 199 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 200
{ } 
#endif
# 202 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 203
{ } 
#endif
# 205 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 206
{ } 
#endif
# 208 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 212
{ } 
#endif
# 214 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 215
{ } 
#endif
# 217 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/usr/local/cuda/bin/..//include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 218
{ } 
#endif
# 87 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 88
{ } 
#endif
# 90 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 91 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 116 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 116
{ } 
#endif
# 117 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 123 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 127
{ } 
#endif
# 128 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 140 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 140
{ } 
#endif
# 141 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 151 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 152 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 152
{ } 
#endif
# 153 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 159 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 159
{ } 
#endif
# 160 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 160
{ } 
#endif
# 162 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 163 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 176
{ } 
#endif
# 177 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 188
{ } 
#endif
# 189 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 195 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 195
{ } 
#endif
# 196 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 196
{ } 
#endif
# 198 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 199
{ } 
#endif
# 200 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 211 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 212 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 212
{ } 
#endif
# 213 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 223 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 224 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 224
{ } 
#endif
# 225 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 244 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 244
{ } 
#endif
# 256 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 256
{ } 
#endif
# 269 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 269
{ } 
#endif
# 281 "/usr/local/cuda/bin/..//include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 281
{ } 
#endif
# 89 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 93
{ } 
#endif
# 95 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 96
{ } 
#endif
# 98 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 99
{ } 
#endif
# 106 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda/bin/..//include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 110
{ } 
#endif
# 93 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 100
{ } 
#endif
# 102 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 103
{ } 
#endif
# 104 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 104
{ } 
#endif
# 105 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda/bin/..//include/crt/sm_70_rt.h"
__attribute__((unused)) static inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
# 111
{ } 
#endif
# 114 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 115
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 116
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 120
::exit(___);}
#if 0
# 116
{ 
# 120
} 
#endif
# 122 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 123
__attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 124
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 130
::exit(___);}
#if 0
# 124
{ 
# 130
} 
#endif
# 132 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 133
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 134
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 138
::exit(___);}
#if 0
# 134
{ 
# 138
} 
#endif
# 141 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 142
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 143
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 147
::exit(___);}
#if 0
# 143
{ 
# 147
} 
#endif
# 149 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 150
__attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 151
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 157
::exit(___);}
#if 0
# 151
{ 
# 157
} 
#endif
# 159 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 160
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 161
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 165
::exit(___);}
#if 0
# 161
{ 
# 165
} 
#endif
# 168 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 169
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 170
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 174
::exit(___);}
#if 0
# 170
{ 
# 174
} 
#endif
# 176 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 177
__attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 178
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 184
::exit(___);}
#if 0
# 178
{ 
# 184
} 
#endif
# 186 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 187
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 188
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 192
::exit(___);}
#if 0
# 188
{ 
# 192
} 
#endif
# 196 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 197
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 198
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 202
::exit(___);}
#if 0
# 198
{ 
# 202
} 
#endif
# 204 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 205
__attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 206
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 212
::exit(___);}
#if 0
# 206
{ 
# 212
} 
#endif
# 215 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 216
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 217
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 221
::exit(___);}
#if 0
# 217
{ 
# 221
} 
#endif
# 224 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 225
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 226
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 230
::exit(___);}
#if 0
# 226
{ 
# 230
} 
#endif
# 232 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 233
__attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 234
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 240
::exit(___);}
#if 0
# 234
{ 
# 240
} 
#endif
# 243 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 244
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 245
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 249
::exit(___);}
#if 0
# 245
{ 
# 249
} 
#endif
# 252 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 253
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 254
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 258
::exit(___);}
#if 0
# 254
{ 
# 258
} 
#endif
# 260 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 262
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 269
::exit(___);}
#if 0
# 262
{ 
# 269
} 
#endif
# 271 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 272
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 273
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 277
::exit(___);}
#if 0
# 273
{ 
# 277
} 
#endif
# 280 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 281
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 282
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 286
::exit(___);}
#if 0
# 282
{ 
# 286
} 
#endif
# 288 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 289
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 290
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 296
::exit(___);}
#if 0
# 290
{ 
# 296
} 
#endif
# 298 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 299
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 300
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 304
::exit(___);}
#if 0
# 300
{ 
# 304
} 
#endif
# 307 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 308
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 309
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 313
::exit(___);}
#if 0
# 309
{ 
# 313
} 
#endif
# 315 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 316
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 317
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 321
::exit(___);}
#if 0
# 317
{ 
# 321
} 
#endif
# 325 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 326
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 327
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 331
::exit(___);}
#if 0
# 327
{ 
# 331
} 
#endif
# 333 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 334
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 335
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 339
::exit(___);}
#if 0
# 335
{ 
# 339
} 
#endif
# 342 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 343
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 344
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 348
::exit(___);}
#if 0
# 344
{ 
# 348
} 
#endif
# 350 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 351
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 352
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 356
::exit(___);}
#if 0
# 352
{ 
# 356
} 
#endif
# 359 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 360
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 361
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 365
::exit(___);}
#if 0
# 361
{ 
# 365
} 
#endif
# 367 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 368
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 369
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 376 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 377
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 378
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 382
::exit(___);}
#if 0
# 378
{ 
# 382
} 
#endif
# 384 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 385
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 386
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 390
::exit(___);}
#if 0
# 386
{ 
# 390
} 
#endif
# 393 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 394
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 395
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 399
::exit(___);}
#if 0
# 395
{ 
# 399
} 
#endif
# 401 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 402
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 403
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 407
::exit(___);}
#if 0
# 403
{ 
# 407
} 
#endif
# 411 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 412
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 413
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 417
::exit(___);}
#if 0
# 413
{ 
# 417
} 
#endif
# 419 "/usr/local/cuda/bin/..//include/surface_functions.h"
template< class T> 
# 420
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 421
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 425
::exit(___);}
#if 0
# 421
{ 
# 425
} 
#endif
# 66 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 67
struct __nv_tex_rmet_ret { }; 
# 69
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 70
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 71
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 72
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 73
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 74
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 75
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 76
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 77
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 80
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 81
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 82
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 83
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 84
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 86
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 89
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 90
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 91
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 92
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 93
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 95
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 107 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 108
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 109
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 110
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 113
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 125 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 126
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 127
{int volatile ___ = 1;(void)t;(void)x;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 135 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 136
struct __nv_tex_rmnf_ret { }; 
# 138
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 139
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 140
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 141
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 142
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 143
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 144
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 156
template< class T> 
# 157
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 158
{int volatile ___ = 1;(void)t;(void)x;
# 165
::exit(___);}
#if 0
# 158
{ 
# 165
} 
#endif
# 168 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 169
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 170
{int volatile ___ = 1;(void)t;(void)x;
# 176
::exit(___);}
#if 0
# 170
{ 
# 176
} 
#endif
# 178 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 179
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 180
{int volatile ___ = 1;(void)t;(void)x;
# 187
::exit(___);}
#if 0
# 180
{ 
# 187
} 
#endif
# 191 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 192
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 193
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 200
::exit(___);}
#if 0
# 193
{ 
# 200
} 
#endif
# 202 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 203
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 204
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 211
::exit(___);}
#if 0
# 204
{ 
# 211
} 
#endif
# 215 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 216
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 217
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 223
::exit(___);}
#if 0
# 217
{ 
# 223
} 
#endif
# 225 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 226
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 227
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 234
::exit(___);}
#if 0
# 227
{ 
# 234
} 
#endif
# 238 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 239
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 240
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 246
::exit(___);}
#if 0
# 240
{ 
# 246
} 
#endif
# 248 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 249
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 250
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 257
::exit(___);}
#if 0
# 250
{ 
# 257
} 
#endif
# 260 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 261
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 262
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 268
::exit(___);}
#if 0
# 262
{ 
# 268
} 
#endif
# 270 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 271
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 272
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 279
::exit(___);}
#if 0
# 272
{ 
# 279
} 
#endif
# 282 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 283
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 284
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 290
::exit(___);}
#if 0
# 284
{ 
# 290
} 
#endif
# 292 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 293
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 294
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 301
::exit(___);}
#if 0
# 294
{ 
# 301
} 
#endif
# 304 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 305
struct __nv_tex2dgather_ret { }; 
# 306
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 307
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 308
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 309
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 310
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 311
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 312
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 319
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 323
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 330
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 334
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 341
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 346
template< class T> 
# 347
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 348
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 355
::exit(___);}
#if 0
# 348
{ 
# 355
} 
#endif
# 358 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 359
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 360
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 361
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 362
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 363
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 364
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 381
template< class T> 
# 382
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 383
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 390
::exit(___);}
#if 0
# 383
{ 
# 390
} 
#endif
# 394 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 395
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 396
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 402
::exit(___);}
#if 0
# 396
{ 
# 402
} 
#endif
# 404 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 405
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 406
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 413
::exit(___);}
#if 0
# 406
{ 
# 413
} 
#endif
# 416 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 417
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 418
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 424
::exit(___);}
#if 0
# 418
{ 
# 424
} 
#endif
# 426 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 427
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 428
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 435
::exit(___);}
#if 0
# 428
{ 
# 435
} 
#endif
# 438 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 439
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 440
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 446
::exit(___);}
#if 0
# 440
{ 
# 446
} 
#endif
# 448 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 449
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 450
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 457
::exit(___);}
#if 0
# 450
{ 
# 457
} 
#endif
# 460 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 461
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 462
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 468
::exit(___);}
#if 0
# 462
{ 
# 468
} 
#endif
# 470 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 471
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 472
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 479
::exit(___);}
#if 0
# 472
{ 
# 479
} 
#endif
# 482 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 483
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 484
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 490
::exit(___);}
#if 0
# 484
{ 
# 490
} 
#endif
# 492 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 493
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 494
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 501
::exit(___);}
#if 0
# 494
{ 
# 501
} 
#endif
# 504 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 505
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 506
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 512
::exit(___);}
#if 0
# 506
{ 
# 512
} 
#endif
# 514 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 515
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 516
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 523
::exit(___);}
#if 0
# 516
{ 
# 523
} 
#endif
# 527 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 528
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 529
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 535
::exit(___);}
#if 0
# 529
{ 
# 535
} 
#endif
# 537 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 538
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 539
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 546
::exit(___);}
#if 0
# 539
{ 
# 546
} 
#endif
# 550 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 551
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 552
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 558
::exit(___);}
#if 0
# 552
{ 
# 558
} 
#endif
# 560 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 561
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 562
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 569
::exit(___);}
#if 0
# 562
{ 
# 569
} 
#endif
# 573 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 574
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 575
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 581
::exit(___);}
#if 0
# 575
{ 
# 581
} 
#endif
# 583 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 584
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 585
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 592
::exit(___);}
#if 0
# 585
{ 
# 592
} 
#endif
# 596 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 597
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 598
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 604
::exit(___);}
#if 0
# 598
{ 
# 604
} 
#endif
# 606 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 607
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 608
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 615
::exit(___);}
#if 0
# 608
{ 
# 615
} 
#endif
# 619 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 620
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 621
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 627
::exit(___);}
#if 0
# 621
{ 
# 627
} 
#endif
# 629 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 630
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 631
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 638
::exit(___);}
#if 0
# 631
{ 
# 638
} 
#endif
# 642 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 643
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 644
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 650
::exit(___);}
#if 0
# 644
{ 
# 650
} 
#endif
# 652 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 653
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 654
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 661
::exit(___);}
#if 0
# 654
{ 
# 661
} 
#endif
# 664 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 665
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 666
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 672
::exit(___);}
#if 0
# 666
{ 
# 672
} 
#endif
# 674 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 675
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 676
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 683
::exit(___);}
#if 0
# 676
{ 
# 683
} 
#endif
# 686 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 687
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 688
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 694
::exit(___);}
#if 0
# 688
{ 
# 694
} 
#endif
# 696 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 697
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 698
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 705
::exit(___);}
#if 0
# 698
{ 
# 705
} 
#endif
# 708 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 709
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 710
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 716
::exit(___);}
#if 0
# 710
{ 
# 716
} 
#endif
# 718 "/usr/local/cuda/bin/..//include/texture_fetch_functions.h"
template< class T> 
# 719
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 720
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 727
::exit(___);}
#if 0
# 720
{ 
# 727
} 
#endif
# 60 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 61
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 62
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 63
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 64
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 65
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 96 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 97
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 98
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 99
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 103
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 104
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 105
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 109
::exit(___);}
#if 0
# 105
{ 
# 109
} 
#endif
# 111 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 112
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 113
{int volatile ___ = 1;(void)texObject;(void)x;
# 119
::exit(___);}
#if 0
# 113
{ 
# 119
} 
#endif
# 121 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 122
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 123
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 127
::exit(___);}
#if 0
# 123
{ 
# 127
} 
#endif
# 130 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 131
tex1D(cudaTextureObject_t texObject, float x) 
# 132
{int volatile ___ = 1;(void)texObject;(void)x;
# 138
::exit(___);}
#if 0
# 132
{ 
# 138
} 
#endif
# 141 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 142
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 143
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 147
::exit(___);}
#if 0
# 143
{ 
# 147
} 
#endif
# 149 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 150
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 151
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 157
::exit(___);}
#if 0
# 151
{ 
# 157
} 
#endif
# 159 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 160
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 161
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 165
::exit(___);}
#if 0
# 161
{ 
# 165
} 
#endif
# 167 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 168
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 169
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 175
::exit(___);}
#if 0
# 169
{ 
# 175
} 
#endif
# 177 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 178
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 179
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 183
::exit(___);}
#if 0
# 179
{ 
# 183
} 
#endif
# 185 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 186
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 187
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 193
::exit(___);}
#if 0
# 187
{ 
# 193
} 
#endif
# 195 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 196
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 197
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 201
::exit(___);}
#if 0
# 197
{ 
# 201
} 
#endif
# 203 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 204
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 205
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 211
::exit(___);}
#if 0
# 205
{ 
# 211
} 
#endif
# 214 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 215
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 216
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 220
::exit(___);}
#if 0
# 216
{ 
# 220
} 
#endif
# 223 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 224
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 225
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 231
::exit(___);}
#if 0
# 225
{ 
# 231
} 
#endif
# 234 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 235
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 236
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 243
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 244
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 250
::exit(___);}
#if 0
# 244
{ 
# 250
} 
#endif
# 252 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 253
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 254
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 258
::exit(___);}
#if 0
# 254
{ 
# 258
} 
#endif
# 260 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 261
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 262
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 268
::exit(___);}
#if 0
# 262
{ 
# 268
} 
#endif
# 272 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 273
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 274
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 278
::exit(___);}
#if 0
# 274
{ 
# 278
} 
#endif
# 280 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 281
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 282
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 288
::exit(___);}
#if 0
# 282
{ 
# 288
} 
#endif
# 291 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 292
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 293
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 297
::exit(___);}
#if 0
# 293
{ 
# 297
} 
#endif
# 299 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 300
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 301
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 307
::exit(___);}
#if 0
# 301
{ 
# 307
} 
#endif
# 310 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 311
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 312
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 316
::exit(___);}
#if 0
# 312
{ 
# 316
} 
#endif
# 318 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 319
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 320
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 326
::exit(___);}
#if 0
# 320
{ 
# 326
} 
#endif
# 329 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 330
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 331
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 335
::exit(___);}
#if 0
# 331
{ 
# 335
} 
#endif
# 337 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 338
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 339
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 345
::exit(___);}
#if 0
# 339
{ 
# 345
} 
#endif
# 348 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 349
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 350
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 354
::exit(___);}
#if 0
# 350
{ 
# 354
} 
#endif
# 356 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 357
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 358
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 364
::exit(___);}
#if 0
# 358
{ 
# 364
} 
#endif
# 367 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 368
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 369
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 373
::exit(___);}
#if 0
# 369
{ 
# 373
} 
#endif
# 375 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 376
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 377
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 383
::exit(___);}
#if 0
# 377
{ 
# 383
} 
#endif
# 386 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 387
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 388
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 392
::exit(___);}
#if 0
# 388
{ 
# 392
} 
#endif
# 394 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 395
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 396
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 402
::exit(___);}
#if 0
# 396
{ 
# 402
} 
#endif
# 404 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 405
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 406
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 410
::exit(___);}
#if 0
# 406
{ 
# 410
} 
#endif
# 412 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 413
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 414
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 420
::exit(___);}
#if 0
# 414
{ 
# 420
} 
#endif
# 422 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 423
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 424
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 428
::exit(___);}
#if 0
# 424
{ 
# 428
} 
#endif
# 430 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 431
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 432
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 438
::exit(___);}
#if 0
# 432
{ 
# 438
} 
#endif
# 441 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 442
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 443
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 448
::exit(___);}
#if 0
# 443
{ 
# 448
} 
#endif
# 450 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 451
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 452
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 458
::exit(___);}
#if 0
# 452
{ 
# 458
} 
#endif
# 461 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 462
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 463
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 467
::exit(___);}
#if 0
# 463
{ 
# 467
} 
#endif
# 469 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 470
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 471
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 477
::exit(___);}
#if 0
# 471
{ 
# 477
} 
#endif
# 480 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 481
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 482
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 486
::exit(___);}
#if 0
# 482
{ 
# 486
} 
#endif
# 488 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 489
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 490
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 496
::exit(___);}
#if 0
# 490
{ 
# 496
} 
#endif
# 499 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 500
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 501
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 505
::exit(___);}
#if 0
# 501
{ 
# 505
} 
#endif
# 507 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 508
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 509
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 515
::exit(___);}
#if 0
# 509
{ 
# 515
} 
#endif
# 518 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 519
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 520
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 524
::exit(___);}
#if 0
# 520
{ 
# 524
} 
#endif
# 526 "/usr/local/cuda/bin/..//include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 527
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 528
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 534
::exit(___);}
#if 0
# 528
{ 
# 534
} 
#endif
# 59 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 99
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 100
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 101
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 105
::exit(___);}
#if 0
# 101
{ 
# 105
} 
#endif
# 107 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 108
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 109
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 115
::exit(___);}
#if 0
# 109
{ 
# 115
} 
#endif
# 117 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 118
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 119
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 123
::exit(___);}
#if 0
# 119
{ 
# 123
} 
#endif
# 125 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 126
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 127
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 133
::exit(___);}
#if 0
# 127
{ 
# 133
} 
#endif
# 136 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 137
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 138
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 142
::exit(___);}
#if 0
# 138
{ 
# 142
} 
#endif
# 144 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 145
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 146
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 152
::exit(___);}
#if 0
# 146
{ 
# 152
} 
#endif
# 154 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 155
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 156
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 160
::exit(___);}
#if 0
# 156
{ 
# 160
} 
#endif
# 162 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 163
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 164
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 170
::exit(___);}
#if 0
# 164
{ 
# 170
} 
#endif
# 172 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 173
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 174
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 178
::exit(___);}
#if 0
# 174
{ 
# 178
} 
#endif
# 180 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 181
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 182
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 188
::exit(___);}
#if 0
# 182
{ 
# 188
} 
#endif
# 190 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 191
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 192
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 196
::exit(___);}
#if 0
# 192
{ 
# 196
} 
#endif
# 198 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 199
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 206
::exit(___);}
#if 0
# 200
{ 
# 206
} 
#endif
# 208 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 209
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 210
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 214
::exit(___);}
#if 0
# 210
{ 
# 214
} 
#endif
# 216 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 217
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 218
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 224
::exit(___);}
#if 0
# 218
{ 
# 224
} 
#endif
# 226 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 227
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 235
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 240
::exit(___);}
#if 0
# 236
{ 
# 240
} 
#endif
# 242 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 243
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 244
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 248
::exit(___);}
#if 0
# 244
{ 
# 248
} 
#endif
# 250 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 251
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 252
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 256
::exit(___);}
#if 0
# 252
{ 
# 256
} 
#endif
# 258 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 259
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 260
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 264
::exit(___);}
#if 0
# 260
{ 
# 264
} 
#endif
# 266 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 267
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 268
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 272
::exit(___);}
#if 0
# 268
{ 
# 272
} 
#endif
# 274 "/usr/local/cuda/bin/..//include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 275
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 276
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 280
::exit(___);}
#if 0
# 276
{ 
# 280
} 
#endif
# 3290 "/usr/local/cuda/bin/..//include/crt/device_functions.h"
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void * stream = 0); 
# 68 "/usr/local/cuda/bin/..//include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 192 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 193
cudaLaunchKernel(const T *
# 194
func, dim3 
# 195
gridDim, dim3 
# 196
blockDim, void **
# 197
args, size_t 
# 198
sharedMem = 0, cudaStream_t 
# 199
stream = 0) 
# 201
{ 
# 202
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 203
} 
# 254 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 255
cudaLaunchCooperativeKernel(const T *
# 256
func, dim3 
# 257
gridDim, dim3 
# 258
blockDim, void **
# 259
args, size_t 
# 260
sharedMem = 0, cudaStream_t 
# 261
stream = 0) 
# 263
{ 
# 264
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 265
} 
# 294 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 295
cudaSetupArgument(T 
# 296
arg, size_t 
# 297
offset) 
# 299
{ 
# 300
return ::cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
# 301
} 
# 334 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 335
event, unsigned 
# 336
flags) 
# 338
{ 
# 339
return ::cudaEventCreateWithFlags(event, flags); 
# 340
} 
# 399 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 400
ptr, size_t 
# 401
size, unsigned 
# 402
flags) 
# 404
{ 
# 405
return ::cudaHostAlloc(ptr, size, flags); 
# 406
} 
# 408
template< class T> static inline cudaError_t 
# 409
cudaHostAlloc(T **
# 410
ptr, size_t 
# 411
size, unsigned 
# 412
flags) 
# 414
{ 
# 415
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 416
} 
# 418
template< class T> static inline cudaError_t 
# 419
cudaHostGetDevicePointer(T **
# 420
pDevice, void *
# 421
pHost, unsigned 
# 422
flags) 
# 424
{ 
# 425
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 426
} 
# 528 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 529
cudaMallocManaged(T **
# 530
devPtr, size_t 
# 531
size, unsigned 
# 532
flags = 1) 
# 534
{ 
# 535
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 536
} 
# 618 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 619
cudaStreamAttachMemAsync(cudaStream_t 
# 620
stream, T *
# 621
devPtr, size_t 
# 622
length = 0, unsigned 
# 623
flags = 4) 
# 625
{ 
# 626
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 627
} 
# 629
template< class T> inline cudaError_t 
# 630
cudaMalloc(T **
# 631
devPtr, size_t 
# 632
size) 
# 634
{ 
# 635
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 636
} 
# 638
template< class T> static inline cudaError_t 
# 639
cudaMallocHost(T **
# 640
ptr, size_t 
# 641
size, unsigned 
# 642
flags = 0) 
# 644
{ 
# 645
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 646
} 
# 648
template< class T> static inline cudaError_t 
# 649
cudaMallocPitch(T **
# 650
devPtr, size_t *
# 651
pitch, size_t 
# 652
width, size_t 
# 653
height) 
# 655
{ 
# 656
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 657
} 
# 696 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 697
cudaMemcpyToSymbol(const T &
# 698
symbol, const void *
# 699
src, size_t 
# 700
count, size_t 
# 701
offset = 0, cudaMemcpyKind 
# 702
kind = cudaMemcpyHostToDevice) 
# 704
{ 
# 705
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 706
} 
# 750 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 751
cudaMemcpyToSymbolAsync(const T &
# 752
symbol, const void *
# 753
src, size_t 
# 754
count, size_t 
# 755
offset = 0, cudaMemcpyKind 
# 756
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 757
stream = 0) 
# 759
{ 
# 760
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 761
} 
# 798 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 799
cudaMemcpyFromSymbol(void *
# 800
dst, const T &
# 801
symbol, size_t 
# 802
count, size_t 
# 803
offset = 0, cudaMemcpyKind 
# 804
kind = cudaMemcpyDeviceToHost) 
# 806
{ 
# 807
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 808
} 
# 852 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 853
cudaMemcpyFromSymbolAsync(void *
# 854
dst, const T &
# 855
symbol, size_t 
# 856
count, size_t 
# 857
offset = 0, cudaMemcpyKind 
# 858
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 859
stream = 0) 
# 861
{ 
# 862
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 863
} 
# 888 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 889
cudaGetSymbolAddress(void **
# 890
devPtr, const T &
# 891
symbol) 
# 893
{ 
# 894
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 895
} 
# 920 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 921
cudaGetSymbolSize(size_t *
# 922
size, const T &
# 923
symbol) 
# 925
{ 
# 926
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 927
} 
# 964 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 965
cudaBindTexture(size_t *
# 966
offset, const texture< T, dim, readMode>  &
# 967
tex, const void *
# 968
devPtr, const cudaChannelFormatDesc &
# 969
desc, size_t 
# 970
size = ((2147483647) * 2U) + 1U) 
# 972 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
{ 
# 973
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 974
} 
# 1010 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1011
cudaBindTexture(size_t *
# 1012
offset, const texture< T, dim, readMode>  &
# 1013
tex, const void *
# 1014
devPtr, size_t 
# 1015
size = ((2147483647) * 2U) + 1U) 
# 1017 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
{ 
# 1018
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 1019
} 
# 1067 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1068
cudaBindTexture2D(size_t *
# 1069
offset, const texture< T, dim, readMode>  &
# 1070
tex, const void *
# 1071
devPtr, const cudaChannelFormatDesc &
# 1072
desc, size_t 
# 1073
width, size_t 
# 1074
height, size_t 
# 1075
pitch) 
# 1077
{ 
# 1078
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1079
} 
# 1126 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1127
cudaBindTexture2D(size_t *
# 1128
offset, const texture< T, dim, readMode>  &
# 1129
tex, const void *
# 1130
devPtr, size_t 
# 1131
width, size_t 
# 1132
height, size_t 
# 1133
pitch) 
# 1135
{ 
# 1136
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1137
} 
# 1169 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1170
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1171
tex, cudaArray_const_t 
# 1172
array, const cudaChannelFormatDesc &
# 1173
desc) 
# 1175
{ 
# 1176
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1177
} 
# 1208 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1209
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1210
tex, cudaArray_const_t 
# 1211
array) 
# 1213
{ 
# 1214
cudaChannelFormatDesc desc; 
# 1215
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1217
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1218
} 
# 1250 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1251
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1252
tex, cudaMipmappedArray_const_t 
# 1253
mipmappedArray, const cudaChannelFormatDesc &
# 1254
desc) 
# 1256
{ 
# 1257
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1258
} 
# 1289 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1290
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1291
tex, cudaMipmappedArray_const_t 
# 1292
mipmappedArray) 
# 1294
{ 
# 1295
cudaChannelFormatDesc desc; 
# 1296
cudaArray_t levelArray; 
# 1297
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1299
if (err != (cudaSuccess)) { 
# 1300
return err; 
# 1301
}  
# 1302
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1304
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1305
} 
# 1332 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1333
cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1334
tex) 
# 1336
{ 
# 1337
return ::cudaUnbindTexture(&tex); 
# 1338
} 
# 1368 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1369
cudaGetTextureAlignmentOffset(size_t *
# 1370
offset, const texture< T, dim, readMode>  &
# 1371
tex) 
# 1373
{ 
# 1374
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1375
} 
# 1421 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1422
cudaFuncSetCacheConfig(T *
# 1423
func, cudaFuncCache 
# 1424
cacheConfig) 
# 1426
{ 
# 1427
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1428
} 
# 1430
template< class T> static inline cudaError_t 
# 1431
cudaFuncSetSharedMemConfig(T *
# 1432
func, cudaSharedMemConfig 
# 1433
config) 
# 1435
{ 
# 1436
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1437
} 
# 1466 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1467
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1468
numBlocks, T 
# 1469
func, int 
# 1470
blockSize, size_t 
# 1471
dynamicSMemSize) 
# 1472
{ 
# 1473
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1474
} 
# 1517 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1518
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1519
numBlocks, T 
# 1520
func, int 
# 1521
blockSize, size_t 
# 1522
dynamicSMemSize, unsigned 
# 1523
flags) 
# 1524
{ 
# 1525
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1526
} 
# 1531
class __cudaOccupancyB2DHelper { 
# 1532
size_t n; 
# 1534
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1535
size_t operator()(int) 
# 1536
{ 
# 1537
return n; 
# 1538
} 
# 1539
}; 
# 1586 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1587
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1588
minGridSize, int *
# 1589
blockSize, T 
# 1590
func, UnaryFunction 
# 1591
blockSizeToDynamicSMemSize, int 
# 1592
blockSizeLimit = 0, unsigned 
# 1593
flags = 0) 
# 1594
{ 
# 1595
cudaError_t status; 
# 1598
int device; 
# 1599
cudaFuncAttributes attr; 
# 1602
int maxThreadsPerMultiProcessor; 
# 1603
int warpSize; 
# 1604
int devMaxThreadsPerBlock; 
# 1605
int multiProcessorCount; 
# 1606
int funcMaxThreadsPerBlock; 
# 1607
int occupancyLimit; 
# 1608
int granularity; 
# 1611
int maxBlockSize = 0; 
# 1612
int numBlocks = 0; 
# 1613
int maxOccupancy = 0; 
# 1616
int blockSizeToTryAligned; 
# 1617
int blockSizeToTry; 
# 1618
int blockSizeLimitAligned; 
# 1619
int occupancyInBlocks; 
# 1620
int occupancyInThreads; 
# 1621
size_t dynamicSMemSize; 
# 1627
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1628
return cudaErrorInvalidValue; 
# 1629
}  
# 1635
status = ::cudaGetDevice(&device); 
# 1636
if (status != (cudaSuccess)) { 
# 1637
return status; 
# 1638
}  
# 1640
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1644
if (status != (cudaSuccess)) { 
# 1645
return status; 
# 1646
}  
# 1648
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1652
if (status != (cudaSuccess)) { 
# 1653
return status; 
# 1654
}  
# 1656
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1660
if (status != (cudaSuccess)) { 
# 1661
return status; 
# 1662
}  
# 1664
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1668
if (status != (cudaSuccess)) { 
# 1669
return status; 
# 1670
}  
# 1672
status = cudaFuncGetAttributes(&attr, func); 
# 1673
if (status != (cudaSuccess)) { 
# 1674
return status; 
# 1675
}  
# 1677
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1683
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1684
granularity = warpSize; 
# 1686
if (blockSizeLimit == 0) { 
# 1687
blockSizeLimit = devMaxThreadsPerBlock; 
# 1688
}  
# 1690
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1691
blockSizeLimit = devMaxThreadsPerBlock; 
# 1692
}  
# 1694
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1695
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1696
}  
# 1698
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1700
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1704
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1705
blockSizeToTry = blockSizeLimit; 
# 1706
} else { 
# 1707
blockSizeToTry = blockSizeToTryAligned; 
# 1708
}  
# 1710
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1712
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1719
if (status != (cudaSuccess)) { 
# 1720
return status; 
# 1721
}  
# 1723
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1725
if (occupancyInThreads > maxOccupancy) { 
# 1726
maxBlockSize = blockSizeToTry; 
# 1727
numBlocks = occupancyInBlocks; 
# 1728
maxOccupancy = occupancyInThreads; 
# 1729
}  
# 1733
if (occupancyLimit == maxOccupancy) { 
# 1734
break; 
# 1735
}  
# 1736
}  
# 1744
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1745
(*blockSize) = maxBlockSize; 
# 1747
return status; 
# 1748
} 
# 1781 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1782
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1783
minGridSize, int *
# 1784
blockSize, T 
# 1785
func, UnaryFunction 
# 1786
blockSizeToDynamicSMemSize, int 
# 1787
blockSizeLimit = 0) 
# 1788
{ 
# 1789
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1790
} 
# 1826 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1827
cudaOccupancyMaxPotentialBlockSize(int *
# 1828
minGridSize, int *
# 1829
blockSize, T 
# 1830
func, size_t 
# 1831
dynamicSMemSize = 0, int 
# 1832
blockSizeLimit = 0) 
# 1833
{ 
# 1834
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 1835
} 
# 1885 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1886
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 1887
minGridSize, int *
# 1888
blockSize, T 
# 1889
func, size_t 
# 1890
dynamicSMemSize = 0, int 
# 1891
blockSizeLimit = 0, unsigned 
# 1892
flags = 0) 
# 1893
{ 
# 1894
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 1895
} 
# 1938 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1939
cudaLaunch(T *
# 1940
func) 
# 1942
{ 
# 1943
return ::cudaLaunch((const void *)func); 
# 1944
} 
# 1976 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1977
cudaFuncGetAttributes(cudaFuncAttributes *
# 1978
attr, T *
# 1979
entry) 
# 1981
{ 
# 1982
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 1983
} 
# 2019 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 2020
cudaFuncSetAttribute(T *
# 2021
entry, cudaFuncAttribute 
# 2022
attr, int 
# 2023
value) 
# 2025
{ 
# 2026
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 2027
} 
# 2051 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim> static inline cudaError_t 
# 2052
cudaBindSurfaceToArray(const surface< T, dim>  &
# 2053
surf, cudaArray_const_t 
# 2054
array, const cudaChannelFormatDesc &
# 2055
desc) 
# 2057
{ 
# 2058
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 2059
} 
# 2082 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
template< class T, int dim> static inline cudaError_t 
# 2083
cudaBindSurfaceToArray(const surface< T, dim>  &
# 2084
surf, cudaArray_const_t 
# 2085
array) 
# 2087
{ 
# 2088
cudaChannelFormatDesc desc; 
# 2089
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2091
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2092
} 
# 2103 "/usr/local/cuda/bin/..//include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 40 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 278 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((10 / 10000000) % 10)), (('0') + ((10 / 1000000) % 10)), (('0') + ((10 / 100000) % 10)), (('0') + ((10 / 10000) % 10)), (('0') + ((10 / 1000) % 10)), (('0') + ((10 / 100) % 10)), (('0') + ((10 / 10) % 10)), (('0') + (10 % 10)), '.', (('0') + ((0 / 10000000) % 10)), (('0') + ((0 / 1000000) % 10)), (('0') + ((0 / 100000) % 10)), (('0') + ((0 / 10000) % 10)), (('0') + ((0 / 1000) % 10)), (('0') + ((0 / 100) % 10)), (('0') + ((0 / 10) % 10)), (('0') + (0 % 10)), '.', (('0') + ((130 / 10000000) % 10)), (('0') + ((130 / 1000000) % 10)), (('0') + ((130 / 100000) % 10)), (('0') + ((130 / 10000) % 10)), (('0') + ((130 / 1000) % 10)), (('0') + ((130 / 100) % 10)), (('0') + ((130 / 10) % 10)), (('0') + (130 % 10)), ']', '\000'}; 
# 325 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Linux]"); 
# 326
const char *info_arch = ("INFO:arch[]"); 
# 331
const char *info_language_dialect_default = ("INFO:dialect_default[98]"); 
# 347 "CMakeCUDACompilerId.cu"
int main(int argc, char *argv[]) 
# 348
{ 
# 349
int require = 0; 
# 350
require += (info_compiler[argc]); 
# 351
require += (info_platform[argc]); 
# 353
require += (info_version[argc]); 
# 361
require += (info_language_dialect_default[argc]); 
# 362
(void)argv; 
# 363
return require; 
# 364
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
