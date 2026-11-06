/*
 * RGB MoCap compile-time adapter for aarch64 Linux. Added by RGB MoCap; not NVIDIA source.
 *
 * Math/SIMD.h includes <immintrin.h>, which does not exist on ARM. CMake puts this folder
 * on the include path for aarch64 builds only, so the numerical C++ source stays unedited.
 * The pinned sse2neon header maps the SSE intrinsics it uses to NEON.
 */
#pragma once

#include <sse2neon.h>

#include <stdint.h>

/*
 * AVX's variable permute, used once by Math/Vector.inl; sse2neon only covers SSE. As on x86,
 * bits 1:0 of each 32-bit control lane select the source lane.
 */
static inline __m128 _mm_permutevar_ps( __m128 a, __m128i control )
{
    float values[4];
    int32_t selectors[4];
    _mm_storeu_ps( values, a );
    _mm_storeu_si128( reinterpret_cast<__m128i*>( selectors ), control );
    return _mm_setr_ps( values[selectors[0] & 3], values[selectors[1] & 3], values[selectors[2] & 3], values[selectors[3] & 3] );
}
