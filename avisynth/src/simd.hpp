/*
  simd.hpp

  This file is part of TCannyMod

  Copyright (C) 2026 Oka Motofumi

  Authors: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/

#ifndef TCM_SIMD_HPP
#define TCM_SIMD_HPP

#include <immintrin.h>
#include <type_traits>


#ifndef SFINLINE
#if defined(_WIN32)
#define SFINLINE static __forceinline
#else
#define SFINLINE static inline __attribute__((always_inline))
#endif
#endif

using std::is_same_v;


template <typename T>
SFINLINE T zero()
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_setzero_ps();
    }
    else if constexpr (is_same_v<T, __m128i>) {
        return _mm_setzero_si128();
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_setzero_ps();
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_setzero_si256();
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_setzero_ps();
    }
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_setzero_si512();
    }
#endif
#endif
}


template <typename T>
SFINLINE T load(const void* p)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    }
    else if constexpr (is_same_v<T, __m128>) {
        return _mm_load_ps(reinterpret_cast<const float*>(p));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_load_ps(reinterpret_cast<const float*>(p));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_load_si512(p);
    } else if constexpr (is_same_v<T, __m512>) {
        return _mm512_load_ps(p);
    }
#endif
#endif
}

template <typename T>
SFINLINE T loadu(const void* p)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(p));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_loadu_si512(p);
    } else if constexpr (is_same_v<T, __m512>) {
        return _mm512_loadu_ps(p);
    }
#endif
#endif
}


template <typename T>
SFINLINE void store(void* p, T& v)
{
    if constexpr (is_same_v<T, __m128i>) {
        _mm_store_si128(reinterpret_cast<__m128i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m128>) {
        _mm_store_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m256>) {
        _mm256_store_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        _mm512_store_si512(p, v);
    }
    else if constexpr (is_same_v<T, __m512>) {
        _mm512_store_ps(p, v);
    }
#endif
#endif
}


template <typename T>
SFINLINE void storeu(void* p, T& v)
{
    if constexpr (is_same_v<T, __m128i>) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m128>) {
        _mm_store_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m256>) {
        _mm256_storeu_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        _mm512_storeu_si512(p, v);
    }
    else if constexpr (is_same_v<T, __m512>) {
        _mm512_storeu_ps(p, v);
    }
#endif
#endif
}

template <typename T>
SFINLINE void stream(void* p, T& v)
{
    if constexpr (is_same_v<T, __m128i>) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m128>) {
        _mm_stream_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(p), v);
    }
    else if constexpr (is_same_v<T, __m256>) {
        _mm256_stream_ps(reinterpret_cast<float*>(p), v);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        _mm512_stream_si512(p, v);
    }
    else if constexpr (is_same_v<T, __m512>) {
        _mm512_stream_ps(p, v);
    }
#endif
#endif
}

template <typename T>
SFINLINE void storel(void* p, const T& x)
{
    if constexpr (is_same_v<T, __m128i>) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(p), x);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        __m128i t = _mm256_castsi256_si128(x);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), t);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m256i>) {
        __m256i t = _mm512_castsi512_si256(x);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), t);
    }
#endif
#endif
}


template <typename T>
SFINLINE T cvtepu8_ps(const void* ptr)
{
    if constexpr (is_same_v<T, __m128>) {
        __m128i t = _mm_cvtsi32_si128(*(reinterpret_cast<const int32_t*>(ptr)));
        t = _mm_cvtepu8_epi32(t);
        return _mm_cvtepi32_ps(t);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        __m128i t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
        __m256i t1 = _mm256_cvtepu8_epi32(t0);
        return _mm256_cvtepi32_ps(t1);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        __m128i t0 = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
        __m512i t1 = _mm512_cvtepu8_epi32(t0);
        return _mm512_cvtepi32_ps(t1);
    }
#endif
#endif
}

template <typename T>
SFINLINE T cvtepu16_ps(const void* ptr)
{
    if constexpr (is_same_v<T, __m128>) {
        __m128i t = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
        t = _mm_cvtepu16_epi32(t);
        return _mm_cvtepi32_ps(t);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        __m128i t0 = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
        __m256i t1 = _mm256_cvtepu16_epi32(t0);
        return _mm256_cvtepi32_ps(t1);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        __m256i t0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        __m512i t1 = _mm512_cvtepu16_epi32(t0);
        return _mm512_cvtepi32_ps(t1);
    }
#endif
#endif
}

template <typename T0, typename T1>
SFINLINE T0 cvtepuX_ps(const void* ptr)
{
    if constexpr (is_same_v<T0, __m128> && is_same_v<T1, uint8_t>) {
        return cvtepu8_ps<__m128>(ptr);
    }
    else if constexpr (is_same_v<T0, __m128> && is_same_v<T1, uint16_t>) {
        return cvtepu16_ps<__m128>(ptr);
    }
    else if constexpr (is_same_v<T0, __m128> && is_same_v<T1, float>) {
        return _mm_load_ps(reinterpret_cast<const float*>(ptr));
    }
#ifdef __AVX2__
    else if constexpr(is_same_v<T0, __m256> && is_same_v<T1, uint8_t>) {
        return cvtepu8_ps<__m256>(ptr);
    }
    else if constexpr (is_same_v<T0, __m256> && is_same_v<T1, uint16_t>) {
        return cvtepu16_ps<__m256>(ptr);
    }
    else if constexpr (is_same_v<T0, __m256> && is_same_v<T1, float>) {
        return _mm256_load_ps(reinterpret_cast<const float*>(ptr));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T0, __m512> && is_same_v<T1, uint8_t>) {
        return cvtepu8_ps<__m512>(ptr);
    }
    else if constexpr (is_same_v<T0, __m512> && is_same_v<T1, uint16_t>) {
        return cvtepu16_ps<__m512>(ptr);
    }
    else if constexpr (is_same_v<T0, __m512> && is_same_v<T1, float>) {
        return _mm512_load_ps(ptr);
    }
#endif
#endif
}

#ifdef __AVX2__
static const __m256i idx_avx2 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
#ifdef __AVX512F__
static const __m512i idx_avx512 = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
static const __m512i idx_avx512_2 = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
#endif
#endif

template <typename T>
SFINLINE T cvtps_epu16(const float* ptr)
{
    if constexpr (is_same_v<T, __m128i>) {
        __m128i s0 = _mm_cvtps_epi32(_mm_load_ps(ptr));
        __m128i s1 = _mm_cvtps_epi32(_mm_load_ps(ptr + 4));
        return _mm_packus_epi32(s0, s1);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        __m256i s0 = _mm256_cvtps_epi32(_mm256_load_ps(ptr));
        __m256i s1 = _mm256_cvtps_epi32(_mm256_load_ps(ptr + 8));
        s0 = _mm256_packus_epi32(s0, s1);
        return _mm256_permute4x64_epi64(s0, _MM_SHUFFLE(3,1,2,0));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        __m512i s0 = _mm512_cvtps_epi32(_mm512_load_ps(ptr));
        __m512i s1 = _mm512_cvtps_epi32(_mm512_load_ps(ptr + 16));
        s0 = _mm512_packus_epi32(s0, s1);
        return _mm512_permutexvar_epi64(idx_avx512, s0);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cvtps_epu16(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128>) {
        __m128i s0 = _mm_cvtps_epi32(x);
        return _mm_packus_epi32(s0, _mm_cvtps_epi32(y));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256>) {
        __m256i s0 = _mm256_cvtps_epi32(x);
        s0 = _mm256_packus_epi32(s0, _mm256_cvtps_epi32(y));
        return _mm256_permute4x64_epi64(s0, _MM_SHUFFLE(3, 1, 2, 0));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, __m512i> && is_same_v<Ta, __m512>) {
        __m512i s0 = _mm512_cvtps_epi32(x);
        s0 = _mm512_packus_epi32(s0, _mm512_cvtps_epi32(y));
        return _mm512_permutexvar_epi64(idx_avx512, s0);
    }
#endif
#endif
}

template <typename T>
SFINLINE T cvtps_epu8(const float* ptr)
{
    if constexpr (is_same_v<T, __m128i>) {
        __m128i s0 = _mm_cvtps_epi32(_mm_load_ps(ptr));
        __m128i s1 = _mm_cvtps_epi32(_mm_load_ps(ptr + 4));
        __m128i t0 = _mm_packus_epi32(s0, s1);
        s0 = _mm_cvtps_epi32(_mm_load_ps(ptr + 8));
        s1 = _mm_cvtps_epi32(_mm_load_ps(ptr + 12));
        s0 = _mm_packus_epi32(s0, s1);
        return _mm_packus_epi16(t0, s0);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        __m256i s0 = _mm256_cvtps_epi32(_mm256_load_ps(ptr));
        __m256i s1 = _mm256_cvtps_epi32(_mm256_load_ps(ptr + 8));
        __m256i t0 = _mm256_packus_epi32(s0, s1);
        s0 = _mm256_cvtps_epi32(_mm256_load_ps(ptr + 16));
        s1 = _mm256_cvtps_epi32(_mm256_load_ps(ptr + 24));
        s0 = _mm256_packus_epi32(s0, s1);
        s0 = _mm256_packus_epi16(t0, s0);
        return _mm256_permutevar8x32_epi32(s0, idx_avx2);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        __m512i s0 = _mm512_cvtps_epi32(_mm512_load_ps(ptr));
        __m512i s1 = _mm512_cvtps_epi32(_mm512_load_ps(ptr + 16));
        __m512i t0 = _mm512_packus_epi32(s0, s1);
        s0 = _mm512_cvtps_epi32(_mm512_load_ps(ptr + 32));
        s1 = _mm512_cvtps_epi32(_mm512_load_ps(ptr + 48));
        s0 = _mm512_packus_epi32(s0, s1);
        s1 = _mm512_packus_epi16(t0, s0);
        return _mm512_permutexvar_epi32(s1, idx_avx512_2);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cvtps_epu8(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128>) {
        __m128i s0 = _mm_cvtps_epi32(x);
        s0 = _mm_packus_epi32(s0, _mm_cvtps_epi32(y));
        return _mm_packus_epi16(s0, s0);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m256>) {
        __m256i s0 = _mm256_cvtps_epi32(x);
        s0 = _mm256_packus_epi32(s0, _mm256_cvtps_epi32(y));
        s0 = _mm256_packus_epi16(s0, s0);
        return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(s0, idx_avx2));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m512>) {
        __m512i s0 = _mm512_cvtps_epi32(x);
        s0 = _mm512_packus_epi32(s0, _mm512_cvtps_epi32(y));
        s0 = _mm512_packus_epi16(s0, s0);
        s0 = _mm512_permutexvar_epi32(idx_avx512_2, s0);
        return _mm512_castsi512_si256(s0);
    }
#endif
#endif
}


#ifdef __AVX512F__
SFINLINE __m512i cvtps_epu8_2(const __m512& a, const __m512& b,
    const __m512& c, const __m512& d)
{
    __m512i s0 = _mm512_cvtps_epi32(a);
    s0 = _mm512_packus_epi32(s0, _mm512_cvtps_epi32(b));
    __m512i s1 = _mm512_cvtps_epi32(c);
    s1 = _mm512_packus_epi32(s1, _mm512_cvtps_epi32(d));
    s0 = _mm512_packus_epi16(s0, s1);
    return _mm512_permutexvar_epi32(idx_avx512_2, s0);
}
#endif


template <typename T>
SFINLINE T set1_ps(const float x)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_set1_ps(x);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_set1_ps(x);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_set1_ps(x);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fadd(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_add_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_add_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_add_ps(x, y);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fsub(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sub_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_sub_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_sub_ps(x, y);
    }
#endif
#endif
}


template <typename T>
SFINLINE T fmul(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_mul_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_mul_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_mul_ps(x, y);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fdiv(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_div_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_div_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_div_ps(x, y);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fmadd(const T& x, const T& y, const T& z)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_add_ps(_mm_mul_ps(x, y), z);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_fmadd_ps(x, y, z);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_fmadd_ps(x, y, z);
    }
#endif
#endif
}


template <typename T>
SFINLINE T fnmadd(const T& x, const T& y, const T& z)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sub_ps(z, _mm_mul_ps(x, y));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_fnmadd_ps(x, y, z);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_fnmadd_ps(x, y, z);
    }
#endif
#endif
}


template <typename T>
SFINLINE T fmsub(const T& x, const T& y, const T& z)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sub_ps(_mm_mul_ps(x, y), z);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_fmsub_ps(x, y, z);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_fmsub_ps(x, y, z);
    }
#endif
#endif
}


template <typename T>
SFINLINE T fmin(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_min_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_min_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_min_ps(x, y);
    }
#endif
#endif
}


template <typename T>
SFINLINE T fmax(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_max_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_max_ps(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_max_ps(x, y);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fabs(const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        return fmax<__m128>(x, fsub(zero<__m128>(), x));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return fmax<__m256>(x, fsub(zero<__m256>(), x));
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_abs_ps(x);
    }
#endif
#endif
}

template <typename T>
SFINLINE T fsqrt(const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sqrt_ps(x);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_sqrt_ps(x);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_sqrt_ps(x);
    }
#endif
#endif
}


template <typename T>
SFINLINE T rcp(const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        __m128 t = _mm_rcp_ps(x);
        return sub(add(t, t), mul(x, mul(t, t)));
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        static const __m256 two = _mm256_set1_ps(2.0f);
        __m256 t = _mm256_rcp_ps(x);
        return mul(t, _mm256_fnmadd_ps(x, t, two));
    }
#ifdef __AVX512F__
    else if constexpr (std::is_same_v<T, __m512>) {
        static const __m512 two = _mm512_set1_ps(2.0f);
        __m512 t = _mm512_pcp14_ps(x);
        return mul(t, _mm512_fnmadd_ps(x, t, two));
    }
#endif
#endif
}

template <typename T>
SFINLINE T _and(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_and_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m128i>) {
        return _mm_and_si128(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_and_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_and_si256(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_and_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_and_si512(x, y);
    }
#endif
#endif
}


template <typename T>
SFINLINE T _or(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_or_ps(x, y);
    } else if constexpr (is_same_v<T, __m128i>) {
        return _mm_or_si128(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_or_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_or_si256(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512>) {
        return _mm512_or_ps(x, y);
    } else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_or_si512(x, y);
    }
#endif
#endif
}


template <typename T>
SFINLINE T blendv(const T& x, const T& y, const T& mask)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_blendv_ps(x, y, mask);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_blendv_ps(x, y, mask);
    }
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr castps_si(const Ta& x)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128>) {
        return _mm_castps_si128(x);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256>) {
        return _mm256_castps_si256(x);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, __m512i> && is_same_v<Ta, __m512>) {
        return _mm512_castps_si512(x);
    }
#endif
#endif
}

template <typename Tr, typename Ta>
SFINLINE Tr castsi_ps(const Ta& x)
{
    if constexpr (is_same_v<Tr, __m128> && is_same_v<Ta, __m128i>) {
        return _mm_castsi128_ps(x);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256> && is_same_v<Ta, __m256i>) {
        return _mm256_castsi256_ps(x);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, __m512> && is_same_v<Ta, __m512i>) {
        return _mm512_castsi512_ps(x);
    }
#endif
#endif
}


template <typename T>
SFINLINE T srli_epi32(const T& x, int v)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_srli_epi32(x, v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_srli_epi32(x, v);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_srli_epi32(x, v);
    }
#endif
#endif
}


template <typename T>
SFINLINE T srli_epi16(const T& x, int v)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_srli_epi16(x, v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_srli_epi16(x, v);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<T, __m512i>) {
        return _mm512_srli_epi16(x, v);
    }
#endif
#endif
}


template <typename T>
SFINLINE T srli_epi8(const T& x, int v)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_srli_epi8(x, v);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_srli_epi8(x, v);
    }
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmpeq_epi32(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        return _mm_cmpeq_epi32(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        return _mm256_cmpeq_epi32(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint16_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmpeq_epi32_mask(x, y);
    }
#endif
#endif
}

template <typename Tr, typename Ta>
SFINLINE Tr cmpeq_epi16(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        return _mm_cmpeq_epi16(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        return _mm256_cmpeq_epi16(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint32_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmpeq_epi16_mask(x, y);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmpeq_epi8(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        return _mm_cmpeq_epi8(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        return _mm256_cmpeq_epi8(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint64_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmpeq_epi8_mask(x, y);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmplt_epi32(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        return _mm_cmplt_epi32(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        return _mm256_cmplt_epi32(x, y);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint16_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmplt_epi32_mask(x, y);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmplt_epu16(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        __m128i t = _mm_cmpeq_epi8(x, x);
        return _mm_xor_si128(_mm_cmpeq_epi16(_mm_max_epu16(x, y), x), t);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        __m256i t = _mm256_cmpeq_epi8(x, x);
        return _mm256_xor_si256(_mm256_cmpeq_epi16(_mm256_max_epu16(x, y), x), t);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint32_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmplt_epu16_mask(x, y);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmplt_epu8(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128i>) {
        __m128i t = _mm_cmpeq_epi8(x, x);
        return _mm_xor_si128(_mm_cmpeq_epi8(_mm_max_epu8(x, y), x), t);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256i>) {
        __m256i t = _mm256_cmpeq_epi8(x, x);
        return _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(x, y), x), t);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint64_t> && is_same_v<Ta, __m512i>) {
        return _mm512_cmplt_epu8_mask(x, y);
    }
#endif
#endif
}


template <typename Tr, typename Ta>
SFINLINE Tr cmplt_ps(const Ta& x, const Ta& y)
{
    if constexpr (is_same_v<Tr, __m128> && is_same_v<Tr, __m128>) {
        return _mm_cmplt_ps(x, y);
    }
#ifdef __AVX2__
    else if constexpr (is_same_v<Tr, __m256> && is_same_v<Ta, __m256>) {
        return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
    }
#ifdef __AVX512F__
    else if constexpr (is_same_v<Tr, uint16_t> && is_same_v<Ta, __m512>) {
        return _mm512_cmplt_ps_mask(x, y);
    }
#endif
#endif
}





#endif  // TCM_SIMD_HPP