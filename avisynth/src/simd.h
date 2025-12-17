/*
  simd.h

  This file is part of TCannyMod

  Copyright (C) 2016 Oka Motofumi

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


#ifndef TCANNY_MOD_SIMD_H
#define TCANNY_MOD_SIMD_H

#include <type_traits>
#include <cstdint>
#include <immintrin.h>
#include "tcannymod.h"


#define SFINLINE static __forceinline

using std::is_same_v;

/* -----set-------------------------*/
template <typename T>
SFINLINE T zero()
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_setzero_si128();
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_setzero_si256();
    }
    else if constexpr (is_same_v<T, __m128>) {
        return _mm_setzero_ps();
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_setzero_ps();
    }
}

template <typename Tr, typename Ta>
SFINLINE Tr set1(const Ta x)
{
    if constexpr (is_same_v<Tr, __m128> && is_same_v<Ta, float>) {
        return _mm_set1_ps(x);
    }
    else if constexpr (is_same_v<Tr, __m256> && is_same_v<Ta, float>) {
        return _mm256_set1_ps(x);
    }
    else if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, int8_t>) {
        return _mm_set1_epi8(x);
    } else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, int8_t>) {
        return _mm256_set1_epi8(x);
    }
}



/*---------------load--------------------*/
template <typename T>
SFINLINE T load(const void* p)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_load_ps(reinterpret_cast<const float*>(p));
    }
    else if constexpr (is_same_v<T, __m128i>) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_load_ps(reinterpret_cast<const float*>(p));
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
    }
}


template <typename T>
SFINLINE T loadu(const void* p)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(p));
    } else if constexpr (is_same_v<T, __m128i>) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
}


/*-------------store---------------------*/

template <typename T>
SFINLINE void store(void* p, const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        _mm_store_ps(reinterpret_cast<float*>(p), x);
    }
    else if constexpr (is_same_v<T, __m128i>) {
        _mm_store_si128(reinterpret_cast<__m128i*>(p), x);
    }
    else if constexpr (is_same_v<T, __m256>) {
        _mm256_store_ps(reinterpret_cast<float*>(p), x);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(p), x);
    }
}

template <typename T>
SFINLINE void storeu(void* p, const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        _mm_storeu_ps(reinterpret_cast<float*>(p), x);
    } else if constexpr (is_same_v<T, __m128i>) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), x);
    } else if constexpr (is_same_v<T, __m256>) {
        _mm256_storeu_ps(reinterpret_cast<float*>(p), x);
    } else if constexpr (is_same_v<T, __m256i>) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), x);
    }
}

template <typename T>
SFINLINE void stream(void* p, const T& x)
{
    if constexpr (is_same_v<T, __m128i>) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
    } else if constexpr (is_same_v<T, __m128>) {
        _mm_stream_ps(reinterpret_cast<float*>(p), x);
    } else if constexpr (is_same_v<T, __m256i>) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
    } else if constexpr (is_same_v<T, __m256>) {
        _mm256_stream_ps(reinterpret_cast<float*>(p), x);
    }
}


/*-----------cast--------------------------*/
template <typename Tr, typename Ta>
SFINLINE Tr cast(const Ta& x)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128>) {
        return _mm_castps_si128(x);
    } else if constexpr (is_same_v<Tr, __m128> && is_same_v<Ta, __m128i>) {
        return _mm_castsi128_ps(x);
    } else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256>) {
        return _mm256_castps_si256(x);
    } else if constexpr (is_same_v<Tr, __m256> && is_same_v<Ta, __m256i>) {
        return _mm256_castsi256_ps(x);
    }
}


/*-------------------logical-------------------------------*/
template <typename T>
SFINLINE T _and(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_and_si128(x, y);
    }
    else if constexpr (is_same_v<T, __m128>) {
        return _mm_and_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_and_si256(x, y);
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_and_ps(x, y);
    }
}

template <typename T>
SFINLINE T _or(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_or_si128(x, y);
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_or_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_or_si256(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_or_ps(x, y);
    }
}

template <typename T>
SFINLINE T _andnot(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_andnot_si128(x, y);
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_andnot_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_andnot_si256(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_andnot_ps(x, y);
    }
}

template <typename T>
SFINLINE T _xor(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_xor_si128(x, y);
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_xor_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_xor_si256(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_xor_ps(x, y);
    }
}


/*-----------------shift-----------------------*/
template <typename T>
SFINLINE T srli_i32(const T& x, int n)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_srli_epi32(x, n);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_srli_epi32(x, n);
    }
}


/*------------------arithmetic--------------------*/
template <typename T>
SFINLINE T add(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_add_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_add_ps(x, y);
    }
}

template <typename T>
SFINLINE T sub(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sub_ps(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_sub_ps(x, y);
    }
}

template <typename T>
SFINLINE T mul(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_mul_ps(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_mul_ps(x, y);
    }
}

template <typename T>
SFINLINE T madd(const T& x, const T& y, const T& z)
{
    if constexpr (is_same_v<T, __m128>) {
        return add(mul(x, y), z);
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_fmadd_ps(x, y, z);
    }
}


/*--------------convert-----------------------*/
template <typename Tr, typename Ta>
SFINLINE Tr cvtps_i32(const Ta& x)
{
    if constexpr (is_same_v<Tr, __m128i> && is_same_v<Ta, __m128>) {
        return _mm_cvtps_epi32(x);
    }
    else if constexpr (is_same_v<Tr, __m256i> && is_same_v<Ta, __m256>) {
        return _mm256_cvtps_epi32(x);
    }
}

template <typename T>
SFINLINE T cvtu8_ps(const uint8_t* ptr)
{
    if constexpr (is_same_v<T, __m128>) {
        __m128i t = _mm_cvtsi32_si128(*(reinterpret_cast<const int32_t*>(ptr)));
        t = _mm_cvtepu8_epi32(t);
        return _mm_cvtepi32_ps(t);
    }
    else if constexpr (is_same_v<T, __m256>) {
        __m128i t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
        __m256i t1 = _mm256_cvtepu8_epi32(t0);
        return _mm256_cvtepi32_ps(t1);
    }
}

template <typename T>
SFINLINE T cvti32_u8(const T& a, const T& b, const T& c, const T& d)
{
    if constexpr (is_same_v<T, __m128i>) {
        __m128i x = _mm_packs_epi32(a, b);
        __m128i y = _mm_packs_epi32(c, d);
        return _mm_packus_epi16(x, y);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        __m256i t0 = _mm256_packs_epi32(a, c);
        __m256i t1 = _mm256_packs_epi32(b, d);
        t0 = _mm256_permute4x64_epi64(t0, _MM_SHUFFLE(3, 1, 2, 0));
        t1 = _mm256_permute4x64_epi64(t1, _MM_SHUFFLE(3, 1, 2, 0));
        return _mm256_packus_epi16(t0, t1);
    }
}


/*-----------------math-----------------------*/
template <typename T>
SFINLINE T max(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_max_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_max_ps(x, y);
    }
}

template <typename T>
SFINLINE T rcp_ps(const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_rcp_ps(x);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_rcp_ps(x);
    }
}

template <typename T>
SFINLINE T sqrt(const T& x)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_sqrt_ps(x);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_sqrt_ps(x);
    }
}


/*-----------compare-------------------------------*/

template <typename T>
SFINLINE T cmpeq_32(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_cmpeq_epi32(x, y);
    }
    else if constexpr (is_same_v<T, __m128>) {
        return _mm_cmpeq_ps(x, y);
    }
    else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_cmpeq_epi32(x, y);
    }
    else if constexpr (is_same_v<T, __m256>) {
        return _mm256_cmpeq_ps(x, y);
    }
}

template <typename T>
SFINLINE T cmpgt_32(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_cmpgt_epi32(x, y);
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_cmpgt_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_cmpgt_epi32(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    }
}

template <typename T>
SFINLINE T cmplt_32(const T& x, const T& y)
{
    return cmpgt_32(y, x);
}

template <typename T>
SFINLINE T cmpge_32(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _or(cmpeq_32(x, y), cmpgt_32(x, y));
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_cmpge_ps(x, y);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _or(cmpeq_32(x, y), cmpgt_32(x, y));
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
    }
}

template <typename T>
SFINLINE T cmple_32(const T& x, const T& y)
{
    return cmpge_32(y, x);
}

template <typename T>
SFINLINE T cmpord_ps(const T& x, const T& y)
{
    if constexpr (is_same_v<T, __m128>) {
        return _mm_cmpord_ps(x, y);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_cmp_ps(x, y, _CMP_ORD_Q);
    }
}


/*----------------misc-----------------------------*/
template <typename T>
SFINLINE T blendv(const T& x, const T& y, const T& mask)
{
    if constexpr (is_same_v<T, __m128i>) {
        return _mm_blendv_epi8(x, y, mask);
    } else if constexpr (is_same_v<T, __m128>) {
        return _mm_blendv_ps(x, y, mask);
    } else if constexpr (is_same_v<T, __m256i>) {
        return _mm256_blendv_epi8(x, y, mask);
    } else if constexpr (is_same_v<T, __m256>) {
        return _mm256_blendv_ps(x, y, mask);
    }
}


template <typename T>
SFINLINE T abs(const T& val)
{
    return max(val, sub(zero<T>(), val));
}

template <typename T>
SFINLINE T rcp_hq(const T& x)
{
    T rcp = rcp_ps(x);
    T t = mul(mul(x, rcp), rcp);
    rcp = add(rcp, rcp);
    return sub(rcp, t);
}


#endif // TCANNY_MOD_SIMD_H

