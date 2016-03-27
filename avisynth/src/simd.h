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

#include <cstdint>
#include <type_traits>
#include <immintrin.h>

#define SFINLINE static __forceinline

enum arch_t {
    HAS_SSE2,
    HAS_SSE41,
    HAS_AVX2,
};


template <typename T>
T zero();

template <>
SFINLINE __m128i zero<__m128i>()
{
    return _mm_setzero_si128();
}

template <>
SFINLINE __m128 zero<__m128>()
{
    return _mm_setzero_ps();
}

template <>
SFINLINE __m256i zero<__m256i>()
{
    return _mm256_setzero_si256();
}

template <>
SFINLINE __m256 zero<__m256>()
{
    return _mm256_setzero_ps();
}

template <typename T>
T load(const float* p);

template <>
SFINLINE __m128 load<__m128>(const float* p)
{
    return _mm_load_ps(p);
}

template <>
SFINLINE __m256 load<__m256>(const float* p)
{
    return _mm256_load_ps(p);
}

template <typename T>
T load(const uint8_t*);

template <>
SFINLINE __m128i load(const uint8_t* p)
{
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
SFINLINE __m256i load(const uint8_t* p)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}

template <typename T>
T load(const int32_t*);

template <>
SFINLINE __m128i load(const int32_t* p)
{
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
SFINLINE __m256i load(const int32_t* p)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}

template <typename T>
T loadu(const float* p);

template <>
SFINLINE __m128 loadu<__m128>(const float* p)
{
    return _mm_loadu_ps(p);
}

template <>
SFINLINE __m256 loadu<__m256>(const float* p)
{
    return _mm256_loadu_ps(p);
}

template <typename T>
T loadu(const uint8_t* p);

template <>
SFINLINE __m128i loadu<__m128i>(const uint8_t* p)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
SFINLINE __m256i loadu<__m256i>(const uint8_t* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

template <typename T>
T loadu(const int32_t* p);

template <>
SFINLINE __m128i loadu<__m128i>(const int32_t* p)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
SFINLINE __m256i loadu<__m256i>(const int32_t* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

SFINLINE __m128i cvtps_i32(const __m128& x)
{
    return _mm_cvtps_epi32(x);
}

SFINLINE __m256i cvtps_i32(const __m256& x)
{
    return _mm256_cvtps_epi32(x);
}

//template <typename T, arch_t ARCH>
//T cvtu8_i32(const uint8_t* ptr);
//
//template <>
//SFINLINE __m128i cvtu8_i32<__m128i, HAS_SSE2>(const uint8_t* ptr)
//{
//    const int32_t* p32 = reinterpret_cast<const int32_t*>(ptr);
//    __m128i t = _mm_cvtsi32_si128(p32[0]);
//    __m128i z = zero<__m128i>();
//    t = _mm_unpacklo_epi8(t, z);
//    return _mm_unpacklo_epi16(t, z);
//}
//
//template <>
//SFINLINE __m128i cvtu8_i32<__m128i, HAS_SSE41>(const uint8_t* ptr)
//{
//    const int32_t* p32 = reinterpret_cast<const int32_t*>(ptr);
//    __m128i t = _mm_cvtsi32_si128(p32[0]);
//    return _mm_cvtepu8_epi32(t);
//}
//
//template <>
//SFINLINE __m256i cvtu8_i32<__m256i, HAS_AVX2>(const uint8_t* ptr)
//{
//    const int64_t* p64 = reinterpret_cast<const int64_t*>(ptr);
//#if defined(__WIN64)
//    __m128i t0 = _mm_cvtsi64_si128(p64[0]);
//#else
//    __m128i t0 = _mm_set1_epi64x(p64[0]);
//#endif
//    return _mm256_cvtepu8_epi32(t0);
//}

template <typename T, arch_t ARCH>
T cvt_to_float(const uint8_t* ptr);

template <>
SFINLINE __m128 cvt_to_float<__m128, HAS_SSE2>(const uint8_t* ptr)
{
    const int32_t* p32 = reinterpret_cast<const int32_t*>(ptr);
    __m128i t = _mm_cvtsi32_si128(p32[0]);
    __m128i z = zero<__m128i>();
    t = _mm_unpacklo_epi8(t, z);
    t = _mm_unpacklo_epi16(t, z);
    return _mm_cvtepi32_ps(t);
}

template <>
SFINLINE __m128 cvt_to_float<__m128, HAS_SSE41>(const uint8_t* ptr)
{
    const int32_t* p32 = reinterpret_cast<const int32_t*>(ptr);
    __m128i t = _mm_cvtsi32_si128(p32[0]);
    t = _mm_cvtepu8_epi32(t);
    return _mm_cvtepi32_ps(t);
}

template <>
SFINLINE __m256 cvt_to_float<__m256, HAS_AVX2>(const uint8_t* ptr)
{
    __m128i t0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
    __m256i t1 = _mm256_cvtepu8_epi32(t0);
    return _mm256_cvtepi32_ps(t1);
}


SFINLINE __m128i
cvti32_i8(const __m128i& a, const __m128i& b, const __m128i& c, const __m128i& d)
{
    __m128i x = _mm_packs_epi32(a, b);
    __m128i y = _mm_packs_epi32(c, d);
    return _mm_packs_epi16(x, y);
}


SFINLINE __m256i
cvti32_i8(const __m256i& a, const __m256i& b, const __m256i& c, const __m256i& d)
{
    __m256i t0 = _mm256_packs_epi32(a, c);
    __m256i t1 = _mm256_packs_epi32(b, d);
    t0 = _mm256_permute4x64_epi64(t0, _MM_SHUFFLE(3, 1, 2, 0));
    t1 = _mm256_permute4x64_epi64(t1, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm256_packs_epi16(t0, t1);
}

SFINLINE __m128i
cvti32_u8(const __m128i& a, const __m128i& b, const __m128i& c, const __m128i& d)
{
    __m128i x = _mm_packs_epi32(a, b);
    __m128i y = _mm_packs_epi32(c, d);
    return _mm_packus_epi16(x, y);
}


SFINLINE __m256i
cvti32_u8(const __m256i& a, const __m256i& b, const __m256i& c, const __m256i& d)
{
    __m256i t0 = _mm256_packs_epi32(a, c);
    __m256i t1 = _mm256_packs_epi32(b, d);
    t0 = _mm256_permute4x64_epi64(t0, _MM_SHUFFLE(3, 1, 2, 0));
    t1 = _mm256_permute4x64_epi64(t1, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm256_packus_epi16(t0, t1);
}

template <typename T>
T set1_ps(const float& x);

template <>
SFINLINE __m128 set1_ps<__m128>(const float& x)
{
    return _mm_set_ps1(x);
}

template <>
SFINLINE __m256 set1_ps<__m256>(const float& x)
{
    return _mm256_set1_ps(x);
}

template <typename T>
T set1_i8(const int8_t&);

template <>
SFINLINE __m128i set1_i8<__m128i>(const int8_t& x)
{
    return _mm_set1_epi8(x);
}

template <>
SFINLINE __m256i set1_i8<__m256i>(const int8_t& x)
{
    return _mm256_set1_epi8(x);
}

SFINLINE __m128 and_ps(const __m128& x, const __m128& y)
{
    return _mm_and_ps(x, y);
}

SFINLINE __m256 and_ps(const __m256& x, const __m256& y)
{
    return _mm256_and_ps(x, y);
}

SFINLINE __m128i and_si(const __m128i& x, const __m128i& y)
{
    return _mm_and_si128(x, y);
}

SFINLINE __m256i and_si(const __m256i& x, const __m256i& y)
{
    return _mm256_and_si256(x, y);
}

SFINLINE __m128 or_ps(const __m128& x, const __m128& y)
{
    return _mm_or_ps(x, y);
}

SFINLINE __m256 or_ps(const __m256& x, const __m256& y)
{
    return _mm256_or_ps(x, y);
}


SFINLINE __m128i or_si(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

SFINLINE __m256i or_si(const __m256i& x, const __m256i& y)
{
    return _mm256_or_si256(x, y);
}

SFINLINE __m128 andnot_ps(const __m128& x, const __m128& y)
{
    return _mm_andnot_ps(x, y);
}

SFINLINE __m256 andnot_ps(const __m256& x, const __m256& y)
{
    return _mm256_andnot_ps(x, y);
}

SFINLINE __m128i andnot_si(const __m128i& x, const __m128i& y)
{
    return _mm_andnot_si128(x, y);
}

SFINLINE __m256i andnot_si(const __m256i& x, const __m256i& y)
{
    return _mm256_andnot_si256(x, y);
}

SFINLINE __m128 xor_ps(const __m128& x, const __m128& y)
{
    return _mm_xor_ps(x, y);
}

SFINLINE __m256 xor_ps(const __m256& x, const __m256& y)
{
    return _mm256_xor_ps(x, y);
}

SFINLINE __m128i xor_si(const __m128i& x, const __m128i& y)
{
    return _mm_xor_si128(x, y);
}

SFINLINE __m256i xor_si(const __m256i& x, const __m256i& y)
{
    return _mm256_xor_si256(x, y);
}

SFINLINE __m128 max(const __m128& x, const __m128& y)
{
    return _mm_max_ps(x, y);
}

SFINLINE __m256 max(const __m256& x, const __m256& y)
{
    return _mm256_max_ps(x, y);
}

SFINLINE __m128 add_ps(const __m128& x, const __m128& y)
{
    return _mm_add_ps(x, y);
}

SFINLINE __m256 add_ps(const __m256& x, const __m256& y)
{
    return _mm256_add_ps(x, y);
}

SFINLINE __m128 sub_ps(const __m128& x, const __m128& y)
{
    return _mm_sub_ps(x, y);
}

SFINLINE __m256 sub_ps(const __m256& x, const __m256& y)
{
    return _mm256_sub_ps(x, y);
}

SFINLINE __m128 mul_ps(const __m128& x, const __m128& y)
{
    return _mm_mul_ps(x, y);
}

SFINLINE __m256 mul_ps(const __m256& x, const __m256& y)
{
    return _mm256_mul_ps(x, y);
}

SFINLINE __m128 madd_ps(const __m128& x, const __m128& y, const __m128& z)
{
    return add_ps(mul_ps(x, y), z);
}

SFINLINE __m256 madd_ps(const __m256& x, const __m256& y, const __m256& z)
{
    return _mm256_fmadd_ps(x, y, z);
}

template <typename T>
void store_ps(float* p, const T& x) {}

template <>
SFINLINE void store_ps<__m128>(float* p, const __m128& x)
{
    _mm_store_ps(p, x);
}

template <>
SFINLINE void store_ps<__m256>(float* p, const __m256& x)
{
    _mm256_store_ps(p, x);
}

template <typename T>
void storeu_ps(float* p, const T& x) {}

template <>
SFINLINE void storeu_ps<__m128>(float* p, const __m128& x)
{
    _mm_storeu_ps(p, x);
}

template <>
SFINLINE void storeu_ps<__m256>(float* p, const __m256& x)
{
    _mm256_storeu_ps(p, x);
}

template <typename T>
void stream_ps(float* p, const T& x) {}

template <>
SFINLINE void stream_ps<__m128>(float* p, const __m128& x)
{
    _mm_stream_ps(p, x);
}

template <>
SFINLINE void stream_ps<__m256>(float* p, const __m256& x)
{
    _mm256_stream_ps(p, x);
}

template <typename T>
void stream_si(uint8_t* p, const T& x) {}

template <>
SFINLINE void stream_si<__m128i>(uint8_t* p, const __m128i& x)
{
    return _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
}

template <>
SFINLINE void stream_si<__m256i>(uint8_t* p, const __m256i& x)
{
    return _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
}

template <typename T>
void stream_si(int32_t* p, const T& x) {}

template <>
SFINLINE void stream_si<__m128i>(int32_t* p, const __m128i& x)
{
    return _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
}

template <>
SFINLINE void stream_si<__m256i>(int32_t* p, const __m256i& x)
{
    return _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
}

SFINLINE __m128i cmpeq_i32(const __m128i& x, const __m128i& y)
{
    return _mm_cmpeq_epi32(x, y);
}

SFINLINE __m256i cmpeq_i32(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpeq_epi32(x, y);
}

SFINLINE __m128i castps_si(const __m128& x)
{
    return _mm_castps_si128(x);
}

SFINLINE __m256i castps_si(const __m256& x)
{
    return _mm256_castps_si256(x);
}

SFINLINE __m128 castsi_ps(const __m128i& x)
{
    return _mm_castsi128_ps(x);
}

SFINLINE __m256 castsi_ps(const __m256i& x)
{
    return _mm256_castsi256_ps(x);
}

SFINLINE __m128 cmplt_ps(const __m128& x, const __m128& y)
{
    return _mm_cmplt_ps(x, y);
}

SFINLINE __m256 cmplt_ps(const __m256& x, const __m256& y)
{
    return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
}

SFINLINE __m128 cmpge_ps(const __m128& x, const __m128& y)
{
    return _mm_cmpge_ps(x, y);
}

SFINLINE __m256 cmpge_ps(const __m256& x, const __m256& y)
{
    return _mm256_cmp_ps(x, y, _CMP_GE_OQ);
}

SFINLINE __m128 cmpord_ps(const __m128& x, const __m128& y)
{
    return _mm_cmpord_ps(x, y);
}

SFINLINE __m256 cmpord_ps(const __m256& x, const __m256& y)
{
    return _mm256_cmp_ps(x, y, _CMP_ORD_Q);
}

SFINLINE __m128i srli_i32(const __m128i& x, int n)
{
    return _mm_srli_epi32(x, n);
}

SFINLINE __m256i srli_i32(const __m256i& x, int n)
{
    return _mm256_srli_epi32(x, n);
}

SFINLINE __m128 blendv(const __m128& x, const __m128& y, const __m128& mask)
{
    return or_ps(and_ps(mask, y), andnot_ps(mask, x));
}

SFINLINE __m256 blendv(const __m256&x, const __m256& y, const __m256& mask)
{
    return _mm256_blendv_ps(x, y, mask);
}

SFINLINE __m128 rcp_ps(const __m128& x)
{
    return _mm_rcp_ps(x);
}

SFINLINE __m256 rcp_ps(const __m256& x)
{
    return _mm256_rcp_ps(x);
}

template <typename T>
SFINLINE T rcp_hq_ps(const T& x)
{
    T rcp = rcp_ps(x);
    T t = mul_ps(mul_ps(x, rcp), rcp);
    rcp = add_ps(rcp, rcp);
    return sub_ps(rcp, t);
}

SFINLINE __m128 sqrt_ps(const __m128& x)
{
    return _mm_sqrt_ps(x);
}

SFINLINE __m256 sqrt_ps(const __m256& x)
{
    return _mm256_sqrt_ps(x);
}

template <typename T>
SFINLINE T abs_ps(const T& val)
{
    return max(val, sub_ps(zero<T>(), val));
}

SFINLINE void cvt_and_output(const __m128i& x, uint8_t* dstp)
{
    __m128i t = _mm_packs_epi32(x, x);
    t = _mm_packus_epi16(t, t);
    int32_t dst = _mm_cvtsi128_si32(t);
    _mm_stream_si32(reinterpret_cast<int32_t*>(dstp), dst);
}

SFINLINE void cvt_and_output(const __m256i& x, uint8_t* dstp)
{
    __m256i t0 = _mm256_permute2f128_si256(x, x, 1);
    t0 = _mm256_packs_epi32(x, t0);
    t0 = _mm256_packus_epi16(t0, t0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp), t0);
}

#endif
