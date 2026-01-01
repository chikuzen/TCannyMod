/*
  edgemask_avx2.hpp

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


#include <array>
#include "edgemask.hpp"
#include "simd.hpp"


SFINLINE void
calc_direction(const __m256& gx0, const __m256& gx1, const __m256& gy0,
    const __m256& gy1, int32_t* dstp)
{
    constexpr float tan_1_8_pi = 0.414213538169860839843750f;
    constexpr float tan_3_8_pi = 2.414213657379150390625000f;
    constexpr float tan_5_8_pi = -2.414213657379150390625000f;
    constexpr float tan_7_8_pi = -0.414213538169860839843750f;

    const __m256 t18p = set1_ps<__m256>(tan_1_8_pi);
    const __m256 t38p = set1_ps<__m256>(tan_3_8_pi);
    const __m256 t58p = set1_ps<__m256>(tan_5_8_pi);
    const __m256 t78p = set1_ps<__m256>(tan_7_8_pi);

    __m256 tangent = fdiv(gy0, gx0);

    __m256 t = _and(_mm256_cmp_ps(tangent, t78p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t18p, _CMP_LT_OQ));
    __m256i angle0 = srli_epi32(castps_si<__m256i, __m256>(t), 28);
    t = _and(_mm256_cmp_ps(tangent, t18p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t38p, _CMP_LT_OQ));
    angle0 = _or(angle0, srli_epi32(castps_si<__m256i, __m256>(t), 27));
    t = _or(_mm256_cmp_ps(tangent, t38p, _CMP_NLT_UQ), _mm256_cmp_ps(tangent, t58p, _CMP_NGE_UQ));
    angle0 = _or(angle0, srli_epi32(castps_si<__m256i, __m256>(t), 26));
    t = _and(_mm256_cmp_ps(tangent, t58p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t78p, _CMP_LT_OQ));
    angle0 = _or(angle0, srli_epi32(castps_si<__m256i, __m256>(t), 25));

    tangent = fdiv(gy1, gx1);
    t = _and(_mm256_cmp_ps(tangent, t78p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t18p, _CMP_LT_OQ));
    __m256i angle1 = srli_epi32(castps_si<__m256i, __m256>(t), 28);
    t = _and(_mm256_cmp_ps(tangent, t18p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t38p, _CMP_LT_OQ));
    angle1 = _or(angle1, srli_epi32(castps_si<__m256i, __m256>(t), 27));
    t = _or(_mm256_cmp_ps(tangent, t38p, _CMP_NLT_UQ), _mm256_cmp_ps(tangent, t58p, _CMP_NGE_UQ));
    angle1 = _or(angle1, srli_epi32(castps_si<__m256i, __m256>(t), 26));
    t = _and(_mm256_cmp_ps(tangent, t58p, _CMP_GE_OQ), _mm256_cmp_ps(tangent, t78p, _CMP_LT_OQ));
    angle1 = _or(angle1, srli_epi32(castps_si<__m256i, __m256>(t), 25));

    storeu<__m256i>(dstp, angle0);
    storeu<__m256i>(dstp + 8, angle1);
}


template <typename Td, bool SCALE, int OPERATOR, bool _STRICT, bool CALC_DIR>
SFINLINE void
emask(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    Td* d = reinterpret_cast<Td*>(dstp);
    int step = sizeof(__m256) / sizeof(float);

    const __m256 p0 = set1_ps<__m256>(opr[0]);
    const __m256 p1 = set1_ps<__m256>(opr[1]);
    const __m256 p2 = set1_ps<__m256>(opr[2]);
    const __m256 sc = set1_ps<__m256>(scale);
    const __m256 maxv = set1_ps<__m256>(maxval);

    memset(d, 0, width * sizeof(Td));
    d += dpitch;

    if constexpr (CALC_DIR) {
        memset(dirp, 0, dirpitch * sizeof(int32_t));
        dirp += dirpitch;
    }

    for (int y = 1; y < height - 1; ++y) {
        const float* above = blurp;             // above      a[x-1] a[x] a[x+1]
        const float* centr = blurp + blpitch;   // center     c[x-1] c[x] c[x+1]
        const float* below = centr + blpitch;   // bellow     b[x-1] b[x] b[x+1]
        d[0] = 0;
        if constexpr (CALC_DIR) {
            dirp[0] = 0;
        }

        for (int x = 1; x < width - 1; x += step * 2) {
            __m256 gx0, gx1, gy0, gy1;
            int L = x - 1, R = x + 1;
            if constexpr (OPERATOR == 0) { // standard
                gx0 = fsub(loadu<__m256>(centr + R), loadu<__m256>(centr + L));
                gx1 = fsub(loadu<__m256>(centr + R + step),
                    loadu<__m256>(centr + L + step));
                gy0 = fsub(loadu<__m256>(above + x), loadu<__m256>(below + x));
                gy1 = fsub(loadu<__m256>(above + x + step),
                    loadu<__m256>(below + x + step));
            }
            else if constexpr (OPERATOR == 1) { // Sobel
                gx0 = loadu<__m256>(above + R);
                gx0 = fsub(gx0, loadu<__m256>(below + L));
                gy0 = gx0;
                __m256 t0 = loadu<__m256>(above + L);
                gx0 = fsub(gx0, t0);
                gy0 = fadd(gy0, t0);
                t0 = load<__m256>(below + R);
                gx0 = fadd(gx0, t0);
                gy0 = fsub(gy0, t0);
                t0 = loadu<__m256>(centr + R);
                gx0 = fadd(gx0, fadd(t0, t0));
                t0 = loadu<__m256>(centr + L);
                gx0 = fsub(gx0, fadd(t0, t0));
                t0 = loadu<__m256>(above + x);
                gy0 = fadd(gy0, fadd(t0, t0));
                t0 = loadu<__m256>(below + x);
                gy0 = fsub(gy0, fadd(t0, t0));

                gx1 = loadu<__m256>(above + R + step);
                gx1 = fsub(gx1, loadu<__m256>(below + L + step));
                gy1 = gx1;
                t0 = loadu<__m256>(above + L + step);
                gx1 = fsub(gx1, t0);
                gy1 = fadd(gy1, t0);
                t0 = load<__m256>(below + R + step);
                gx1 = fadd(gx1, t0);
                gy1 = fsub(gy1, t0);
                t0 = loadu<__m256>(centr + R + step);
                gx1 = fadd(gx1, fadd(t0, t0));
                t0 = loadu<__m256>(centr + L + step);
                gx1 = fsub(gx1, fadd(t0, t0));
                t0 = loadu<__m256>(above + x + step);
                gy1 = fadd(gy1, fadd(t0, t0));
                t0 = loadu<__m256>(below + x + step);
                gy1 = fsub(gy1, fadd(t0, t0));
            } else {
                __m256 t0 = fmul(loadu<__m256>(below + R), p2);
                __m256 t1 = fmul(loadu<__m256>(above + L), p0);
                gx0 = fsub(t0, t1);
                gy0 = fsub(t1, t0);
                t0 = loadu<__m256>(above + R);
                gx0 = fmadd(t0, p0, gx0);
                gy0 = fmadd(t0, p2, gy0);
                t0 = loadu<__m256>(below + L);
                gx0 = fnmadd(t0, p2, gx0);
                gy0 = fnmadd(t0, p0, gy0);
                gx0 = fmadd(loadu<__m256>(centr + R), p1, gx0);
                gx0 = fnmadd(loadu<__m256>(centr + L), p1, gx0);
                gy0 = fmadd(loadu<__m256>(above + x), p1, gy0);
                gy0 = fnmadd(loadu<__m256>(below + x), p1, gy0);

                t0 = fmul(loadu<__m256>(below + R + step), p2);
                t1 = fmul(loadu<__m256>(above + L + step), p0);
                gx1 = fsub(t0, t1);
                gy1 = fsub(t1, t0);
                t0 = loadu<__m256>(above + R + step);
                gx1 = fmadd(t0, p0, gx1);
                gy1 = fmadd(t0, p2, gy1);
                t0 = loadu<__m256>(below + L + step);
                gx1 = fnmadd(t0, p2, gx1);
                gy1 = fnmadd(t0, p0, gy1);
                gx1 = fmadd(loadu<__m256>(centr + R + step), p1, gx1);
                gx1 = fnmadd(loadu<__m256>(centr + L + step), p1, gx1);
                gy1 = fmadd(loadu<__m256>(above + x + step), p1, gy1);
                gy1 = fnmadd(loadu<__m256>(below + x + step), p1, gy1);
            }
            if constexpr (CALC_DIR) {
                calc_direction(gx0, gx1, gy0, gy1, dirp + x);
            }
            __m256 mag0, mag1;
            if constexpr (_STRICT) {
                mag0 = fsqrt(fmadd(gy0, gy0, fmul(gx0, gx0)));
                mag1 = fsqrt(fmadd(gy1, gy1, fmul(gx1, gx1)));
            }
            else {
                mag0 = fadd(fabs(gx0), fabs<__m256>(gy0));
                mag1 = fadd(fabs<__m256>(gx1), fabs<__m256>(gy1));
            }
            if constexpr (SCALE != 0) {
                mag0 = fmul(mag0, sc);
                mag1 = fmul(mag1, sc);
            }
            mag0 = fmin<__m256>(mag0, maxv);
            mag1 = fmin<__m256>(mag1, maxv);

            if constexpr (is_same_v<Td, float>) {
                storeu<__m256>(d + x, mag0);
                storeu<__m256>(d + x + step, mag1);
            }
            else if constexpr (is_same_v<Td, uint16_t>) {
                __m256i m0 = cvtps_epu16<__m256i, __m256>(mag0, mag1);
                storeu<__m256i>(d + x, m0);
            }
            else if constexpr (is_same_v<Td, uint8_t>) {
                __m128i m0 = cvtps_epu8<__m128i, __m256>(mag0, mag1);
                storeu<__m128i>(d + x, m0);
            }
        }
        d[width - 1] = 0;
        blurp += blpitch;
        d += dpitch;
        if constexpr (CALC_DIR) {
            dirp[width - 1] = 0;
            dirp += dirpitch;
        }
    }
    memset(d, 0, width * sizeof(Td));
}


void nms_avx2(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height)
{
    int step = sizeof(__m256) / sizeof(float);

    memset(dstp, 0, dpitch * sizeof(float));

    for (int y = 1; y < height - 1; ++y) {
        emaskp += epitch;
        dirp += dirpitch;
        dstp += dpitch;
        dstp[0] = 0;
        for (int x = 1; x < width - 1; x += step) {
            __m256i dir = loadu<__m256i>(dirp + x);
            __m256i angle = cmpeq_epi32<__m256i, __m256i>(dir, dir);

            angle = srli_epi32(angle, 25); // 135 deg
            __m256 mask = castsi_ps<__m256, __m256i>(
                cmpeq_epi32<__m256i, __m256i>(dir, angle));
            __m256 t = fmax(loadu<__m256>(emaskp + x - 1 - epitch),
                loadu<__m256>(emaskp + x + 1 + epitch));
            __m256 p0 = _and(t, mask);

            angle = srli_epi32(angle, 1); // 90 deg
            mask = castsi_ps<__m256, __m256i>(
                cmpeq_epi32<__m256i, __m256i>(dir, angle));
            t = fmax(loadu<__m256>(emaskp + x - epitch),
                loadu<__m256>(emaskp + x + epitch));
            p0 = _or(p0, _and(t, mask));

            angle = srli_epi32(angle, 1); // 45 deg
            mask = castsi_ps<__m256, __m256i>(
                cmpeq_epi32<__m256i, __m256i>(dir, angle));
            t = fmax(loadu<__m256>(emaskp + x - epitch + 1),
                loadu<__m256>(emaskp + x + epitch - 1));
            p0 = _or(p0, _and(t, mask));

            angle = srli_epi32(angle, 1); // 0 deg
            mask = castsi_ps<__m256, __m256i>(
                cmpeq_epi32<__m256i, __m256i>(dir, angle));
            t = fmax(loadu<__m256>(emaskp + x + 1),
                loadu<__m256>(emaskp + x - 1));
            p0 = _or(p0, _and(t, mask));

            __m256 edge = loadu<__m256>(emaskp + x);
            mask = cmplt_ps<__m256, __m256>(edge, p0);
            edge = blendv(edge, zero<__m256>(), mask);
            storeu<__m256>(dstp + x, edge);
        }
        dstp[width - 1] = 0;
    }
    memset(dstp + dpitch, 0, dpitch * sizeof(float));
}


void emask_avx2_u8_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u8_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_u16_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_ns_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx2_flt_sc_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}
