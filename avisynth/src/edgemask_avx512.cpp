/*
  edgemask_avx512.cpp

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
calc_direction(const __m512& gx0, const __m512& gx1, const __m512& gx2,
    const __m512& gx3, const __m512& gy0, const __m512& gy1, const __m512& gy2,
    const __m512& gy3, int32_t* dstp)
{
    constexpr float tan_1_8_pi = 0.414213538169860839843750f;
    constexpr float tan_3_8_pi = 2.414213657379150390625000f;
    constexpr float tan_5_8_pi = -2.414213657379150390625000f;
    constexpr float tan_7_8_pi = -0.414213538169860839843750f;

    const __m512 t18p = set1_ps<__m512>(tan_1_8_pi);
    const __m512 t38p = set1_ps<__m512>(tan_3_8_pi);
    const __m512 t58p = set1_ps<__m512>(tan_5_8_pi);
    const __m512 t78p = set1_ps<__m512>(tan_7_8_pi);

    const __m512i z = zero<__m512i>();
    const __m512i a000deg = _mm512_set1_epi32(15);
    const __m512i a045deg = _mm512_set1_epi32(31);
    const __m512i a090deg = _mm512_set1_epi32(63);
    const __m512i a135deg = _mm512_set1_epi32(127);

    __m512 tangent = fdiv(gy0, gx0);
    uint16_t t = _mm512_cmp_ps_mask(tangent, t78p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t18p, _CMP_LT_OQ);
    __m512i angle0 = _mm512_mask_blend_epi32(t, z, a000deg);
    t = _mm512_cmp_ps_mask(tangent, t18p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t38p, _CMP_LT_OQ);
    angle0 = _or(angle0, _mm512_mask_blend_epi32(t, z, a045deg));
    t = _mm512_cmp_ps_mask(tangent, t38p, _CMP_NLT_UQ) | _mm512_cmp_ps_mask(tangent, t58p, _CMP_NGE_UQ);
    angle0 = _or(angle0, _mm512_mask_blend_epi32(t, z, a090deg));
    t = _mm512_cmp_ps_mask(tangent, t58p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t78p, _CMP_LT_OQ);
    angle0 = _or(angle0, _mm512_mask_blend_epi32(t, z, a135deg));

    tangent = fdiv(gy1, gx1);
    t = _mm512_cmp_ps_mask(tangent, t78p, _CMP_NLT_UQ) & _mm512_cmp_ps_mask(tangent, t18p, _CMP_NGE_UQ);
    __m512i angle1 = _mm512_mask_blend_epi32(t, z, a000deg);
    t = _mm512_cmp_ps_mask(tangent, t18p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t38p, _CMP_LT_OQ);
    angle1 = _or(angle1, _mm512_mask_blend_epi32(t, z, a045deg));
    t = _mm512_cmp_ps_mask(tangent, t38p, _CMP_NLT_UQ) | _mm512_cmp_ps_mask(tangent, t58p, _CMP_NGE_UQ);
    angle1 = _or(angle1, _mm512_mask_blend_epi32(t, z, a090deg));
    t = _mm512_cmp_ps_mask(tangent, t58p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t78p, _CMP_LT_OQ);
    angle1 = _or(angle1, _mm512_mask_blend_epi32(t, z, a135deg));

    tangent = fdiv(gy2, gx2);
    t = _mm512_cmp_ps_mask(tangent, t78p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t18p, _CMP_LT_OQ);
    __m512i angle2 = _mm512_mask_blend_epi32(t, z, a000deg);
    t = _mm512_cmp_ps_mask(tangent, t18p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t38p, _CMP_LT_OQ);
    angle2 = _or(angle2, _mm512_mask_blend_epi32(t, z, a045deg));
    t = _mm512_cmp_ps_mask(tangent, t38p, _CMP_NLT_UQ) | _mm512_cmp_ps_mask(tangent, t58p, _CMP_NGE_UQ);
    angle2 = _or(angle2, _mm512_mask_blend_epi32(t, z, a090deg));
    t = _mm512_cmp_ps_mask(tangent, t58p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t78p, _CMP_LT_OQ);
    angle2 = _or(angle2, _mm512_mask_blend_epi32(t, z, a135deg));

    tangent = fdiv(gy3, gx3);
    t = _mm512_cmp_ps_mask(tangent, t78p, _CMP_GE_OQ) | _mm512_cmp_ps_mask(tangent, t18p, _CMP_LT_OQ);
    __m512i angle3 = _mm512_mask_blend_epi32(t, z, a000deg);
    t = _mm512_cmp_ps_mask(tangent, t18p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t38p, _CMP_LT_OQ);
    angle3 = _or(angle3, _mm512_mask_blend_epi32(t, z, a045deg));
    t = _mm512_cmp_ps_mask(tangent, t38p, _CMP_NLT_UQ) | _mm512_cmp_ps_mask(tangent, t58p, _CMP_NGE_UQ);
    angle3 = _or(angle3, _mm512_mask_blend_epi32(t, z, a090deg));
    t = _mm512_cmp_ps_mask(tangent, t58p, _CMP_GE_OQ) & _mm512_cmp_ps_mask(tangent, t78p, _CMP_LT_OQ);
    angle3 = _or(angle3, _mm512_mask_blend_epi32(t, z, a135deg));

    storeu<__m512i>(dstp, angle0);
    storeu<__m512i>(dstp + 16, angle1);
    storeu<__m512i>(dstp + 32, angle2);
    storeu<__m512i>(dstp + 48, angle3);
}


template <typename Td, bool SCALE, int OPERATOR, bool _STRICT, bool CALC_DIR>
SFINLINE void
emask(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    Td* d = reinterpret_cast<Td*>(dstp);
    int step = sizeof(__m512) / sizeof(float);

    const __m512 p0 = set1_ps<__m512>(opr[0]);
    const __m512 p1 = set1_ps<__m512>(opr[1]);
    const __m512 p2 = set1_ps<__m512>(opr[2]);
    const __m512 sc = set1_ps<__m512>(scale);
    const __m512 maxv = set1_ps<__m512>(maxval);

    memset(d, 0, width + sizeof(Td));
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

        for (int x = 1; x < width - 1; x += step * 4) {
            __m512 gx0, gx1, gx2, gx3, gy0, gy1, gy2, gy3;
            int L0 = x - 1, L1 = x - 1 + step, L2 = x - 1 + step * 2, L3 = x - 1 + step * 3,
                R0 = x + 1, R1 = x + 1 + step, R2 = x + 1 + step * 2, R3 = x + 1 + step * 3,
                C1 = x + step, C2 = x + step * 2, C3 = x + step * 3;
            if constexpr (OPERATOR == 0) { // standard
                gx0 = fsub(loadu<__m512>(centr + R0), loadu<__m512>(centr + L0));
                gx1 = fsub(loadu<__m512>(centr + R1), loadu<__m512>(centr + L1));
                gx2 = fsub(loadu<__m512>(centr + R2), loadu<__m512>(centr + L2));
                gx3 = fsub(loadu<__m512>(centr + R3), loadu<__m512>(centr + L3));
                gy0 = fsub(loadu<__m512>(above + x), loadu<__m512>(below + x));
                gy1 = fsub(loadu<__m512>(above + C1), loadu<__m512>(below + C1));
                gy2 = fsub(loadu<__m512>(above + C2), loadu<__m512>(below + C2));
                gy3 = fsub(loadu<__m512>(above + C3), loadu<__m512>(below + C3));

            } else if constexpr (OPERATOR == 1) { // Sobel
                gx0 = loadu<__m512>(above + R0);
                gx0 = fsub(gx0, loadu<__m512>(below + L0));
                gy0 = gx0;
                __m512 t0 = loadu<__m512>(above + L0);
                gx0 = fsub(gx0, t0);
                gy0 = fadd(gy0, t0);
                t0 = load<__m512>(below + R0);
                gx0 = fadd(gx0, t0);
                gy0 = fsub(gy0, t0);
                t0 = loadu<__m512>(centr + R0);
                gx0 = fadd(gx0, fadd(t0, t0));
                t0 = loadu<__m512>(centr + L0);
                gx0 = fsub(gx0, fadd(t0, t0));
                t0 = loadu<__m512>(above + x);
                gy0 = fadd(gy0, fadd(t0, t0));
                t0 = loadu<__m512>(below + x);
                gy0 = fsub(gy0, fadd(t0, t0));

                gx1 = loadu<__m512>(above + R1);
                gx1 = fsub(gx1, loadu<__m512>(below + L1));
                gy1 = gx1;
                t0 = loadu<__m512>(above + L1);
                gx1 = fsub(gx1, t0);
                gy1 = fadd(gy1, t0);
                t0 = load<__m512>(below + R1);
                gx1 = fadd(gx1, t0);
                gy1 = fsub(gy1, t0);
                t0 = loadu<__m512>(centr + R1);
                gx1 = fadd(gx1, fadd(t0, t0));
                t0 = loadu<__m512>(centr + L1);
                gx1 = fsub(gx1, fadd(t0, t0));
                t0 = loadu<__m512>(above + C1);
                gy1 = fadd(gy1, fadd(t0, t0));
                t0 = loadu<__m512>(below + C1);
                gy1 = fsub(gy1, fadd(t0, t0));

                gx2 = loadu<__m512>(above + R2);
                gx2 = fsub(gx2, loadu<__m512>(below + L2));
                gy2 = gx2;
                t0 = loadu<__m512>(above + L2);
                gx2 = fsub(gx2, t0);
                gy2 = fadd(gy2, t0);
                t0 = load<__m512>(below + R2);
                gx2 = fadd(gx2, t0);
                gy2 = fsub(gy2, t0);
                t0 = loadu<__m512>(centr + R2);
                gx2 = fadd(gx2, fadd(t0, t0));
                t0 = loadu<__m512>(centr + L2);
                gx2 = fsub(gx2, fadd(t0, t0));
                t0 = loadu<__m512>(above + C2);
                gy2 = fadd(gy2, fadd(t0, t0));
                t0 = loadu<__m512>(below + C2);
                gy2 = fsub(gy2, fadd(t0, t0));

                gx3 = loadu<__m512>(above + R3);
                gx3 = fsub(gx3, loadu<__m512>(below + L3));
                gy3 = gx3;
                t0 = loadu<__m512>(above + L3);
                gx3 = fsub(gx3, t0);
                gy3 = fadd(gy3, t0);
                t0 = load<__m512>(below + R3);
                gx3 = fadd(gx3, t0);
                gy3 = fsub(gy3, t0);
                t0 = loadu<__m512>(centr + R3);
                gx3 = fadd(gx3, fadd(t0, t0));
                t0 = loadu<__m512>(centr + L3);
                gx3 = fsub(gx3, fadd(t0, t0));
                t0 = loadu<__m512>(above + C3);
                gy3 = fadd(gy3, fadd(t0, t0));
                t0 = loadu<__m512>(below + C3);
                gy3 = fsub(gy3, fadd(t0, t0));

            } else {
                __m512 t0 = fmul(loadu<__m512>(below + R0), p2);
                __m512 t1 = fmul(loadu<__m512>(above + L0), p0);
                gx0 = fsub(t0, t1);
                gy0 = fsub(t1, t0);
                t0 = loadu<__m512>(above + R0);
                gx0 = fmadd(t0, p0, gx0);
                gy0 = fmadd(t0, p2, gy0);
                t0 = loadu<__m512>(below + L0);
                gx0 = fnmadd(t0, p2, gx0);
                gy0 = fnmadd(t0, p0, gy0);
                gx0 = fmadd(loadu<__m512>(centr + R0), p1, gx0);
                gx0 = fnmadd(loadu<__m512>(centr + L0), p1, gx0);
                gy0 = fmadd(loadu<__m512>(above + x), p1, gy0);
                gy0 = fnmadd(loadu<__m512>(below + x), p1, gy0);

                t0 = fmul(loadu<__m512>(below + R1), p2);
                t1 = fmul(loadu<__m512>(above + L1), p0);
                gx1 = fsub(t0, t1);
                gy1 = fsub(t1, t0);
                t0 = loadu<__m512>(above + R1);
                gx1 = fmadd(t0, p0, gx1);
                gy1 = fmadd(t0, p2, gy1);
                t0 = loadu<__m512>(below + L1);
                gx1 = fnmadd(t0, p2, gx1);
                gy1 = fnmadd(t0, p0, gy1);
                gx1 = fmadd(loadu<__m512>(centr + R1), p1, gx1);
                gx1 = fnmadd(loadu<__m512>(centr + L1), p1, gx1);
                gy1 = fmadd(loadu<__m512>(above + C1), p1, gy1);
                gy1 = fnmadd(loadu<__m512>(below + C1), p1, gy1);

                t0 = fmul(loadu<__m512>(below + R2), p2);
                t1 = fmul(loadu<__m512>(above + L2), p0);
                gx2 = fsub(t0, t1);
                gy2 = fsub(t1, t0);
                t0 = loadu<__m512>(above + R2);
                gx2 = fmadd(t0, p0, gx2);
                gy2 = fmadd(t0, p2, gy2);
                t0 = loadu<__m512>(below + L2);
                gx2 = fnmadd(t0, p2, gx2);
                gy2 = fnmadd(t0, p0, gy2);
                gx2 = fmadd(loadu<__m512>(centr + R2), p1, gx2);
                gx2 = fnmadd(loadu<__m512>(centr + L2), p1, gx2);
                gy2 = fmadd(loadu<__m512>(above + C2), p1, gy2);
                gy2 = fnmadd(loadu<__m512>(below + C2), p1, gy2);

                t0 = fmul(loadu<__m512>(below + R3), p2);
                t1 = fmul(loadu<__m512>(above + L3), p0);
                gx3 = fsub(t0, t1);
                gy3 = fsub(t1, t0);
                t0 = loadu<__m512>(above + R3);
                gx3 = fmadd(t0, p0, gx3);
                gy3 = fmadd(t0, p2, gy3);
                t0 = loadu<__m512>(below + L3);
                gx3 = fnmadd(t0, p2, gx3);
                gy3 = fnmadd(t0, p0, gy3);
                gx3 = fmadd(loadu<__m512>(centr + R3), p1, gx3);
                gx3 = fnmadd(loadu<__m512>(centr + L3), p1, gx3);
                gy3 = fmadd(loadu<__m512>(above + C3), p1, gy3);
                gy3 = fnmadd(loadu<__m512>(below + C3), p1, gy3);
            }
            if constexpr (CALC_DIR) {
                calc_direction(gx0, gx1, gx2, gx3, gy0, gy1, gy2, gy3, dirp + x);
            }
            __m512 mag0, mag1, mag2, mag3;
            if constexpr (_STRICT) {
                mag0 = fsqrt(fmadd(gy0, gy0, fmul(gx0, gx0)));
                mag1 = fsqrt(fmadd(gy1, gy1, fmul(gx1, gx1)));
                mag2 = fsqrt(fmadd(gy2, gy2, fmul(gx2, gx2)));
                mag3 = fsqrt(fmadd(gy3, gy3, fmul(gx3, gx3)));
            } else {
                mag0 = fadd(fabs(gx0), fabs(gy0));
                mag1 = fadd(fabs(gx1), fabs(gy1));
                mag2 = fadd(fabs(gx2), fabs(gy2));
                mag3 = fadd(fabs(gx3), fabs(gy3));
            }
            if constexpr (SCALE != 0) {
                mag0 = fmul(mag0, sc);
                mag1 = fmul(mag1, sc);
                mag2 = fmul(mag2, sc);
                mag3 = fmul(mag3, sc);
            }
            mag0 = fmin<__m512>(mag0, maxv);
            mag1 = fmin<__m512>(mag1, maxv);
            mag2 = fmin<__m512>(mag2, maxv);
            mag3 = fmin<__m512>(mag3, maxv);
            if constexpr (is_same_v<Td, float>) {
                storeu<__m512>(d + x, mag0);
                storeu<__m512>(d + C1, mag1);
                storeu<__m512>(d + C2, mag2);
                storeu<__m512>(d + C3, mag3);
            }
            else if constexpr (is_same_v<Td, uint16_t>) {
                __m512i m0 = cvtps_epu16<__m512i, __m512>(mag0, mag1);
                __m512i m1 = cvtps_epu16<__m512i, __m512>(mag2, mag3);
                storeu<__m512i>(d + x, m0);
                storeu<__m512i>(d + C2, m1);
            }
            else if constexpr (is_same_v<Td, uint8_t>) {
                __m512i m0 = cvtps_epu8_2(mag0, mag1, mag2, mag3);
                storeu<__m512i>(d + x, m0);
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


void nms_avx512(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height)
{
    int step = sizeof(__m512) / sizeof(float);

    const __m512i a000deg = _mm512_set1_epi32(15);
    const __m512i a045deg = _mm512_set1_epi32(31);
    const __m512i a090deg = _mm512_set1_epi32(63);
    const __m512i a135deg = _mm512_set1_epi32(127);
    const __m512 zero = _mm512_setzero_ps();

    memset(dstp, 0, dpitch * sizeof(float));
    for (int y = 1; y < height - 1; ++y) {
        emaskp += epitch;
        dirp += dirpitch;
        dstp += dpitch;
        dstp[0] = 0;
        for (int x = 1; x < width - 1; x += step) {
            __m512i dir = loadu<__m512i>(dirp + x);

            auto mask = cmpeq_epi32<uint16_t, __m512i>(dir, a135deg);
            __m512 t = fmax(loadu<__m512>(emaskp + x - 1 - epitch),
                loadu<__m512>(emaskp + x + 1 + epitch));
            __m512 p0 = _mm512_mask_blend_ps(mask, zero, t);

            mask = cmpeq_epi32<uint16_t, __m512i>(dir, a090deg);
            t = fmax(loadu<__m512>(emaskp + x - epitch),
                loadu<__m512>(emaskp + x + epitch));
            p0 = _or(p0, _mm512_mask_blend_ps(mask, zero, t));

            mask = cmpeq_epi32<uint16_t, __m512i>(dir, a045deg);
            t = fmax(loadu<__m512>(emaskp + x - epitch + 1),
                loadu<__m512>(emaskp + x + epitch - 1));
            p0 = _or(p0, _mm512_mask_blend_ps(mask, zero, t));

            mask = cmpeq_epi32<uint16_t, __m512i>(dir, a000deg);
            t = fmax(loadu<__m512>(emaskp + x + 1),
                loadu<__m512>(emaskp + x - 1));
            p0 = _or(p0, _mm512_mask_blend_ps(mask, zero, t));

            __m512 edge = loadu<__m512>(emaskp + x);
            mask = _mm512_cmp_ps_mask(edge, p0, _CMP_LT_OQ);
            edge = _mm512_mask_blend_ps(mask, edge, zero);
            storeu<__m512>(dstp + x, edge);
        }
        dstp[width - 1] = 0;
    }
    memset(dstp + dpitch, 0, dpitch * sizeof(float));
}


void emask_avx512_u8_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u8_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint8_t, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_u16_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<uint16_t, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, false, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, true, false>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, false, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 0, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 0, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 1, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 1, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_ns_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, false, 2, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}

void emask_avx512_flt_sc_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    emask<float, true, 2, true, true>(blurp, blpitch, dstp, dpitch, opr,
        scale, width, height, maxval, dirp, dirpitch);
}
