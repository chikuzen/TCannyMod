/*
  gaussian_blur_sse4.cpp

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


#include <vector>
#include <algorithm>
#include "simd.hpp"
#include "gaussian_blur.hpp"


template <typename Ts>
SFINLINE void
convert_to_float(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    constexpr size_t step = sizeof(__m128) / sizeof(Ts);
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    float* d = reinterpret_cast<float*>(dstp);

    if constexpr (is_same_v<Ts, float>) {
        for (int y = 0; y < height; ++y) {
            memcpy(d, s, sizeof(float) * width);
            s += spitch;
            d += dpitch;
        }
    } else {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; x += step) {
                __m128 v = cvtepuX_ps<__m128, Ts>(s + x);
                store<__m128>(d + x, v);
            }
            s += spitch;
            d += dpitch;
        }
    }
}


template <typename Td>
SFINLINE void
hblur(float* srcp, const int spitch, Td* dstp, const int dpitch,
    const int width, const int radius, const float* weights, const int remains)
{
    constexpr size_t step = sizeof(__m128) / sizeof(float);

    auto s0 = srcp + spitch * 0; auto d0 = dstp + dpitch * 0;
    auto s1 = srcp + spitch * 1; auto d1 = dstp + dpitch * 1;
    auto s2 = srcp + spitch * 2; auto d2 = dstp + dpitch * 2;
    auto s3 = srcp + spitch * 3; auto d3 = dstp + dpitch * 3;

    for (int r = 1; r <= radius; ++r) {
        s0[-r] = s0[r];
        s0[width + r - 1] = s0[width - r - 1];
        s1[-r] = s1[r];
        s1[width + r - 1] = s1[width - r - 1];
        s2[-r] = s2[r];
        s2[width + r - 1] = s2[width - r - 1];
        s3[-r] = s3[r];
        s3[width + r - 1] = s3[width - r - 1];
    }

    weights += radius;
    for (int x = 0; x < width; x += step * 2) {

        __m128 sum00 = zero<__m128>(); __m128 sum01 = zero<__m128>();
        __m128 sum10 = zero<__m128>(); __m128 sum11 = zero<__m128>();
        __m128 sum20 = zero<__m128>(); __m128 sum21 = zero<__m128>();
        __m128 sum30 = zero<__m128>(); __m128 sum31 = zero<__m128>();

        for (int v = -radius; v <= radius; ++v) {
            __m128 k = set1_ps<__m128>(weights[v]);
            sum00 = fmadd<__m128>(k, loadu<__m128>(s0 + x + v), sum00);
            sum01 = fmadd<__m128>(k, loadu<__m128>(s0 + x + v + step), sum01);
            sum10 = fmadd<__m128>(k, loadu<__m128>(s1 + x + v), sum10);
            sum11 = fmadd<__m128>(k, loadu<__m128>(s1 + x + v + step), sum11);
            sum20 = fmadd<__m128>(k, loadu<__m128>(s2 + x + v), sum20);
            sum21 = fmadd<__m128>(k, loadu<__m128>(s2 + x + v + step), sum21);
            sum30 = fmadd<__m128>(k, loadu<__m128>(s3 + x + v), sum30);
            sum31 = fmadd<__m128>(k, loadu<__m128>(s3 + x + v + step), sum31);
        }
        if constexpr (is_same_v<Td, float>) {
            store<__m128>(d0 + x, sum00);
            store<__m128>(d0 + x + step, sum01);
            if (remains < 2) continue;
            store<__m128>(d1 + x, sum10);
            store<__m128>(d1 + x + step, sum11);
            if (remains < 3) continue;
            store<__m128>(d2 + x, sum20);
            store<__m128>(d2 + x + step, sum21);
            if (remains < 4) continue;
            store<__m128>(d3 + x, sum30);
            store<__m128>(d3 + x + step, sum31);
        }
        else if constexpr (is_same_v<Td, uint16_t>) {
            __m128i data = cvtps_epu16<__m128i, __m128>(sum00, sum01);
            store<__m128i>(d0 + x, data);
            if (remains < 2) continue;
            data = cvtps_epu16<__m128i, __m128>(sum10, sum11);
            store<__m128i>(d1 + x, data);
            if (remains < 2) continue;
            data = cvtps_epu16<__m128i, __m128>(sum20, sum21);
            store<__m128i>(d2 + x, data);
            if (remains < 3) continue;
            data = cvtps_epu16<__m128i, __m128>(sum30, sum31);
            store<__m128i>(d3 + x, data);
        }
        else if constexpr (is_same_v<Td, uint8_t>) {
            __m128i data = cvtps_epu8<__m128i, __m128>(sum00, sum01);
            storel(d0 + x, data);
            if (remains < 2) continue;
            data = cvtps_epu8<__m128i, __m128>(sum10, sum11);
            storel(d1 + x, data);
            if (remains < 2) continue;
            data = cvtps_epu8<__m128i, __m128>(sum20, sum21);
            storel(d2 + x, data);
            if (remains < 3) continue;
            data = cvtps_epu8<__m128i, __m128>(sum30, sum31);
            storel(d3 + x, data);
        }
    }
}


template <typename Ts, int RADIUS, typename Td>
SFINLINE void
gblur(const void* srcp, int spitch, float* hbuffp, int hbpitch, void* dstp,
    int dpitch, int width, int height, int radius, const float* weights,
    const float)
{
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    Td* d = reinterpret_cast<Td*>(dstp);

    constexpr size_t step = sizeof(__m128) / sizeof(float);
    const int length = radius * 2 + 1;
    std::vector<const Ts*> ptr(length + 3, nullptr);
    ptr[radius] = s;
    for (int r = 1; r <= radius; ++r) {
        ptr[radius + r] = s + r * spitch;
        ptr[radius - r] = ptr[radius + r];
    }
    ptr[length + 0] = s + spitch * (radius + 1);
    ptr[length + 1] = s + spitch * (radius + 2);
    ptr[length + 2] = s + spitch * (radius + 3);

    float* hb[4] = {
        hbuffp,
        hbuffp + 1 * hbpitch,
        hbuffp + 2 * hbpitch,
        hbuffp + 3 * hbpitch
    };

    for (int y = 0; y < height; y += 4) {
        for (int x = 0; x < width; x += step) {
            __m128 k0 = set1_ps<__m128>(weights[0]);
            __m128 val = cvtepuX_ps<__m128, Ts>(ptr[0] + x);
            __m128 sum0 = fmul<__m128>(k0, val);

            __m128 k1 = set1_ps<__m128>(weights[1]);
            val = cvtepuX_ps<__m128, Ts>(ptr[1] + x);
            sum0 = fmadd<__m128>(k1, val, sum0);
            __m128 sum1 = fmul<__m128>(k0, val);

            __m128 k2 = set1_ps<__m128>(weights[2]);
            val = cvtepuX_ps<__m128, Ts>(ptr[2] + x);
            sum0 = fmadd<__m128>(k2, val, sum0);
            sum1 = fmadd<__m128>(k1, val, sum1);
            __m128 sum2 = fmul<__m128>(k0, val);

            if constexpr (RADIUS == 1) {
                store<__m128>(hb[0] + x, sum0);

                val = cvtepuX_ps<__m128, Ts>(ptr[3] + x);
                sum1 = fmadd<__m128>(k2, val, sum1);
                sum2 = fmadd<__m128>(k1, val, sum2);
                __m128 sum3 = fmul<__m128>(k0, val);
                store<__m128>(hb[1] + x, sum1);

                val = cvtepuX_ps<__m128, Ts>(ptr[4] + x);
                sum2 = fmadd<__m128>(k2, val, sum2);
                sum3 = fmadd<__m128>(k1, val, sum3);
                store<__m128>(hb[2] + x, sum2);

                val = cvtepuX_ps<__m128, Ts>(ptr[5] + x);
                sum3 = fmadd<__m128>(k2, val, sum3);
                store<__m128>(hb[3] + x, sum3);

            } else {
                __m128 k3 = set1_ps<__m128>(weights[3]);
                val = cvtepuX_ps<__m128, Ts>(ptr[3] + x);
                sum0 = fmadd<__m128>(k3, val, sum0);
                sum1 = fmadd<__m128>(k2, val, sum1);
                sum2 = fmadd<__m128>(k1, val, sum2);
                __m128 sum3 = fmul<__m128>(k0, val);

                for (int v = 4; v < length; ++v) {
                    k0 = k1;
                    k1 = k2;
                    k2 = k3;
                    k3 = set1_ps<__m128>(weights[v]);
                    val = cvtepuX_ps<__m128, Ts>(ptr[v] + x);
                    sum0 = fmadd<__m128>(k3, val, sum0);
                    sum1 = fmadd<__m128>(k2, val, sum1);
                    sum2 = fmadd<__m128>(k1, val, sum2);
                    sum3 = fmadd<__m128>(k0, val, sum3);
                }
                store<__m128>(hb[0] + x, sum0);

                val = cvtepuX_ps<__m128, Ts>(ptr[length] + x);
                sum1 = fmadd<__m128>(k3, val, sum1);
                sum2 = fmadd<__m128>(k2, val, sum2);
                sum3 = fmadd<__m128>(k1, val, sum3);
                store<__m128>(hb[1] + x, sum1);

                val = cvtepuX_ps<__m128, Ts>(ptr[length + 1] + x);
                sum2 = fmadd<__m128>(k3, val, sum2);
                sum3 = fmadd<__m128>(k2, val, sum3);
                store<__m128>(hb[2] + x, sum2);

                val = cvtepuX_ps<__m128, Ts>(ptr[length + 2] + x);
                sum3 = fmadd<__m128>(k3, val, sum3);
                store<__m128>(hb[3] + x, sum3);
            }
        }
        int remains = std::min(height - y, 4);
        hblur<Td>(hbuffp, hbpitch, d, dpitch, width, radius, weights, remains);
        d += dpitch * 4;

        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < length + 2; ++l) {
                ptr[l] = ptr[l + 1];
            }
            if (y < height - 4 - radius - i) {
                ptr[length + 2] += spitch;
            } else if (y >= height - 4 - radius - i) {
                ptr[length + 2] -= spitch;
            }
        }
    }
}


void cvt2flt_sse4_u8(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<uint8_t>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void cvt2flt_sse4_u16(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<uint16_t>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void cvt2flt_sse4_flt(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<float>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void gblur_sse4_u8_r1_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 1, uint8_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u8_r2_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 2, uint8_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u8_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u8_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u16_r1_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 1, uint16_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u16_r2_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 2, uint16_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u16_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_u16_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_flt_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<float, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_sse4_flt_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<float, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}
