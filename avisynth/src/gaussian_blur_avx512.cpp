/*
  gaussian_blur_avx512.cpp

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
    constexpr size_t step = sizeof(__m512) / sizeof(float);
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    float* d = reinterpret_cast<float*>(dstp);

    if constexpr (is_same_v<Ts, float>) {
        for (int y = 0; y < height; ++y) {
            memcpy(dstp, srcp, sizeof(Ts) * width);
            s += spitch;
            d += dpitch;
        }
    } else {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; x += step) {
                __m512 v = cvtepuX_ps<__m512, Ts>(s + x);
                store<__m512>(d + x, v);
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
    constexpr size_t step1 = sizeof(__m512) / sizeof(float);
    constexpr size_t step2 = step1 * 2;
    constexpr size_t step3 = step1 * 3;
    constexpr size_t step4 = step1 * 4;

    auto s0 = srcp + spitch * 0; auto d0 = dstp + dpitch * 0;
    auto s1 = srcp + spitch * 1; auto d1 = dstp + dpitch * 1;
    auto s2 = srcp + spitch * 2; auto d2 = dstp + dpitch * 2;
    auto s3 = srcp + spitch * 3; auto d3 = dstp + dpitch * 3;
    auto s4 = srcp + spitch * 4; auto d4 = dstp + dpitch * 4;
    auto s5 = srcp + spitch * 5; auto d5 = dstp + dpitch * 5;

    for (int r = 1; r <= radius; ++r) {
        s0[-r] = s0[r];
        s0[width + r - 1] = s0[width - r - 1];
        s1[-r] = s1[r];
        s1[width + r - 1] = s1[width - r - 1];
        s2[-r] = s2[r];
        s2[width + r - 1] = s2[width - r - 1];
        s3[-r] = s3[r];
        s3[width + r - 1] = s3[width - r - 1];
        s4[-r] = s4[r];
        s4[width + r - 1] = s4[width - r - 1];
        s5[-r] = s5[r];
        s5[width + r - 1] = s5[width - r - 1];
    }

    weights += radius;
    for (int x = 0; x < width; x += step4) {

        __m512 sum00 = zero<__m512>(); __m512 sum01 = zero<__m512>();
        __m512 sum02 = zero<__m512>(); __m512 sum03 = zero<__m512>();

        __m512 sum10 = zero<__m512>(); __m512 sum11 = zero<__m512>();
        __m512 sum12 = zero<__m512>(); __m512 sum13 = zero<__m512>();

        __m512 sum20 = zero<__m512>(); __m512 sum21 = zero<__m512>();
        __m512 sum22 = zero<__m512>(); __m512 sum23 = zero<__m512>();

        __m512 sum30 = zero<__m512>(); __m512 sum31 = zero<__m512>();
        __m512 sum32 = zero<__m512>(); __m512 sum33 = zero<__m512>();

        __m512 sum40 = zero<__m512>(); __m512 sum41 = zero<__m512>();
        __m512 sum42 = zero<__m512>(); __m512 sum43 = zero<__m512>();

        __m512 sum50 = zero<__m512>(); __m512 sum51 = zero<__m512>();
        __m512 sum52 = zero<__m512>(); __m512 sum53 = zero<__m512>();

        for (int v = -radius; v <= radius; ++v) {
            __m512 k = set1_ps<__m512>(weights[v]);
            sum00 = fmadd<__m512>(k, loadu<__m512>(s0 + x + v +     0), sum00);
            sum01 = fmadd<__m512>(k, loadu<__m512>(s0 + x + v + step1), sum01);
            sum02 = fmadd<__m512>(k, loadu<__m512>(s0 + x + v + step2), sum02);
            sum03 = fmadd<__m512>(k, loadu<__m512>(s0 + x + v + step3), sum03);

            sum10 = fmadd<__m512>(k, loadu<__m512>(s1 + x + v +     0), sum10);
            sum11 = fmadd<__m512>(k, loadu<__m512>(s1 + x + v + step1), sum11);
            sum12 = fmadd<__m512>(k, loadu<__m512>(s1 + x + v + step2), sum12);
            sum13 = fmadd<__m512>(k, loadu<__m512>(s1 + x + v + step3), sum13);

            sum20 = fmadd<__m512>(k, loadu<__m512>(s2 + x + v +     0), sum20);
            sum21 = fmadd<__m512>(k, loadu<__m512>(s2 + x + v + step1), sum21);
            sum22 = fmadd<__m512>(k, loadu<__m512>(s2 + x + v + step2), sum22);
            sum23 = fmadd<__m512>(k, loadu<__m512>(s2 + x + v + step3), sum23);

            sum30 = fmadd<__m512>(k, loadu<__m512>(s3 + x + v +     0), sum30);
            sum31 = fmadd<__m512>(k, loadu<__m512>(s3 + x + v + step1), sum31);
            sum32 = fmadd<__m512>(k, loadu<__m512>(s3 + x + v + step2), sum32);
            sum33 = fmadd<__m512>(k, loadu<__m512>(s3 + x + v + step3), sum33);

            sum40 = fmadd<__m512>(k, loadu<__m512>(s4 + x + v +     0), sum40);
            sum41 = fmadd<__m512>(k, loadu<__m512>(s4 + x + v + step1), sum41);
            sum42 = fmadd<__m512>(k, loadu<__m512>(s4 + x + v + step2), sum42);
            sum43 = fmadd<__m512>(k, loadu<__m512>(s4 + x + v + step3), sum43);

            sum50 = fmadd<__m512>(k, loadu<__m512>(s5 + x + v +     0), sum50);
            sum51 = fmadd<__m512>(k, loadu<__m512>(s5 + x + v + step1), sum51);
            sum52 = fmadd<__m512>(k, loadu<__m512>(s5 + x + v + step2), sum52);
            sum53 = fmadd<__m512>(k, loadu<__m512>(s5 + x + v + step3), sum53);
        }
        if constexpr (is_same_v<Td, float>) {
            store<__m512>(d0 + x +     0, sum00);
            store<__m512>(d0 + x + step1, sum01);
            store<__m512>(d0 + x + step2, sum02);
            store<__m512>(d0 + x + step3, sum03);

            if (remains < 2) continue;
            store<__m512>(d1 + x +     0, sum10);
            store<__m512>(d1 + x + step1, sum11);
            store<__m512>(d1 + x + step2, sum12);
            store<__m512>(d1 + x + step3, sum13);

            if (remains < 3) continue;
            store<__m512>(d2 + x +     0, sum20);
            store<__m512>(d2 + x + step1, sum21);
            store<__m512>(d2 + x + step2, sum22);
            store<__m512>(d2 + x + step3, sum23);

            if (remains < 4) continue;
            store<__m512>(d3 + x +     0, sum30);
            store<__m512>(d3 + x + step1, sum31);
            store<__m512>(d3 + x + step2, sum32);
            store<__m512>(d3 + x + step3, sum33);

            if (remains < 5) continue;
            store<__m512>(d4 + x +     0, sum40);
            store<__m512>(d4 + x + step1, sum41);
            store<__m512>(d4 + x + step2, sum42);
            store<__m512>(d4 + x + step3, sum43);

            if (remains < 6) continue;
            store<__m512>(d5 + x +     0, sum50);
            store<__m512>(d5 + x + step1, sum51);
            store<__m512>(d5 + x + step2, sum52);
            store<__m512>(d5 + x + step3, sum53);
        }
        else if constexpr (is_same_v<Td, uint16_t>) {
            __m512i data0 = cvtps_epu16<__m512i, __m512>(sum00, sum01);
            __m512i data1 = cvtps_epu16<__m512i, __m512>(sum02, sum03);
            store<__m512i>(d0 + x, data0);
            store<__m512i>(d0 + x + step2, data1);
            if (remains < 2) continue;

            data0 = cvtps_epu16<__m512i, __m512>(sum10, sum11);
            data1 = cvtps_epu16<__m512i, __m512>(sum12, sum13);
            store<__m512i>(d1 + x, data0);
            store<__m512i>(d1 + x + step2, data1);
            if (remains < 3) continue;

            data0 = cvtps_epu16<__m512i, __m512>(sum20, sum21);
            data1 = cvtps_epu16<__m512i, __m512>(sum22, sum23);
            store<__m512i>(d2 + x, data0);
            store<__m512i>(d2 + x + step2, data1);
            if (remains < 4) continue;

            data0 = cvtps_epu16<__m512i, __m512>(sum30, sum31);
            data1 = cvtps_epu16<__m512i, __m512>(sum32, sum33);
            store<__m512i>(d3 + x, data0);
            store<__m512i>(d3 + x + step2, data1);
            if (remains < 5) continue;

            data0 = cvtps_epu16<__m512i, __m512>(sum40, sum41);
            data1 = cvtps_epu16<__m512i, __m512>(sum42, sum43);
            store<__m512i>(d4 + x, data0);
            store<__m512i>(d4 + x + step2, data1);
            if (remains < 6) continue;

            data0 = cvtps_epu16<__m512i, __m512>(sum50, sum51);
            data1 = cvtps_epu16<__m512i, __m512>(sum52, sum53);
            store<__m512i>(d5 + x, data0);
            store<__m512i>(d5 + x + step2, data1);
        }
        else if constexpr (is_same_v<Td, uint8_t>) {
            __m512i data = cvtps_epu8_2(sum00, sum01, sum02, sum03);
            store<__m512i>(d0 + x, data);
            if (remains < 2) continue;

            data = cvtps_epu8_2(sum10, sum11, sum12, sum13);
            store<__m512i>(d1 + x, data);
            if (remains < 3) continue;

            data = cvtps_epu8_2(sum20, sum21, sum22, sum23);
            store<__m512i>(d2 + x, data);
            if (remains < 4) continue;

            data = cvtps_epu8_2(sum30, sum31, sum32, sum33);
            store<__m512i>(d3 + x, data);
            if (remains < 5) continue;

            data = cvtps_epu8_2(sum40, sum41, sum42, sum43);
            store<__m512i>(d4 + x, data);
            if (remains < 6) continue;

            data = cvtps_epu8_2(sum50, sum51, sum52, sum53);
            store<__m512i>(d5 + x, data);
        }
    }
}


template <typename Ts, int RADIUS, typename Td>
SFINLINE void gblur(const void* srcp, int spitch, float* hbuffp, int hbpitch, void* dstp,
    int dpitch, int width, int height, int radius, const float* weights,
    const float)
{
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    Td* d = reinterpret_cast<Td*>(dstp);

    constexpr size_t step = sizeof(__m512) / sizeof(float);
    const int length = radius * 2 + 1;
    std::vector<const Ts*> ptr(length + 5, nullptr);
    ptr[radius] = s;
    for (int r = 1; r <= radius; ++r) {
        ptr[radius + r] = s + r * spitch;
        ptr[radius - r] = ptr[radius + r];
    }
    ptr[length + 0] = s + spitch * (radius + 1);
    ptr[length + 1] = s + spitch * (radius + 2);
    ptr[length + 2] = s + spitch * (radius + 3);
    ptr[length + 3] = s + spitch * (radius + 4);
    ptr[length + 4] = s + spitch * (radius + 5);

    float* hb[6] = {
        hbuffp,
        hbuffp + 1 * hbpitch,
        hbuffp + 2 * hbpitch,
        hbuffp + 3 * hbpitch,
        hbuffp + 4 * hbpitch,
        hbuffp + 5 * hbpitch,
    };

    for (int y = 0; y < height; y += 6) {
        for (int x = 0; x < width; x += step) {
            __m512 k0 = set1_ps<__m512>(weights[0]);
            __m512 val = cvtepuX_ps<__m512, Ts>(ptr[0] + x);
            __m512 sum0 = fmul<__m512>(k0, val);

            __m512 k1 = set1_ps<__m512>(weights[1]);
            val = cvtepuX_ps<__m512, Ts>(ptr[1] + x);
            sum0 = fmadd<__m512>(k1, val, sum0);
            __m512 sum1 = fmul<__m512>(k0, val);

            __m512 k2 = set1_ps<__m512>(weights[2]);
            val = cvtepuX_ps<__m512, Ts>(ptr[2] + x);
            sum0 = fmadd<__m512>(k2, val, sum0);
            sum1 = fmadd<__m512>(k1, val, sum1);
            __m512 sum2 = fmul<__m512>(k0, val);

            if constexpr (RADIUS == 1) {
                store<__m512>(hb[0] + x, sum0);

                val = cvtepuX_ps<__m512, Ts>(ptr[3] + x);
                sum1 = fmadd<__m512>(k2, val, sum1);
                sum2 = fmadd<__m512>(k1, val, sum2);
                __m512 sum3 = fmul<__m512>(k0, val);
                store<__m512>(hb[1] + x, sum1);

                val = cvtepuX_ps<__m512, Ts>(ptr[4] + x);
                sum2 = fmadd<__m512>(k2, val, sum2);
                sum3 = fmadd<__m512>(k1, val, sum3);
                __m512 sum4 = fmul<__m512>(k0, val);
                store<__m512>(hb[2] + x, sum2);

                val = cvtepuX_ps<__m512, Ts>(ptr[5] + x);
                sum3 = fmadd<__m512>(k2, val, sum3);
                sum4 = fmadd<__m512>(k1, val, sum4);
                __m512 sum5 = fmul<__m512>(k0, val);
                store<__m512>(hb[3] + x, sum3);

                val = cvtepuX_ps<__m512, Ts>(ptr[6] + x);
                sum4 = fmadd<__m512>(k2, val, sum4);
                sum5 = fmadd<__m512>(k1, val, sum5);
                store<__m512>(hb[4] + x, sum4);

                val = cvtepuX_ps<__m512, Ts>(ptr[7] + x);
                sum5 = fmadd<__m512>(k2, val, sum5);
                store<__m512>(hb[5] + x, sum5);

            } else if constexpr (RADIUS == 2) {
                __m512 k3 = set1_ps<__m512>(weights[3]);
                val = cvtepuX_ps<__m512, Ts>(ptr[3] + x);
                sum0 = fmadd<__m512>(k3, val, sum0);
                sum1 = fmadd<__m512>(k2, val, sum1);
                sum2 = fmadd<__m512>(k1, val, sum2);
                __m512 sum3 = fmul<__m512>(k0, val);

                __m512 k4 = set1_ps<__m512>(weights[4]);
                val = cvtepuX_ps<__m512, Ts>(ptr[4] + x);
                sum0 = fmadd<__m512>(k4, val, sum0);
                sum1 = fmadd<__m512>(k3, val, sum1);
                sum2 = fmadd<__m512>(k2, val, sum2);
                sum3 = fmadd<__m512>(k1, val, sum3);
                __m512 sum4 = fmul<__m512>(k0, val);
                store<__m512>(hb[0] + x, sum0);

                val = cvtepuX_ps<__m512, Ts>(ptr[5] + x);
                sum1 = fmadd<__m512>(k4, val, sum1);
                sum2 = fmadd<__m512>(k3, val, sum2);
                sum3 = fmadd<__m512>(k2, val, sum3);
                sum4 = fmadd<__m512>(k1, val, sum4);
                __m512 sum5 = fmul<__m512>(k0, val);
                store<__m512>(hb[1] + x, sum1);

                val = cvtepuX_ps<__m512, Ts>(ptr[6] + x);
                sum2 = fmadd<__m512>(k4, val, sum2);
                sum3 = fmadd<__m512>(k3, val, sum3);
                sum4 = fmadd<__m512>(k2, val, sum4);
                sum5 = fmadd<__m512>(k1, val, sum5);
                store<__m512>(hb[2] + x, sum2);

                val = cvtepuX_ps<__m512, Ts>(ptr[7] + x);
                sum3 = fmadd<__m512>(k4, val, sum3);
                sum4 = fmadd<__m512>(k3, val, sum4);
                sum5 = fmadd<__m512>(k2, val, sum5);
                store<__m512>(hb[3] + x, sum3);

                val = cvtepuX_ps<__m512, Ts>(ptr[8] + x);
                sum4 = fmadd<__m512>(k4, val, sum4);
                sum5 = fmadd<__m512>(k3, val, sum5);
                store<__m512>(hb[4] + x, sum4);

                val = cvtepuX_ps<__m512, Ts>(ptr[9] + x);
                sum5 = fmadd<__m512>(k4, val, sum5);
                store<__m512>(hb[5] + x, sum5);

            } else {
                __m512 k3 = set1_ps<__m512>(weights[3]);
                val = cvtepuX_ps<__m512, Ts>(ptr[3] + x);
                sum0 = fmadd<__m512>(k3, val, sum0);
                sum1 = fmadd<__m512>(k2, val, sum1);
                sum2 = fmadd<__m512>(k1, val, sum2);
                __m512 sum3 = fmul<__m512>(k0, val);

                __m512 k4 = set1_ps<__m512>(weights[4]);
                val = cvtepuX_ps<__m512, Ts>(ptr[4] + x);
                sum0 = fmadd<__m512>(k4, val, sum0);
                sum1 = fmadd<__m512>(k3, val, sum1);
                sum2 = fmadd<__m512>(k2, val, sum2);
                sum3 = fmadd<__m512>(k1, val, sum3);
                __m512 sum4 = fmul<__m512>(k0, val);

                __m512 k5 = set1_ps<__m512>(weights[5]);
                val = cvtepuX_ps<__m512, Ts>(ptr[5] + x);
                sum0 = fmadd<__m512>(k5, val, sum0);
                sum1 = fmadd<__m512>(k4, val, sum1);
                sum2 = fmadd<__m512>(k3, val, sum2);
                sum3 = fmadd<__m512>(k2, val, sum3);
                sum4 = fmadd<__m512>(k1, val, sum4);
                __m512 sum5 = fmul<__m512>(k0, val);

                for (int v = 6; v < length; ++v) {
                    k0 = k1;
                    k1 = k2;
                    k2 = k3;
                    k3 = k4;
                    k4 = k5;
                    k5 = set1_ps<__m512>(weights[v]);
                    val = cvtepuX_ps<__m512, Ts>(ptr[v] + x);
                    sum0 = fmadd<__m512>(k5, val, sum0);
                    sum1 = fmadd<__m512>(k4, val, sum1);
                    sum2 = fmadd<__m512>(k3, val, sum2);
                    sum3 = fmadd<__m512>(k2, val, sum3);
                    sum4 = fmadd<__m512>(k1, val, sum4);
                    sum5 = fmadd<__m512>(k0, val, sum5);
                }
                store<__m512>(hb[0] + x, sum0);

                val = cvtepuX_ps<__m512, Ts>(ptr[length] + x);
                sum1 = fmadd<__m512>(k5, val, sum1);
                sum2 = fmadd<__m512>(k4, val, sum2);
                sum3 = fmadd<__m512>(k3, val, sum3);
                sum4 = fmadd<__m512>(k2, val, sum4);
                sum5 = fmadd<__m512>(k1, val, sum5);
                store<__m512>(hb[1] + x, sum1);

                val = cvtepuX_ps<__m512, Ts>(ptr[length + 1] + x);
                sum2 = fmadd<__m512>(k5, val, sum2);
                sum3 = fmadd<__m512>(k4, val, sum3);
                sum4 = fmadd<__m512>(k3, val, sum4);
                sum5 = fmadd<__m512>(k2, val, sum5);
                store<__m512>(hb[2] + x, sum2);

                val = cvtepuX_ps<__m512, Ts>(ptr[length + 2] + x);
                sum3 = fmadd<__m512>(k5, val, sum3);
                sum4 = fmadd<__m512>(k4, val, sum4);
                sum5 = fmadd<__m512>(k3, val, sum5);
                store<__m512>(hb[3] + x, sum3);

                val = cvtepuX_ps<__m512, Ts>(ptr[length + 3] + x);
                sum4 = fmadd<__m512>(k5, val, sum4);
                sum5 = fmadd<__m512>(k4, val, sum5);
                store<__m512>(hb[4] + x, sum4);

                val = cvtepuX_ps<__m512, Ts>(ptr[length + 4] + x);
                sum5 = fmadd<__m512>(k5, val, sum5);
                store<__m512>(hb[5] + x, sum5);
            }
        }
        int remains = std::min(height - y, 6);
        hblur(hbuffp, hbpitch, d, dpitch, width, radius, weights, remains);
        d += dpitch * 6;

        for (int i = 0; i < 6; ++i) {
            for (int l = 0; l < length + 4; ++l) {
                ptr[l] = ptr[l + 1];
            }
            if (y < height - 6 - radius - i) {
                ptr[length + 4] += spitch;
            } else if (y >= height - 6 - radius - i) {
                ptr[length + 4] -= spitch;
            }
        }
    }
}



void cvt2flt_avx512_u8(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<uint8_t>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void cvt2flt_avx512_u16(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<uint16_t>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void cvt2flt_avx512_flt(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    convert_to_float<float>(srcp, spitch, 0, 0, dstp, dpitch, width, height,
        0, nullptr, 0);
}

void gblur_avx512_u8_r1_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 1, uint8_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u8_r2_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 2, uint8_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u8_r3_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 3, uint8_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u8_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u8_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u8_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint8_t, 3, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r1_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 1, uint16_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r2_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 2, uint16_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r3_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 3, uint16_t>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_u16_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<uint16_t, 3, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_flt_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<float, 1, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_flt_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<float, 2, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

void gblur_avx512_flt_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float)
{
    gblur<float, 3, float>(srcp, spitch, hbuffp, fbpitch, dstp, dpitch,
        width, height, radius, weights, 0);
}

