/*
  edge_detect.cpp

  This file is part of TCannyMod

  Copyright (C) 2013 Oka Motofumi

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
#include <float.h>
#include <math.h>
#include <emmintrin.h>
#include "tcannymod.hpp"


static inline void line_copy(float* dstp, const float* srcp, int width)
{
    memcpy(dstp, srcp, width * sizeof(float));
    dstp[-1] = dstp[0];
    dstp[width] = dstp[width - 1];
}


static inline __m128 mm_abs_ps(const __m128& val)
{
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    return _mm_and_ps(val, mask);
}


static inline __m128 mm_ivtsign_ps(const __m128& val)
{
    const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    return _mm_xor_ps(val, mask);
}


static inline __m128 mm_rcp_hq_ps(const __m128& xmm0)
{
    __m128 rcp = _mm_rcp_ps(xmm0);
    __m128 xmm1 =  _mm_mul_ps(_mm_mul_ps(xmm0, rcp), rcp);
    rcp = _mm_add_ps(rcp, rcp);
    return _mm_sub_ps(rcp, xmm1);
}


static inline __m128i mm_calc_dir(const __m128& gx, const __m128& gy)
{
    static const float t0225 = (float)(sqrt(2.0) - 1.0); // tan(pi/8)
    static const float t0675 = (float)(sqrt(2.0) + 1.0); // tan(3*pi/8)
    static const float t1125 = -t0675;
    static const float t1575 = -t0225;

    __m128 th0 = _mm_setzero_ps();
    __m128 th1 = _mm_set1_ps(90.0f);

    __m128 mask = _mm_cmplt_ps(gy, th0);
    __m128 gx2 = _mm_or_ps(_mm_and_ps(mask, mm_ivtsign_ps(gx)),
                           _mm_andnot_ps(mask, gx));

    __m128 tan = _mm_mul_ps(mm_rcp_hq_ps(gx2), mm_abs_ps(gy));
    mask = _mm_cmpord_ps(tan, tan);
    tan = _mm_or_ps(_mm_and_ps(mask, tan), _mm_andnot_ps(mask, th1));

    th0 = _mm_set1_ps(t0225);
    th1 = _mm_set1_ps(t0675);
    __m128 th2 = _mm_set1_ps(t1125);
    __m128 th3 = _mm_set1_ps(t1575);
    __m128i d0 = _mm_castps_si128(_mm_and_ps(_mm_cmpge_ps(tan, th3),
                                             _mm_cmplt_ps(tan, th0)));
    __m128i d1 = _mm_castps_si128(_mm_and_ps(_mm_cmpge_ps(tan, th0),
                                             _mm_cmplt_ps(tan, th1)));
    __m128i d2 = _mm_castps_si128(_mm_or_ps(_mm_cmpge_ps(tan, th1),
                                            _mm_cmplt_ps(tan, th2)));
    __m128i d3 = _mm_castps_si128(_mm_and_ps(_mm_cmpge_ps(tan, th2),
                                             _mm_cmplt_ps(tan, th3)));

    d0 = _mm_srli_epi32(d0, 31); // 1
    d1 = _mm_srli_epi32(d1, 30); // 3
    d2 = _mm_srli_epi32(d2, 29); // 7
    d3 = _mm_srli_epi32(d3, 28); // 15
    return _mm_or_si128(_mm_or_si128(d0, d1), _mm_or_si128(d2, d3));
}


void __stdcall TCannyM::edge_detect(int width, int height)
{
    const float* srcp = blur_frame;
    float* dstp = edge_mask;
    uint8_t *dir = direction;

    float* p0 = buff + 8;
    float* p1 = p0 + buff_pitch;
    float* p2 = p1 + buff_pitch;
    float* orig = p0;
    float* end = p2;

    line_copy(p0, srcp, width);
    line_copy(p1, srcp, width);

    const __m128 tmin = _mm_set1_ps(th_min);
    const __m128 zero = _mm_setzero_ps();

    for (int y = 0; y < height; y++) {
        srcp += frame_pitch * (y < height - 1 ? 1 : 0);
        line_copy(p2, srcp, width);
        for (int x = 0; x < width; x += 16) {
            __m128i d[4];
            for (int i = 0; i < 4; i++) {
                __m128 gy = _mm_sub_ps(_mm_load_ps(p0 + x + i * 4),
                                       _mm_load_ps(p2 + x + i * 4));
                __m128 gx = _mm_sub_ps(_mm_loadu_ps(p1 + x + i * 4 + 1),
                                       _mm_loadu_ps(p1 + x + i * 4 - 1));

                d[i] = mm_calc_dir(gx, gy);

                gx = _mm_add_ps(_mm_mul_ps(gx, gx), _mm_mul_ps(gy, gy));
                gx = _mm_sqrt_ps(gx);
                _mm_store_ps(dstp + x + i * 4, gx);
            }

            d[0] = _mm_packs_epi32(d[0], d[1]);
            d[1] = _mm_packs_epi32(d[2], d[3]);
            d[0] = _mm_packs_epi16(d[0], d[1]);
            _mm_store_si128((__m128i*)(dir + x), d[0]);
        }
        dstp += frame_pitch;
        dir += frame_pitch;
        p0 = p1;
        p1 = p2;
        p2 = (p2 == end) ? orig : p2 + buff_pitch;
    }
}


void __stdcall TCannyM::non_max_suppress(int width, int height)
{
    const float* edgep = edge_mask;
    float* dstp = blur_frame;
    const uint8_t* dir = direction;

    memset(dstp, 0, frame_pitch * sizeof(float));
    edgep += frame_pitch;
    dstp += frame_pitch;
    memcpy(dstp, edgep, frame_pitch * sizeof(float) * (height - 2));

    for (int y = 1; y < height - 1; y++) {
        dir += frame_pitch;
        dstp[0] = -FLT_MAX;
        for (int x = 1; x < width - 1; x++) {
            float p0;
            if (dir[x] == 1) {
                p0 = max(edgep[x + 1], edgep[x - 1]);
            } else if (dir[x] == 3) {
                p0 = max(edgep[x + 1 - frame_pitch], edgep[x - 1 + frame_pitch]);
            } else if (dir[x] == 7) {
                p0 = max(edgep[x - frame_pitch], edgep[x + frame_pitch]);
            } else {
                p0 = max(edgep[x - 1 - frame_pitch], edgep[x + 1 + frame_pitch]);
            }
            if (edgep[x] < p0) {
                dstp[x] = -FLT_MAX;
            }
        }
        dstp[width - 1] = -FLT_MAX;
        edgep += frame_pitch;
        dstp += frame_pitch;
    }
    memset(dstp, 0, frame_pitch * sizeof(float));
}


void __stdcall TCannyM::hysteresiss(int width, int height)
{
    uint8_t* map = hysteresiss_map;
    float* edgep = blur_frame;
    int pitch = frame_pitch;
    float tmax = th_max;
    float tmin = th_min;

    memset(map, 0, width * height);
    std::vector<int32_t> stack(1024);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edgep[x + y * pitch] < tmax || map[x + y * width]) {
                continue;
            }
            edgep[x + y * pitch] = FLT_MAX;
            map[x + y * width] = 0xFF;
            stack.push_back((x << 16) | y);

            while (!stack.empty()) {
                int32_t posx = stack.back();
                stack.pop_back();
                int32_t posy = posx & 0xFFFF;
                posx >>= 16;
                int32_t xmin = posx > 0 ? posx - 1 : 0;
                int32_t xmax = posx < width - 1 ? posx + 1 : posx;
                int32_t ymin = posy > 0 ? posy - 1 : 0;
                int32_t ymax = posy < height - 1 ? posy + 1 : posy;
                for (int32_t yy = ymin; yy <= ymax; yy++) {
                    for (int32_t xx = xmin; xx <= xmax; xx++) {
                        if (edgep[xx + yy * pitch] > tmin
                            && !map[xx + yy * width]) {
                            edgep[xx + yy * pitch] = FLT_MAX;
                            map[xx + yy * width] = 0xFF;
                            stack.push_back((xx << 16) | yy);
                        }
                    }
                }
            }
        }
    }
}
