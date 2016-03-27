/*
  edge_detection.h
  
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


#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <cstdint>
#include "simd.h"

template <typename Vf, typename Vi>
SFINLINE void calc_direction(const Vf& gx, const Vf& gy, int32_t* dirp)
{

    static const Vf t0225 = set1_ps<Vf>(std::sqrt(2.0f) - 1.0f); // tan(pi/8)
    static const Vf t0675 = set1_ps<Vf>(std::sqrt(2.0f) + 1.0f); // tan(3*pi/8)
    static const Vf t1125 = sub(zero<Vf>(), t0675); // tan(5*pi/8) = -tan(3*pi/8)
    static const Vf t1575 = sub(zero<Vf>(), t0225); // tan(7*pi/8) = -tan(pi/8)

    Vf z = zero<Vf>();
    Vf vertical = set1_ps<Vf>(90.0f);

    // if gy < 0, gx = -gx
    Vf mask = cmplt_ps(gy, z);
    Vf gx2 = blendv(gx, sub(z, gx), mask);

    // tan = gy / gx
    Vf tan = mul(rcp_hq(gx2), abs(gy));

    // if tan is unorderd(inf or NaN), tan = 90.0f
    mask = cmpord_ps(tan, tan);
    tan = blendv(vertical, tan, mask);

    // if t1575 <= tan < t0225, direction is 31 (horizontal)
    Vi d0 = castps_si(and_ps(cmpge_ps(tan, t1575), cmplt_ps(tan, t0225)));
    d0 = srli_i32(d0, 27);

    // if t0225 <= tan < t0675, direction is 63 (45' up)
    Vi d1 = castps_si(and_ps(cmpge_ps(tan, t0225), cmplt_ps(tan, t0675)));
    d1 = srli_i32(d1, 26);

    // if t0675 <= tan or tan < t1125, direction is 127 (vertical)
    Vi d2 = castps_si(or_ps(cmpge_ps(tan, t0675), cmplt_ps(tan, t1125)));
    d2 = srli_i32(d2, 25);

    // if t1125 <= tan < t1575, direction is 255 (45' down)
    Vi d3 = castps_si(and_ps(cmpge_ps(tan, t1125), cmplt_ps(tan, t1575)));
    d3 = srli_i32(d3, 24);

    d0 = or_si(or_si(d0, d1), or_si(d2, d3));

    stream<Vi>(dirp, d0);
}


template <typename Vf, typename Vi, bool CALC_DIR>
void __stdcall
standard(float* blurp, const size_t blur_pitch, float* emaskp,
         const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
         const size_t width, const size_t height)
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    float* p0 = blurp;
    float* p1 = blurp;
    float* p2 = blurp + blur_pitch;

    for (size_t y = 0; y < height; y++) {
        p1[-1] = p1[0];
        p1[width] = p1[width - 1];

        for (size_t x = 0; x < width; x += step) {
            Vf gy = sub(load<Vf>(p0 + x), load<Vf>(p2 + x)); // [1, 0, -1]
            Vf gx = sub(loadu<Vf>(p1 + x + 1), loadu<Vf>(p1 + x - 1)); // [-1, 0, 1]

            if (CALC_DIR) {
                calc_direction<Vf, Vi>(gx, gy, dirp + x);
            }

            Vf magnitude = mul(gx, gx);
            magnitude = madd(gy, gy, magnitude);
            magnitude = sqrt(magnitude);
            stream<Vf>(emaskp + x, magnitude);
        }
        emaskp += emask_pitch;
        dirp += dir_pitch;
        p0 = p1;
        p1 = p2;
        p2 += y < height - 1 ? blur_pitch : 0;
    }
}



/*
    sobel operator(3x3)
         |    p0   |      p1    |     p2  | 
    H = [-1,  0,  1, -2,  0,  2, -1,  0,  1]
    V = [ 1,  2,  1,  0,  0,  0, -1, -2, -1]
*/
template <typename Vf, typename Vi, bool CALC_DIR>
void __stdcall
sobel(float* blurp, const size_t blur_pitch, float* emaskp,
      const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
      const size_t width, const size_t height)
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    float* p0 = blurp;
    float* p1 = blurp;
    float* p2 = blurp + blur_pitch;

    p1[-1] = p1[0];
    p1[width] = p1[width - 1];

    for (size_t y = 0; y < height; y++) {
        p2[-1] = p2[0];
        p2[width] = p2[width - 1];

        for (size_t x = 0; x < width; x += step) {
            Vf gx = sub(loadu<Vf>(p0 + x + 1), loadu<Vf>(p2 + x - 1));
            Vf gy = gx;
            Vf t = loadu<Vf>(p0 + x - 1);
            gx = sub(gx, t);
            gy = add(gy, t);
            t = loadu<Vf>(p2 + x + 1);
            gx = add(gx, t);
            gy = sub(gy, t);
            t = loadu<Vf>(p1 + x - 1);
            gx = sub(gx, add(t, t));
            t = loadu<Vf>(p1 + x + 1);
            gx = add(gx, add(t, t));
            t = load<Vf>(p0 + x);
            gy = add(gy, add(t, t));
            t = load<Vf>(p2 + x);
            gy = sub(gy, add(t, t));

            if (CALC_DIR) {
                calc_direction<Vf, Vi>(gx, gy, dirp + x);
            }

            Vf magnitude = mul(gx, gx);
            magnitude = madd(gy, gy, magnitude);
            magnitude = sqrt(magnitude);
            stream<Vf>(emaskp + x, magnitude);
        }
        emaskp += emask_pitch;
        dirp += dir_pitch;
        p0 = p1;
        p1 = p2;
        p2 += y < height - 1 ? blur_pitch : 0;
    }
}


template <typename Vf, typename Vi>
void __stdcall
non_max_suppress(const float* emaskp, const size_t em_pitch,
                 const int32_t* dirp, const size_t dir_pitch, float* blurp,
                 const size_t blur_pitch, const size_t width, const size_t height)
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    static const Vf FLT_MAX_neg = set1_ps<Vf>(-FLT_MAX);

    memset(blurp - 4, 0, blur_pitch * sizeof(float));
    for (size_t y = 1; y < height - 1; ++y) {
        memcpy(blurp + blur_pitch * y, emaskp + em_pitch * y,
            width * sizeof(float));
    }
    blurp += blur_pitch;
    emaskp += em_pitch;

    for (size_t y = 1; y < height - 1; ++y) {
        dirp += dir_pitch;
        blurp[-1] = blurp[0] = -FLT_MAX;

        for (size_t x = 1; x < width - 1; x += step) {
            const Vi dir = loadu<Vi>(dirp + x);
            const Vi m135 = srli_i32(cmpeq_i32(dir, dir), 24);
            Vf mask = castsi_ps(cmpeq_i32(dir, m135));
            Vf temp = max(loadu<Vf>(emaskp + x - 1 - em_pitch),
                          loadu<Vf>(emaskp + x + 1 + em_pitch));
            Vf p0 = and_ps(temp, mask);

            mask = castsi_ps(cmpeq_i32(dir, srli_i32(m135, 1)));
            temp = max(loadu<Vf>(emaskp + x - em_pitch),
                       loadu<Vf>(emaskp + x + em_pitch));
            p0 = or_ps(p0, and_ps(temp, mask));

            mask = castsi_ps(cmpeq_i32(dir, srli_i32(m135, 2)));
            temp = max(loadu<Vf>(emaskp + x + 1 - em_pitch),
                       loadu<Vf>(emaskp + x - 1 + em_pitch));
            p0 = or_ps(p0, and_ps(temp, mask));

            mask = castsi_ps(cmpeq_i32(dir, srli_i32(m135, 3)));
            temp = max(loadu<Vf>(emaskp + x - 1),
                       loadu<Vf>(emaskp + x + 1));
            p0 = or_ps(p0, and_ps(temp, mask));

            mask = cmplt_ps(loadu<Vf>(emaskp + x), p0);

            Vf blur = loadu<Vf>(blurp + x);
            blur = blendv(blur, FLT_MAX_neg, mask);
            storeu(blurp + x, blur);
        }
        blurp[width - 1] = blurp[width] = -FLT_MAX;
        emaskp += em_pitch;
        blurp += blur_pitch;
    }
}


using edge_detection_t = void(__stdcall *)(
    float* blurp, const size_t blur_pitch, float* emaskp,
    const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
    const size_t width, const size_t height);


using non_max_suppress_t = void (__stdcall *)(
    const float* emaskp, const size_t em_pitch, const int32_t* dirp,
    const size_t dir_pitch, float* blurp, const size_t blr_pitch,
    const size_t width, const size_t height);


void __stdcall
hysteresis(uint8_t* hystp, const size_t hpitch, float* blurp,
           const size_t bpitch, const int width, const int height,
           const float tmin, const float tmax);

#endif

