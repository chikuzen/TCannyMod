/*
  gaussian_blur.h

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

#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <cstdint>
#include "simd.h"


template <typename Vf, arch_t ARCH>
static void __stdcall
convert_to_float(const size_t width, const size_t height, const uint8_t* srcp,
                 const int src_pitch, float* blurp, const size_t blur_pitch)
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x += step) {
            Vf val = cvtu8_ps<Vf, ARCH>(srcp + x);
            stream<Vf>(blurp + x, val);
        }
        srcp += src_pitch;
        blurp += blur_pitch;
    }
}


template <typename Vf>
static void
horizontal_blur(const float* hkernel, float* buffp, const int radius,
                const size_t width, float* blurp)
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);
    const int length = radius * 2 + 1;

    for (int i = 1; i <= radius ; ++i) {
        buffp[-i] = buffp[i - 1];
        buffp[width - 1 + i] = buffp[width - i];
    }

    for (size_t x = 0; x < width; x += step) {
        Vf sum = zero<Vf>();
        for (int i = -radius; i <= radius; ++i) {
            Vf k = load<Vf>(hkernel + (i + radius) * step);
            Vf val = loadu<Vf>(buffp + x + i);
            sum = madd(k, val, sum);
        }
        stream<Vf>(blurp + x, sum);
    }
}


template <typename Vf, size_t MAX_LENGTH, arch_t ARCH>
static void __stdcall
gaussian_blur(const int radius, const float* kernel, const float* hkernel,
              float* buffp, float* blurp, const size_t blur_pitch,
              const uint8_t* srcp, const size_t src_pitch, const size_t width,
              const size_t height)
{
    if (radius == 0) {
        convert_to_float<Vf, ARCH>(
                width, height, srcp, src_pitch, blurp, blur_pitch);
        return;
    }

    constexpr size_t align = sizeof(Vf);
    constexpr size_t step = align / sizeof(float);
    const int length = radius * 2 + 1;

    const uint8_t* p[MAX_LENGTH];
    p[radius] = srcp;
    for (int r = 1; r <= radius; ++r) {
        p[radius + r] = srcp + r * src_pitch;
        p[radius - r] = p[radius + r - 1];
    }

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x += step) {
            Vf sum = zero<Vf>();

            for (int l = 0; l < length; ++l) {
                Vf input = cvtu8_ps<Vf, ARCH>(p[l] + x);
                Vf k = set1_ps<Vf>(kernel[l]);

                sum = madd(k, input, sum);
            }
            store<Vf>(buffp + x, sum);
        }
        horizontal_blur<Vf>(hkernel, buffp, radius, width, blurp);
        blurp += blur_pitch;

        for (int l = 0; l < length - 1; ++l) {
            p[l] = p[l + 1];
        }
        if (y < height - 1 - radius) {
            p[length - 1] += src_pitch;
        } else if (y > height - 1 - radius) {
            p[length - 1] -= src_pitch;
        }
    }
}


using gaussian_blur_t = void(__stdcall *)(
    const int radius, const float* kernel, const float* hkernel, float* buffp,
    float* blurp, const size_t blur_pitch, const uint8_t* srcp,
    const size_t src_pitch, const size_t width, const size_t height);

#endif

