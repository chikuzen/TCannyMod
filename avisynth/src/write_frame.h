/*
  write_frame.h
  
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


#ifndef WRITE_FRAME_H
#define WRITE_FRAME_H

#include <cstdint>
#include "simd.h"


template <typename Vf, typename Vi, bool SCALE>
static void __stdcall
write_gradient_mask(const float* srcp, uint8_t* dstp, const size_t width,
                    const size_t height, const size_t dst_pitch,
                    const size_t src_pitch, const float scale)
{
    constexpr size_t align = sizeof(Vi);
    constexpr size_t step = align / sizeof(float);

    static const Vf vec_scale = set1_ps<Vf>(scale);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; x += align) {
            Vf f0 = load<Vf>(srcp + x + step * 0);
            Vf f1 = load<Vf>(srcp + x + step * 1);
            Vf f2 = load<Vf>(srcp + x + step * 2);
            Vf f3 = load<Vf>(srcp + x + step * 3);
            if (SCALE) {
                f0 = mul_ps(f0, vec_scale);
                f1 = mul_ps(f1, vec_scale);
                f2 = mul_ps(f2, vec_scale);
                f3 = mul_ps(f3, vec_scale);
            }
            Vi x0 = cvtps_i32(f0);
            Vi x1 = cvtps_i32(f1);
            Vi x2 = cvtps_i32(f2);
            Vi x3 = cvtps_i32(f3);

            Vi ret = cvti32_u8(x0, x1, x2, x3);
            stream_si<Vi>(dstp + x, ret);
        }
        srcp += src_pitch;
        dstp += dst_pitch;
    }
}


template <typename Vf, typename Vi>
void __stdcall
write_direction_map(const uint8_t* hystp, const uint8_t* dirp,
                    const size_t dir_pitch, uint8_t* dstp,
                    const size_t dst_pitch, const size_t width,
                    const size_t height)
{
    constexpr size_t step = sizeof(Vi);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x += step) {
            Vi x0 = load<Vi>(hystp + x);
            Vi x1 = load<Vi>(dirp + x);
            x0 = and_si(x0, x1);
            stream_si<Vi>(dstp + x, x0);
        }
        dirp += dir_pitch;
        hystp += dir_pitch;
        dstp += dst_pitch;
    }
}


using write_gradient_mask_t = void(_stdcall *)(
    const float* srcp, uint8_t* dstp, const size_t width,
    const size_t height, const size_t dst_pitch, const size_t src_pitch,
    const float scale);


using write_direction_map_t = void (__stdcall *)(
    const uint8_t* hystp, const uint8_t* dirp, const size_t dir_pitch,
    uint8_t* dstp, const size_t dst_pitch, const size_t width,
    const size_t height);


#endif

