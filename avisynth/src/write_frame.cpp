/*
  write_frame.cpp

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



#include <cstdint>
#include "tcannymod.h"
#include "simd.h"


template <typename Vf, typename Vi, bool SCALE>
static void __stdcall
write_gradient_mask(const float* srcp, uint8_t* dstp, const size_t width,
                    const size_t height, const size_t dst_pitch,
                    const size_t src_pitch, const float scale) noexcept
{
    constexpr size_t align = sizeof(Vi);
    constexpr size_t step = align / sizeof(float);

    static const Vf vec_scale = set1<Vf, float>(scale);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; x += align) {
            Vf f0 = load<Vf>(srcp + x + step * 0);
            Vf f1 = load<Vf>(srcp + x + step * 1);
            Vf f2 = load<Vf>(srcp + x + step * 2);
            Vf f3 = load<Vf>(srcp + x + step * 3);
            if constexpr (SCALE) {
                f0 = mul(f0, vec_scale);
                f1 = mul(f1, vec_scale);
                f2 = mul(f2, vec_scale);
                f3 = mul(f3, vec_scale);
            }
            Vi x0 = cvtps_i32<Vi, Vf>(f0);
            Vi x1 = cvtps_i32<Vi, Vf>(f1);
            Vi x2 = cvtps_i32<Vi, Vf>(f2);
            Vi x3 = cvtps_i32<Vi, Vf>(f3);

            Vi ret = cvti32_u8(x0, x1, x2, x3);
            stream(dstp + x, ret);
        }
        srcp += src_pitch;
        dstp += dst_pitch;
    }
}


template <typename Vi>
static void __stdcall
write_gradient_direction(const int32_t* dirp, uint8_t* dstp,
                         const size_t dir_pitch, const size_t dst_pitch,
                         const size_t width, const size_t height) noexcept
{
    constexpr size_t align = sizeof(Vi);
    constexpr size_t step = align / sizeof(int32_t);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; x += align) {
            Vi x0 = load<Vi>(dirp + x + step * 0);
            Vi x1 = load<Vi>(dirp + x + step * 1);
            Vi x2 = load<Vi>(dirp + x + step * 2);
            Vi x3 = load<Vi>(dirp + x + step * 3);
            Vi dst = cvti32_u8(x0, x1, x2, x3);
            stream(dstp + x, dst);
        }
        dirp += dir_pitch;
        dstp += dst_pitch;
    }
}


template <typename Vi>
static void __stdcall
write_edge_direction(const int32_t* dirp, const uint8_t* hystp, uint8_t* dstp,
                     const size_t dir_pitch, const size_t hyst_pitch,
                     const size_t dst_pitch, const size_t width,
                     const size_t height) noexcept
{
    constexpr size_t align = sizeof(Vi);
    constexpr size_t step = align / sizeof(int32_t);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x += align) {
            const Vi x0 = load<Vi>(dirp + x + step * 0);
            const Vi x1 = load<Vi>(dirp + x + step * 1);
            const Vi x2 = load<Vi>(dirp + x + step * 2);
            const Vi x3 = load<Vi>(dirp + x + step * 3);
            const Vi dir = cvti32_u8(x0, x1, x2, x3);
            const Vi hyst = load<Vi>(hystp + x);
            const Vi dst = _and(dir, hyst);
            stream(dstp + x, dst);
        }
        dirp += dir_pitch;
        hystp += hyst_pitch;
        dstp += dst_pitch;
    }
}


write_gradient_mask_t get_write_gradient_mask(bool scale, arch_t arch) noexcept
{
#if defined(__AVX2__)
    if (arch == HAS_AVX2) {
        return scale ? write_gradient_mask<__m256, __m256i, true>
            : write_gradient_mask<__m256, __m256i, false>;
    }
#endif
    return scale ? write_gradient_mask<__m128, __m128i, true>
        : write_gradient_mask<__m128, __m128i, false>;

}


write_gradient_direction_t get_write_gradient_direction(arch_t arch) noexcept
{
#if defined(__AVX2__)
    if (arch == HAS_AVX2) {
        return write_gradient_direction<__m256i>;
    }
#endif
    return write_gradient_direction<__m128i>;
}

write_edge_direction_t get_write_edge_direction(arch_t arch) noexcept
{
#if defined(__AVX2__)
    if (arch == HAS_AVX2) {
        return write_edge_direction<__m256i>;
    }
#endif
    return write_edge_direction<__m128i>;

}
