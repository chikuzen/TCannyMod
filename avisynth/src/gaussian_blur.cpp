/*
  gaussian_blur.cpp

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


#include <type_traits>
#include <algorithm>
#include <unordered_map>
#include <format>

#include "tcannymod.hpp"
#include "gaussian_blur.hpp"


template <typename Ts>
static void
convert_to_float(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float)
{
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    float* d = reinterpret_cast<float*>(dstp);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            d[x] = s[x];
        }
        d += dpitch;
        s += spitch;
    }
}


template <typename Td>
static inline void
hblur(float* srcp, Td* dstp, int width, int radius, const float* weights,
    const float maxval)
{
    constexpr float ro = 0.5f;

    for (int x = 0; x < width; ++x) {
        float sum = 0.0f;
        for (int v = -radius; v <= radius; ++v) {
            int xc = x + v;
            if (xc < 0) {
                xc = -xc;
            } else if (xc >= width) {
                xc = 2 * (width - 1) - xc;
            }
            sum += srcp[xc] * weights[v];
        }
        if constexpr (!std::is_same_v<Td, float>) {
            sum = std::clamp(sum + ro, 0.0f, maxval);
        }
        dstp[x] = static_cast<Td>(sum);
    }
}


template <typename Ts, typename Td>
static void
gblur(const void* srcp, int spitch, float* hbuffp, int hbpitch, void* dstp,
    int dpitch, int width, int height, int radius, const float* weights,
    const float maxval)
{
    const Ts* s = reinterpret_cast<const Ts*>(srcp);
    Td* d = reinterpret_cast<Td*>(dstp);
    weights += radius;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int v = -radius; v <= radius; ++v) {
                int yc = y + v;
                if (yc < 0) {
                    yc = -yc;
                } else if (yc >= height - 1) {
                    yc = 2 * (height - 1) - yc;
                }
                sum += s[x + spitch * yc] * weights[v];
            }
            hbuffp[x] = sum;
        }
        hblur(hbuffp, d, width, radius, weights, maxval);
        d += dpitch;
    }
}


gblur_t get_gblur(int bytes, arch_t arch, int radius, int mode)
{
    using std::format;
    std::unordered_map<std::string, gblur_t> func;

    if (arch < USE_AVX512) radius = std::min(radius, 2);
    if (arch == USE_AVX512) radius = std::min(radius, 3);
    if (arch == NO_SIMD) radius = 1;

    if (mode & mode_t::DO_NOT_BLUR) {
        func[format("{}{}", a2s(NO_SIMD), 1)] = convert_to_float<uint8_t>;
        func[format("{}{}", a2s(NO_SIMD), 2)] = convert_to_float<uint16_t>;
        func[format("{}{}", a2s(NO_SIMD), 4)] = convert_to_float<float>;
        func[format("{}{}", a2s(USE_SSE4), 1)] = cvt2flt_sse4_u8;
        func[format("{}{}", a2s(USE_SSE4), 2)] = cvt2flt_sse4_u16;
        func[format("{}{}", a2s(USE_SSE4), 4)] = cvt2flt_sse4_flt;
        func[format("{}{}", a2s(USE_AVX2), 1)] = cvt2flt_avx2_u8;
        func[format("{}{}", a2s(USE_AVX2), 2)] = cvt2flt_avx2_u16;
        func[format("{}{}", a2s(USE_AVX2), 4)] = cvt2flt_avx2_flt;
        func[format("{}{}", a2s(USE_AVX512), 1)] = cvt2flt_avx512_u8;
        func[format("{}{}", a2s(USE_AVX512), 2)] = cvt2flt_avx512_u16;
        func[format("{}{}", a2s(USE_AVX512), 4)] = cvt2flt_avx512_flt;

        auto key = format("{}{}", a2s(arch), bytes);
        return func.at(key);
    }

    if (mode & mode_t::DO_BLUR_ONLY) {
        func[format("{}{}{}", a2s(NO_SIMD), 1, 1)] = gblur<uint8_t, uint8_t>;
        func[format("{}{}{}", a2s(NO_SIMD), 2, 1)] = gblur<uint16_t, uint16_t>;
        func[format("{}{}{}", a2s(NO_SIMD), 4, 1)] = gblur<float, float>;
        func[format("{}{}{}", a2s(USE_SSE4), 1, 1)] = gblur_sse4_u8_r1_u8;
        func[format("{}{}{}", a2s(USE_SSE4), 1, 2)] = gblur_sse4_u8_r2_u8;
        func[format("{}{}{}", a2s(USE_SSE4), 2, 1)] = gblur_sse4_u16_r1_u16;
        func[format("{}{}{}", a2s(USE_SSE4), 2, 2)] = gblur_sse4_u16_r2_u16;
        func[format("{}{}{}", a2s(USE_SSE4), 4, 1)] = gblur_sse4_flt_r1_flt;
        func[format("{}{}{}", a2s(USE_SSE4), 4, 2)] = gblur_sse4_flt_r2_flt;
        func[format("{}{}{}", a2s(USE_AVX2), 1, 1)] = gblur_avx2_u8_r1_u8;
        func[format("{}{}{}", a2s(USE_AVX2), 1, 2)] = gblur_avx2_u8_r2_u8;
        func[format("{}{}{}", a2s(USE_AVX2), 2, 1)] = gblur_avx2_u16_r1_u16;
        func[format("{}{}{}", a2s(USE_AVX2), 2, 2)] = gblur_avx2_u16_r2_u16;
        func[format("{}{}{}", a2s(USE_AVX2), 4, 1)] = gblur_avx2_flt_r1_flt;
        func[format("{}{}{}", a2s(USE_AVX2), 4, 2)] = gblur_avx2_flt_r2_flt;
        func[format("{}{}{}", a2s(USE_AVX512), 1, 1)] = gblur_avx512_u8_r1_u8;
        func[format("{}{}{}", a2s(USE_AVX512), 1, 2)] = gblur_avx512_u8_r2_u8;
        func[format("{}{}{}", a2s(USE_AVX512), 1, 3)] = gblur_avx512_u8_r3_u8;
        func[format("{}{}{}", a2s(USE_AVX512), 2, 1)] = gblur_avx512_u16_r1_u16;
        func[format("{}{}{}", a2s(USE_AVX512), 2, 2)] = gblur_avx512_u16_r2_u16;
        func[format("{}{}{}", a2s(USE_AVX512), 2, 3)] = gblur_avx512_u16_r3_u16;
        func[format("{}{}{}", a2s(USE_AVX512), 4, 1)] = gblur_avx512_flt_r1_flt;
        func[format("{}{}{}", a2s(USE_AVX512), 4, 2)] = gblur_avx512_flt_r2_flt;
        func[format("{}{}{}", a2s(USE_AVX512), 4, 3)] = gblur_avx512_flt_r3_flt;

        auto key = format("{}{}{}", a2s(arch), bytes, radius);
        return func.at(key);
    }

    func[format("{}{}{}", a2s(NO_SIMD),  1, 1)] = gblur<uint8_t, float>;
    func[format("{}{}{}", a2s(NO_SIMD),  2, 1)] = gblur<uint16_t, float>;
    func[format("{}{}{}", a2s(NO_SIMD),  4, 1)] = gblur<float, float>;
    func[format("{}{}{}", a2s(USE_SSE4), 1, 1)] = gblur_sse4_u8_r1_flt;
    func[format("{}{}{}", a2s(USE_SSE4), 1, 2)] = gblur_sse4_u8_r2_flt;
    func[format("{}{}{}", a2s(USE_SSE4), 2, 1)] = gblur_sse4_u16_r1_flt;
    func[format("{}{}{}", a2s(USE_SSE4), 2, 2)] = gblur_sse4_u16_r2_flt;
    func[format("{}{}{}", a2s(USE_SSE4), 4, 1)] = gblur_sse4_flt_r1_flt;
    func[format("{}{}{}", a2s(USE_SSE4), 4, 2)] = gblur_sse4_flt_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 1, 1)] = gblur_avx2_u8_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 1, 2)] = gblur_avx2_u8_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 2, 1)] = gblur_avx2_u16_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 2, 2)] = gblur_avx2_u16_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 4, 1)] = gblur_avx2_flt_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX2), 4, 2)] = gblur_avx2_flt_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 1, 1)] = gblur_avx512_u8_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 1, 2)] = gblur_avx512_u8_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 1, 3)] = gblur_avx512_u8_r3_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 2, 1)] = gblur_avx512_u16_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 2, 2)] = gblur_avx512_u16_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 2, 3)] = gblur_avx512_u16_r3_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 4, 1)] = gblur_avx512_flt_r1_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 4, 2)] = gblur_avx512_flt_r2_flt;
    func[format("{}{}{}", a2s(USE_AVX512), 4, 3)] = gblur_avx512_flt_r3_flt;

    auto key = format("{}{}{}", a2s(arch), bytes, radius);
    return func.at(key);
}
