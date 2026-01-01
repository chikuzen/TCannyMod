/*
  edgemask.cpp

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
#include <cstring>
#include <format>
#include <unordered_map>
#include "tcannymod.hpp"
#include "edgemask.hpp"


#ifndef SFINLINE
#if defined(_WIN32)
#define SFINLINE static __forceinline
#else
#define SFINLINE static inline __attribute__((always_inline))
#endif
#endif

SFINLINE void
calc_direction(const float gx, const float gy, int32_t* dirp)
{
    constexpr float tan_1_8_pi = 0.414213538169860839843750f;
    constexpr float tan_3_8_pi = 2.414213657379150390625000f;
    constexpr float tan_5_8_pi = -2.414213657379150390625000f;
    constexpr float tan_7_8_pi = -0.414213538169860839843750f;

    // Normalize to four values depending on the angle range.
    //   0 deg (7/8pi to 1/8pi) is  15,
    //  45 deg (1/8pi to 3/8pi) is  31,
    //  90 deg (3/8pi to 5/8pi) is  63,
    // 135 deg (5/8pi to 7/8pi) is 127,

    // inf, -inf and NaN are assumed to be 90 deg.
    // magnitude is always 0 if tangent is NaN, so the angle can be anything.
    if (gx == 0.0f) {
        *dirp = 63;
        return;
    }
    auto tangent = gy / gx;
    if (tan_7_8_pi <= tangent && tangent < tan_1_8_pi) {        //   0 deg
        *dirp = 15;
    } else if (tan_1_8_pi <= tangent && tangent < tan_3_8_pi) { //  45 deg
        *dirp = 31;
    } else if (tan_3_8_pi <= tangent || tangent < tan_5_8_pi) { //  90 deg
        *dirp = 63;
    } else {                                                    // 135 deg
        *dirp = 127;
    }
}


template <typename Td, bool SCALE, int OPERATOR, bool _STRICT, bool CALC_DIR>
static void
emask(const float* blurp, int blpitch, void* dstp, int dpitch, operator_t& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch)
{
    constexpr float ro = std::is_same_v<Td, float> ? 0.0f : 0.5f;
    Td* d = reinterpret_cast<Td*>(dstp);

    const float k0 = opr[0];
    const float k1 = opr[1];
    const float k2 = opr[2];
    memset(d, 0, dpitch * sizeof(Td));
    d += dpitch;

    if constexpr (CALC_DIR) {
        memset(dirp, 0, dirpitch * sizeof(int32_t));
        dirp += dirpitch;
    }

    for (int y = 1; y < height - 1; ++y) {
        const float* above = blurp;
        const float* centr = blurp + blpitch;
        const float* below = centr + blpitch;
        d[0] = 0;
        if constexpr (CALC_DIR) dirp[0] = 0;

        for (int x = 1; x < width - 1; ++x) {
            int L = x - 1, R = x + 1;
            float gx, gy;
            if constexpr (OPERATOR == 0) {
                gx = centr[R] - centr[L];
                gy = above[x] - below[x];
            }
            else if constexpr (OPERATOR == 1) {
                gx = above[R] + centr[R] + centr[R] + below[R] -
                    (above[L] + centr[L] + centr[L] + below[L]);
                gy = above[L] + above[x] + above[x] + above[R] -
                    (below[L] + below[x] + below[x] + below[R]);
            }
            else {
                gx = above[R] * k0 + centr[R] * k1 + below[R] * k2 -
                    (above[L] * k0 + centr[L] * k1 + below[L] * k2);
                gy = above[L] * k0 + above[x] * k1 + above[R] * k2 -
                    (below[L] * k0 + below[x] * k1 + below[R] * k2);
            }
            if constexpr (CALC_DIR) {
                calc_direction(gx, gy, dirp + x);
            }
            float magnitude;
            if constexpr (_STRICT) {
                magnitude = std::sqrt(gx * gx + gy * gy);
            } else {
                magnitude = std::abs(gx) + std::abs(gy);
            }
            if constexpr (SCALE) {
                magnitude *= scale;
            }
            if (!CALC_DIR) {
                magnitude = std::clamp(magnitude + ro, 0.0f, maxval);
            }
            d[x] = static_cast<Td>(magnitude);
        }
        d[width - 1] = 0;
        blurp += blpitch;
        d += dpitch;
        if constexpr (CALC_DIR) {
            dirp[width - 1] = 0;
            dirp += dirpitch;
        }
    }
    memset(d, 0, dpitch * sizeof(Td));
}


template <typename Td>
static void write_directions(const int32_t* dirp, int drpitch, void* dstp,
    int dpitch, int width, int height)
{
    Td* d = reinterpret_cast<Td*>(dstp);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            d[x] = static_cast<Td>(dirp[x]);
        }
        dirp += drpitch;
        d += dpitch;
    }
}


static void
nms(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height)
{
    memset(dstp, 0, dpitch * sizeof(float));

    for (int y = 1; y < height - 1; ++y) {
        emaskp += epitch;
        dirp += dirpitch;
        dstp += dpitch;
        dstp[0] = 0;
        for (int x = 1; x < width - 1; ++x) {
            float v = emaskp[x];
            int dir = dirp[x];
            if (dir == 127) {
                if (v < emaskp[x - 1 - epitch] || v < emaskp[x + 1 + epitch])
                    v = 0;
            } else if (dir == 63) {
                if (v < emaskp[x - epitch] || v < emaskp[x + epitch])
                    v = 0;
            } else if (dir == 31) {
                if (v < emaskp[x + 1 - epitch] || v < emaskp[x - 1 + epitch])
                    v = 0;
            } else {
                if (v < emaskp[x - 1] || v < emaskp[x + 1])
                    v = 0;
            }
            dstp[x] = v;
        }
        dstp[width - 1] = 0;
    }
    memset(dstp + dpitch, 0, dpitch * sizeof(float));
}



edgemask_t get_emask(int bytes, arch_t arch, int mode)
{
    using std::format;
    std::unordered_map<std::string, edgemask_t> func;

                                //arch, bytes, SCALE, OPERATOR, _STRICT, CALC_DIR
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  0, true,  false)] = emask<uint8_t, true,  0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  0, false, false)] = emask<uint8_t, true,  0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  1, true,  false)] = emask<uint8_t, true,  1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  1, false, false)] = emask<uint8_t, true,  1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  2, true,  false)] = emask<uint8_t, true,  2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, true,  2, false, false)] = emask<uint8_t, true,  2, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 0, true,  false)] = emask<uint8_t, false, 0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 0, false, false)] = emask<uint8_t, false, 0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 1, true,  false)] = emask<uint8_t, false, 1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 1, false, false)] = emask<uint8_t, false, 1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 2, true,  false)] = emask<uint8_t, false, 2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 1, false, 2, false, false)] = emask<uint8_t, false, 2, false, false>;

    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  0, true,  false)] = emask<uint16_t, true,  0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  0, false, false)] = emask<uint16_t, true,  0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  1, true,  false)] = emask<uint16_t, true,  1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  1, false, false)] = emask<uint16_t, true,  1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  2, true,  false)] = emask<uint16_t, true,  2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, true,  2, false, false)] = emask<uint16_t, true,  2, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 0, true,  false)] = emask<uint16_t, false, 0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 0, false, false)] = emask<uint16_t, false, 0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 1, true,  false)] = emask<uint16_t, false, 1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 1, false, false)] = emask<uint16_t, false, 1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 2, true,  false)] = emask<uint16_t, false, 2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 2, false, 2, false, false)] = emask<uint16_t, false, 2, false, false>;

    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  0, true,  false)] = emask<float, true,  0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  0, false, false)] = emask<float, true,  0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  1, true,  false)] = emask<float, true,  1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  1, false, false)] = emask<float, true,  1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  2, true,  false)] = emask<float, true,  2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  2, false, false)] = emask<float, true,  2, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 0, true,  false)] = emask<float, false, 0, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 0, false, false)] = emask<float, false, 0, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 1, true,  false)] = emask<float, false, 1, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 1, false, false)] = emask<float, false, 1, false, false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 2, true,  false)] = emask<float, false, 2, true,  false>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 2, false, false)] = emask<float, false, 2, false, false>;

    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  0, true,  true)] = emask<float, true,  0, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  0, false, true)] = emask<float, true,  0, false, true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  1, true,  true)] = emask<float, true,  1, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  1, false, true)] = emask<float, true,  1, false, true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  2, true,  true)] = emask<float, true,  2, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, true,  2, false, true)] = emask<float, true,  2, false, true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 0, true,  true)] = emask<float, false, 0, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 0, false, true)] = emask<float, false, 0, false, true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 1, true,  true)] = emask<float, false, 1, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 1, false, true)] = emask<float, false, 1, false, true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 2, true,  true)] = emask<float, false, 2, true,  true>;
    func[format("{}{}{}{}{}{}", a2s(NO_SIMD), 4, false, 2, false, true)] = emask<float, false, 2, false, true>;

    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 0, true,  false)] = emask_sse4_u8_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 0, false, false)] = emask_sse4_u8_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 1, true,  false)] = emask_sse4_u8_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 1, false, false)] = emask_sse4_u8_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 2, true,  false)] = emask_sse4_u8_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, false, 2, false, false)] = emask_sse4_u8_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  0, true,  false)] = emask_sse4_u8_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  0, false, false)] = emask_sse4_u8_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  1, true,  false)] = emask_sse4_u8_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  1, false, false)] = emask_sse4_u8_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  2, true,  false)] = emask_sse4_u8_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 1, true,  2, false, false)] = emask_sse4_u8_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 0, true,  false)] = emask_sse4_u16_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 0, false, false)] = emask_sse4_u16_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 1, true,  false)] = emask_sse4_u16_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 1, false, false)] = emask_sse4_u16_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 2, true,  false)] = emask_sse4_u16_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, false, 2, false, false)] = emask_sse4_u16_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  0, true,  false)] = emask_sse4_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  0, false, false)] = emask_sse4_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  1, true,  false)] = emask_sse4_u16_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  1, false, false)] = emask_sse4_u16_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  2, true,  false)] = emask_sse4_u16_sc_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 2, true,  2, false, false)] = emask_sse4_u16_sc_custom_strict;

    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 0, true,  false)] = emask_sse4_flt_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 0, false, false)] = emask_sse4_flt_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 1, true,  false)] = emask_sse4_flt_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 1, false, false)] = emask_sse4_flt_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 2, true,  false)] = emask_sse4_flt_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 2, false, false)] = emask_sse4_flt_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  0, true,  false)] = emask_sse4_flt_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  0, false, false)] = emask_sse4_flt_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  1, true,  false)] = emask_sse4_flt_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  1, false, false)] = emask_sse4_flt_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  2, true,  false)] = emask_sse4_flt_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  2, false, false)] = emask_sse4_flt_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 0, true,  true)] = emask_sse4_flt_ns_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 0, false, true)] = emask_sse4_flt_ns_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 1, true,  true)] = emask_sse4_flt_ns_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 1, false, true)] = emask_sse4_flt_ns_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 2, true,  true)] = emask_sse4_flt_ns_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, false, 2, false, true)] = emask_sse4_flt_ns_custom_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  0, true,  true)] = emask_sse4_flt_sc_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  0, false, true)] = emask_sse4_flt_sc_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  1, true,  true)] = emask_sse4_flt_sc_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  1, false, true)] = emask_sse4_flt_sc_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  2, true,  true)] = emask_sse4_flt_sc_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_SSE4), 4, true,  2, false, true)] = emask_sse4_flt_sc_custom_fast_dir;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 0, true,  false)] = emask_avx2_u8_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 0, false, false)] = emask_avx2_u8_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 1, true,  false)] = emask_avx2_u8_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 1, false, false)] = emask_avx2_u8_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 2, true,  false)] = emask_avx2_u8_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, false, 2, false, false)] = emask_avx2_u8_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  0, true,  false)] = emask_avx2_u8_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  0, false, false)] = emask_avx2_u8_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  1, true,  false)] = emask_avx2_u8_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  1, false, false)] = emask_avx2_u8_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  2, true,  false)] = emask_avx2_u8_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 1, true,  2, false, false)] = emask_avx2_u8_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 0, true,  false)] = emask_avx2_u16_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 0, false, false)] = emask_avx2_u16_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 1, true,  false)] = emask_avx2_u16_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 1, false, false)] = emask_avx2_u16_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 2, true,  false)] = emask_avx2_u16_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, false, 2, false, false)] = emask_avx2_u16_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  0, true,  false)] = emask_avx2_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  0, false, false)] = emask_avx2_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  1, true,  false)] = emask_avx2_u16_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  1, false, false)] = emask_avx2_u16_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  2, true,  false)] = emask_avx2_u16_sc_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 2, true,  2, false, false)] = emask_avx2_u16_sc_custom_strict;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 0, true,  false)] = emask_avx2_flt_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 0, false, false)] = emask_avx2_flt_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 1, true,  false)] = emask_avx2_flt_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 1, false, false)] = emask_avx2_flt_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 2, true,  false)] = emask_avx2_flt_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 2, false, false)] = emask_avx2_flt_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  0, true,  false)] = emask_avx2_flt_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  0, false, false)] = emask_avx2_flt_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  1, true,  false)] = emask_avx2_flt_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  1, false, false)] = emask_avx2_flt_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  2, true,  false)] = emask_avx2_flt_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  2, false, false)] = emask_avx2_flt_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 0, true,  true)] = emask_avx2_flt_ns_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 0, false, true)] = emask_avx2_flt_ns_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 1, true,  true)] = emask_avx2_flt_ns_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 1, false, true)] = emask_avx2_flt_ns_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 2, true,  true)] = emask_avx2_flt_ns_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, false, 2, false, true)] = emask_avx2_flt_ns_custom_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  0, true,  true)] = emask_avx2_flt_sc_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  0, false, true)] = emask_avx2_flt_sc_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  1, true,  true)] = emask_avx2_flt_sc_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  1, false, true)] = emask_avx2_flt_sc_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  2, true,  true)] = emask_avx2_flt_sc_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX2), 4, true,  2, false, true)] = emask_avx2_flt_sc_custom_fast_dir;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 0, true,  false)] = emask_avx512_u8_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 0, false, false)] = emask_avx512_u8_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 1, true,  false)] = emask_avx512_u8_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 1, false, false)] = emask_avx512_u8_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 2, true,  false)] = emask_avx512_u8_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, false, 2, false, false)] = emask_avx512_u8_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  0, true,  false)] = emask_avx512_u8_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  0, false, false)] = emask_avx512_u8_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  1, true,  false)] = emask_avx512_u8_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  1, false, false)] = emask_avx512_u8_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  2, true,  false)] = emask_avx512_u8_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 1, true,  2, false, false)] = emask_avx512_u8_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 0, true,  false)] = emask_avx512_u16_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 0, false, false)] = emask_avx512_u16_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 1, true,  false)] = emask_avx512_u16_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 1, false, false)] = emask_avx512_u16_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 2, true,  false)] = emask_avx512_u16_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, false, 2, false, false)] = emask_avx512_u16_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  0, true,  false)] = emask_avx512_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  0, false, false)] = emask_avx512_u16_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  1, true,  false)] = emask_avx512_u16_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  1, false, false)] = emask_avx512_u16_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  2, true,  false)] = emask_avx512_u16_sc_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 2, true,  2, false, false)] = emask_avx512_u16_sc_custom_strict;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 0, true,  false)] = emask_avx512_flt_ns_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 0, false, false)] = emask_avx512_flt_ns_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 1, true,  false)] = emask_avx512_flt_ns_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 1, false, false)] = emask_avx512_flt_ns_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 2, true,  false)] = emask_avx512_flt_ns_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 2, false, false)] = emask_avx512_flt_ns_custom_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  0, true,  false)] = emask_avx512_flt_sc_std_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  0, false, false)] = emask_avx512_flt_sc_std_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  1, true,  false)] = emask_avx512_flt_sc_sobel_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  1, false, false)] = emask_avx512_flt_sc_sobel_fast;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  2, true,  false)] = emask_avx512_flt_sc_custom_strict;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  2, false, false)] = emask_avx512_flt_sc_custom_fast;

    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 0, true,  true)] = emask_avx512_flt_ns_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 0, false, true)] = emask_avx512_flt_ns_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 1, true,  true)] = emask_avx512_flt_ns_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 1, false, true)] = emask_avx512_flt_ns_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 2, true,  true)] = emask_avx512_flt_ns_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, false, 2, false, true)] = emask_avx512_flt_ns_custom_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  0, true,  true)] = emask_avx512_flt_sc_std_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  0, false, true)] = emask_avx512_flt_sc_std_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  1, true,  true)] = emask_avx512_flt_sc_sobel_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  1, false, true)] = emask_avx512_flt_sc_sobel_fast_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  2, true,  true)] = emask_avx512_flt_sc_custom_strict_dir;
    func[format("{}{}{}{}{}{}", a2s(USE_AVX512), 4, true,  2, false, true)] = emask_avx512_flt_sc_custom_fast_dir;

    bool scale = (mode & SCALE_MAGNITUDE);
    int opr = (mode & USE_STANDARD_OPERATOR) ? 0
        : (mode & USE_SOBEL_OPERATOR) ? 1 : 2;
    bool strict = (mode & STRICT_MAGNITUDE);
    bool dir = (mode & CALC_DIRECTION);
    if (dir) bytes = 4;

    auto key = format("{}{}{}{}{}{}", a2s(arch), bytes, scale, opr, strict, dir);
    return func.at(key);
}


write_direction_t get_write_dir(int bytes)
{
    switch (bytes) {
    case 1:
        return write_directions<uint8_t>;
    case 2:
        return write_directions<uint16_t>;
    default:
        return write_directions<float>;
    }
}


nms_t get_nms(arch_t arch)
{
    switch (arch) {
    case arch_t::NO_SIMD:
        return nms;
    case arch_t::USE_SSE4:
        return nms_sse4;
    case arch_t::USE_AVX2:
        return nms_avx2;
    default:
        return nms_avx512;
    }
}
