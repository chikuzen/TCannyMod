/*
  gaussian_blur.cpp

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
#include <algorithm>
#include <unordered_map>
#include <format>
#include "tcannymod.h"
#include "simd.h"


template <typename Vf, bool USE_CACHE>
static void __stdcall
convert_to_float(const size_t width, const size_t height, const uint8_t* srcp,
                 const int src_pitch, float* blurp, const size_t blur_pitch) noexcept
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x += step) {
            Vf val = cvtu8_ps<Vf>(srcp + x);
            if constexpr (USE_CACHE) {
                store(blurp + x, val);
            } else {
                stream(blurp + x, val);
            }
        }
        srcp += src_pitch;
        blurp += blur_pitch;
    }
}

template <typename Vf, bool USE_CACHE>
static void
horizontal_blur(const float* kernel, float* gbtp, const int radius,
    const size_t width, float* blurp) noexcept
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);
    const int length = radius * 2 + 1;

    for (int i = 1; i <= radius; ++i) {
        gbtp[-i] = gbtp[i - 1];
        gbtp[width - 1 + i] = gbtp[width - i];
    }

    for (size_t x = 0; x < width; x += step) {
        Vf sum = zero<Vf>();
        for (int i = -radius; i <= radius; ++i) {
            Vf k = set1<Vf, float>(kernel[i + radius]);
            Vf val = loadu<Vf>(gbtp + x + i);
            sum = madd(k, val, sum);
        }
        if (USE_CACHE) {
            store(blurp + x, sum);
        } else {
            stream(blurp + x, sum);
        }
    }
}


template <typename Vf, int RADIUS, bool USE_CACHE>
static void __stdcall
gaussian_blur(const int radius, const float* kernel, float* gbtp,
    const size_t gbt_pitch, float* blurp, const size_t blur_pitch,
    const uint8_t* srcp, const size_t src_pitch, const size_t width,
    const size_t height)
{
    if constexpr (RADIUS == 0) {
        convert_to_float<Vf, USE_CACHE>(
                width, height, srcp, src_pitch, blurp, blur_pitch);
        return;
    } else {

        constexpr size_t align = sizeof(Vf);
        constexpr size_t step = align / sizeof(float);
        const int length = radius * 2 + 1;

        const uint8_t* p[GB_MAX_LENGTH + 3];
        p[radius] = srcp;
        for (int r = 1; r <= radius; ++r) {
            p[radius + r] = srcp + r * src_pitch;
            p[radius - r] = p[radius + r - 1];
        }
        p[length] = srcp + src_pitch * (radius + 1);
        p[length + 1] = srcp + src_pitch * (radius + 2);
        p[length + 2] = srcp + src_pitch * (radius + 3);

        for (size_t y = 0; y < height; y += RADIUS + 2) {
            for (size_t x = 0; x < width; x += step) {
                Vf input = cvtu8_ps<Vf>(p[0] + x);
                Vf k0 = set1<Vf, float>(kernel[0]);
                Vf sum0 = mul(input, k0);

                input = cvtu8_ps<Vf>(p[1] + x);
                Vf k1 = set1<Vf, float>(kernel[1]);
                sum0 = madd(input, k1, sum0);
                Vf sum1 = mul(input, k0);

                input = cvtu8_ps<Vf>(p[2] + x);
                Vf k2 = set1<Vf, float>(kernel[2]);
                sum0 = madd(input, k2, sum0);
                sum1 = madd(input, k1, sum1);
                Vf sum2 = mul(input, k0);

                if constexpr (RADIUS == 1) {
                    input = cvtu8_ps<Vf>(p[3] + x);
                    sum1 = madd(input, k2, sum1);
                    sum2 = madd(input, k1, sum2);
                    input = cvtu8_ps<Vf>(p[4] + x);
                    sum2 = madd(input, k2, sum2);
                    store(gbtp + x, sum0);
                    store(gbtp + x + gbt_pitch, sum1);
                    store(gbtp + x + gbt_pitch * 2, sum2);

                } else {
                    input = cvtu8_ps<Vf>(p[3] + x);
                    Vf k3 = set1<Vf, float>(kernel[3]);
                    sum0 = madd(input, k3, sum0);
                    sum1 = madd(input, k2, sum1);
                    sum2 = madd(input, k1, sum2);
                    Vf sum3 = mul(input, k0);

                    for (int l = 4; l < length; ++l) {
                        k0 = k1;
                        k1 = k2;
                        k2 = k3;
                        k3 = set1<Vf, float>(kernel[l]);
                        input = cvtu8_ps<Vf>(p[l] + x);
                        sum0 = madd(input, k3, sum0);
                        sum1 = madd(input, k2, sum1);
                        sum2 = madd(input, k1, sum2);
                        sum3 = madd(input, k0, sum3);
                    }
                    input = cvtu8_ps<Vf>(p[length] + x);
                    sum1 = madd(input, k3, sum1);
                    sum2 = madd(input, k2, sum2);
                    sum3 = madd(input, k1, sum3);
                    input = cvtu8_ps<Vf>(p[length + 1] + x);
                    sum2 = madd(input, k3, sum2);
                    sum3 = madd(input, k2, sum3);
                    input = cvtu8_ps<Vf>(p[length + 2] + x);
                    sum3 = madd(input, k3, sum3);
                    store(gbtp + x, sum0);
                    store(gbtp + x + gbt_pitch, sum1);
                    store(gbtp + x + gbt_pitch * 2, sum2);
                    store(gbtp + x + gbt_pitch * 3, sum3);
                }
            }
            horizontal_blur<Vf, USE_CACHE>(kernel, gbtp, radius, width, blurp);
            horizontal_blur<Vf, USE_CACHE>(kernel, gbtp + gbt_pitch, radius,
                width, blurp + blur_pitch);
            horizontal_blur<Vf, USE_CACHE>(kernel, gbtp + 2 * gbt_pitch, radius,
                width, blurp + 2 * blur_pitch);
            if (RADIUS > 1) {
                horizontal_blur<Vf, USE_CACHE>(kernel, gbtp + 3 * gbt_pitch, radius,
                    width, blurp + 3 * blur_pitch);
            }
            blurp += blur_pitch * (RADIUS + 2);
            for (int i = 0; i < RADIUS + 2; ++i) {
                for (int l = 0; l < length + 2; ++l) {
                    p[l] = p[l + 1];
                }
                if (y < height - 1 - radius) {
                    p[length + 2] += src_pitch;
                } else if (y > height - 1 - radius) {
                    p[length + 2] -= src_pitch;
                }
            }
        }
    }
}


gaussian_blur_t get_gaussian_blur(int radius, bool use_cache, arch_t arch) noexcept
{
    using std::format;

    int R = std::min(radius, 2);
    int A = arch == HAS_SSE41 ? 0 : 1;

    std::unordered_map<std::string, gaussian_blur_t> func;
    func[format("{}{}{}", 0, false, 0)] = gaussian_blur<__m128, 0, false>;
    func[format("{}{}{}", 1, false, 0)] = gaussian_blur<__m128, 1, false>;
    func[format("{}{}{}", 2, false, 0)] = gaussian_blur<__m128, 2, false>;
    func[format("{}{}{}", 0, true, 0)] = gaussian_blur<__m128, 0, true>;
    func[format("{}{}{}", 1, true, 0)] = gaussian_blur<__m128, 1, true>;
    func[format("{}{}{}", 2, true, 0)] = gaussian_blur<__m128, 2, true>;
    func[format("{}{}{}", 0, false, 1)] = gaussian_blur<__m256, 0, false>;
    func[format("{}{}{}", 1, false, 1)] = gaussian_blur<__m256, 1, false>;
    func[format("{}{}{}", 2, false, 1)] = gaussian_blur<__m256, 2, false>;
    func[format("{}{}{}", 0, true, 1)] = gaussian_blur<__m256, 0, true>;
    func[format("{}{}{}", 1, true, 1)] = gaussian_blur<__m256, 1, true>;
    func[format("{}{}{}", 2, true, 1)] = gaussian_blur<__m256, 2, true>;

    return func[format("{}{}{}", R, use_cache, A)];
}
