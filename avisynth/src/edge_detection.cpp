/*
  edge_detection.cpp

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

constexpr float tangent_pi_x1_8 = 0.414213538169860839843750f;
constexpr float tangent_pi_x3_8 = 2.414213657379150390625000f;
constexpr float tangent_pi_x5_8 = -2.414213657379150390625000f;
constexpr float tangent_pi_x7_8 = -0.414213538169860839843750f;


template <typename Vf, typename Vi, bool USE_CACHE>
SFINLINE void
calc_direction(const Vf& gx, const Vf& gy, int32_t* dstp, const Vf& t0225,
    const Vf& t0675, const Vf& t1125, const Vf& t1575) noexcept
{
    const Vf z = zero<Vf>();
    const Vf vertical = set1<Vf, float>(90.0f);
    // if gy < 0, gx = -gx
    Vf mask = cmplt_32(gy, z);
    Vf gx2 = blendv(gx, sub(z, gx), mask);
    // tan = gy / gx
    Vf tan = mul(rcp_hq(gx2), abs(gy));
    // if tan is unorderd(inf or NaN), tan = 90.0f
    mask = cmpord_ps(tan, tan);
    tan = blendv(vertical, tan, mask);
    // if t1575 <= tan < t0225, direction is 31 (horizontal)
    Vi d0 = cast<Vi, Vf>(_and(cmpge_32(tan, t1575), cmplt_32(tan, t0225)));
    d0 = srli_i32(d0, 27);
    // if t0225 <= tan < t0675, direction is 63 (45' up)
    Vi d1 = cast<Vi, Vf>(_and(cmpge_32(tan, t0225), cmplt_32(tan, t0675)));
    d1 = srli_i32(d1, 26);
    // if t0675 <= tan or tan < t1125, direction is 127 (vertical)
    Vi d2 = cast<Vi, Vf>(_or(cmpge_32(tan, t0675), cmplt_32(tan, t1125)));
    d2 = srli_i32(d2, 25);
    // if t1125 <= tan < t1575, direction is 255 (45' down)
    Vi d3 = cast<Vi, Vf>(_and(cmpge_32(tan, t1125), cmplt_32(tan, t1575)));
    d3 = srli_i32(d3, 24);
    d0 = _or(_or(d0, d1), _or(d2, d3));
    if constexpr (USE_CACHE) {
        store(dstp, d0);
    } else {
        stream(dstp, d0);
    }

}

template <typename Vf, typename Vi, bool CALC_DIRECTION, bool USE_CACHE>
static void __stdcall
standard(float* blurp, const size_t blur_pitch, float* emaskp,
         const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
         const size_t width, const size_t height) noexcept
{

    constexpr size_t step = sizeof(Vf) / sizeof(float);

    float* p0 = blurp;
    float* p1 = blurp;
    float* p2 = blurp + blur_pitch;

    const Vf t0225 = set1<Vf>(tangent_pi_x1_8);
    const Vf t0675 = set1<Vf>(tangent_pi_x3_8);
    const Vf t1125 = set1<Vf>(tangent_pi_x5_8);
    const Vf t1575 = set1<Vf>(tangent_pi_x7_8);

    for (size_t y = 0; y < height; y++) {
        p1[-1] = p1[0];
        p1[width] = p1[width - 1];

        for (size_t x = 0; x < width; x += step) {
            Vf gy = sub(load<Vf>(p0 + x), load<Vf>(p2 + x)); // [1, 0, -1]
            Vf gx = sub(loadu<Vf>(p1 + x + 1), loadu<Vf>(p1 + x - 1)); // [-1, 0, 1]

            if constexpr (CALC_DIRECTION) {
                calc_direction<Vf, Vi, USE_CACHE>(gx, gy, dirp + x, t0225,
                    t0675, t1125, t1575);
            }

            Vf magnitude = mul(gx, gx);
            magnitude = madd(gy, gy, magnitude);
            magnitude = sqrt(magnitude);
            if constexpr (USE_CACHE) {
                store(emaskp + x, magnitude);
            } else {
                stream(emaskp + x, magnitude);
            }

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

    H = [-1,  0,  1,    -> p0
         -2,  0,  2,    -> p1
         -1,  0,  1]    -> p2
    V = [ 1,  2,  1,    -> p0
          0,  0,  0,    -> p1
         -1, -2, -1]    -> p2
*/
template <typename Vf, typename Vi, bool CALC_DIRECTION, bool USE_CACHE>
static void __stdcall
sobel(float* blurp, const size_t blur_pitch, float* emaskp,
      const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
      const size_t width, const size_t height) noexcept
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    float* p0 = blurp;
    float* p1 = blurp;
    float* p2 = blurp + blur_pitch;

    p1[-1] = p1[0];
    p1[width] = p1[width - 1];

    const Vf t0225 = set1<Vf>(tangent_pi_x1_8);
    const Vf t0675 = set1<Vf>(tangent_pi_x3_8);
    const Vf t1125 = set1<Vf>(tangent_pi_x5_8);
    const Vf t1575 = set1<Vf>(tangent_pi_x7_8);

    for (size_t y = 0; y < height; y++) {
        p2[-1] = p2[0];
        p2[width] = p2[width - 1];

        for (size_t x = 0; x < width; x += step) {
            Vf t = loadu<Vf>(p0 + x + 1);
            Vf gx = t;
            t = loadu<Vf>(p2 + x - 1);
            gx = sub(gx, t);
            Vf gy = gx;
            t = loadu<Vf>(p0 + x - 1);
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

            if constexpr (CALC_DIRECTION) {
                calc_direction<Vf, Vi, USE_CACHE>(gx, gy, dirp, t0225, t0675,
                    t1125, t1575);
            }

            Vf magnitude = mul(gx, gx);
            magnitude = madd(gy, gy, magnitude);
            magnitude = sqrt(magnitude);
            if constexpr (USE_CACHE) {
                store(emaskp + x, magnitude);
            } else {
                stream(emaskp + x, magnitude);
            }

        }
        emaskp += emask_pitch;
        dirp += dir_pitch;
        p0 = p1;
        p1 = p2;
        p2 += y < height - 1 ? blur_pitch : 0;
    }
}


template <typename Vf, typename Vi>
static void __stdcall
non_max_suppress(const float* emaskp, const size_t em_pitch,
                 const int32_t* dirp, const size_t dir_pitch, float* blurp,
                 const size_t blur_pitch, const size_t width,
                 const size_t height) noexcept
{
    constexpr size_t step = sizeof(Vf) / sizeof(float);

    static const Vf FLT_MAX_neg = set1<Vf, float>(-FLT_MAX);

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
            const Vi m135 = srli_i32(cmpeq_32(dir, dir), 24);
            Vf mask = cast<Vf, Vi>(cmpeq_32(dir, m135));
            Vf temp = max(loadu<Vf>(emaskp + x - 1 - em_pitch),
                          loadu<Vf>(emaskp + x + 1 + em_pitch));
            Vf p0 = _and(temp, mask);

            mask = cast<Vf, Vi>(cmpeq_32(dir, srli_i32(m135, 1)));
            temp = max(loadu<Vf>(emaskp + x - em_pitch),
                       loadu<Vf>(emaskp + x + em_pitch));
            p0 = _or(p0, _and(temp, mask));

            mask = cast<Vf, Vi>(cmpeq_32(dir, srli_i32(m135, 2)));
            temp = max(loadu<Vf>(emaskp + x + 1 - em_pitch),
                       loadu<Vf>(emaskp + x - 1 + em_pitch));
            p0 = _or(p0, _and(temp, mask));

            mask = cast<Vf, Vi>(cmpeq_32(dir, srli_i32(m135, 3)));
            temp = max(loadu<Vf>(emaskp + x - 1),
                       loadu<Vf>(emaskp + x + 1));
            p0 = _or(p0, _and(temp, mask));

            mask = cmplt_32(loadu<Vf>(emaskp + x), p0);

            Vf blur = loadu<Vf>(blurp + x);
            blur = blendv(blur, FLT_MAX_neg, mask);
            storeu(blurp + x, blur);
        }
        blurp[width - 1] = blurp[width] = -FLT_MAX;
        emaskp += em_pitch;
        blurp += blur_pitch;
    }
}


edge_detection_t
get_edge_detection(bool use_sobel, bool calc_dir, bool use_cache, arch_t arch) noexcept
{
    using std::format;
    std::unordered_map<std::string, edge_detection_t> func;
    int a = arch == HAS_SSE41 ? 1 : 2;

    func[format("{}{}{}{}", false, false, false, 1)] = standard<__m128, __m128i, false, false>;
    func[format("{}{}{}{}", false, true, false, 1)] = standard<__m128, __m128i, true, false>;
    func[format("{}{}{}{}", true, false, false, 1)] = sobel<__m128, __m128i, false, false>;
    func[format("{}{}{}{}", true, true, false, 1)] = sobel<__m128, __m128i, true, false>;
    func[format("{}{}{}{}", false, false, true, 1)] = standard<__m128, __m128i, false, true>;
    func[format("{}{}{}{}", false, true, true, 1)] = standard<__m128, __m128i, true, true>;
    func[format("{}{}{}{}", true, false, true, 1)] = sobel<__m128, __m128i, false, true>;
    func[format("{}{}{}{}", true, true, true, 1)] = sobel<__m128, __m128i, true, true>;
    func[format("{}{}{}{}", false, false, false, 2)] = standard<__m256, __m256i, false, false>;
    func[format("{}{}{}{}", false, true, false, 2)] = standard<__m256, __m256i, true, false>;
    func[format("{}{}{}{}", true, false, false, 2)] = sobel<__m256, __m256i, false, false>;
    func[format("{}{}{}{}", true, true, false, 2)] = sobel<__m256, __m256i, true, false>;
    func[format("{}{}{}{}", false, false, true, 2)] = standard<__m256, __m256i, false, true>;
    func[format("{}{}{}{}", false, true, true, 2)] = standard<__m256, __m256i, true, true>;
    func[format("{}{}{}{}", true, false, true, 2)] = sobel<__m256, __m256i, false, true>;
    func[format("{}{}{}{}", true, true, true, 2)] = sobel<__m256, __m256i, true, true>;

    return func[format("{}{}{}{}", use_sobel, calc_dir, use_cache, a)];
}


non_max_suppress_t get_non_max_suppress(arch_t arch) noexcept
{
    if (arch == HAS_AVX2) {
        return non_max_suppress<__m256, __m256i>;
    }
    return non_max_suppress<__m128, __m128i>;
}


