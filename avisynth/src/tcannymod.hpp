/*
  tcannymod.hpp

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

#ifndef TCANNY_M_HPP
#define TCANNY_M_HPP

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <array>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#define NOGDI
#include <windows.h>
#include <avisynth.h>
#include <avs/alignment.h>
#else
#include <avisynth/avisynth.h>
#include <avisynth/avs/alignment.h>
#endif


#define TCANNY_M_VERSION "2.0.0"

enum arch_t : int32_t {
    NO_SIMD,
    USE_SSE4,
    USE_AVX2,
    USE_AVX512,
};

enum mode_t : int32_t {
    AT_LEAST_V8 = 1 << 0,
    DO_BLUR_ONLY = 1 << 1,
    DO_NOT_BLUR = 1 << 2,
    DETECT_EDGE = 1 << 3,
    CALC_DIRECTION = 1 << 4,
    SHOW_DIRECTION = 1 << 5,
    GENERATE_CANNY_IMAGE = 1 << 6,
    USE_STANDARD_OPERATOR = 1 << 7,
    USE_SOBEL_OPERATOR = 1 << 8,
    USE_CUSTOM_OPERATOR = 1 << 9,
    STRICT_MAGNITUDE = 1 << 10,
    SCALE_MAGNITUDE = 1 << 11,
    DO_NOT_TOUCH_CHROMA = 1 << 12,
    PROC_CHROMA = 1 << 13,
    COPY_CHROMA = 1 << 14,
    FILL_HALF_CHROMA = 1 << 15,
    FILL_ZERO_CHROMA = 1 << 16,
    SET_DEBUG_INFO = 1 << 17,
};

using ise_t = IScriptEnvironment;

using operator_t = std::array<float, 3>;

using gblur_t = void(*)(
    const void* srcp, int sstride, float* hbuffp, int hbpitch, void* dstp,
    int dstride, int width, int height, int radius, const float* weights,
    const float maxval);

using edgemask_t = void(*)(
    const float* blurp, int blpitch, void* dstp, int dpitch, operator_t& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

using write_direction_t = void(*)(
    const int32_t* dirp, int dirpitch, void* dstp, int dpitch, int width,
    int height);

using nms_t = void(*)(
    float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height);

using hysteresis_t = void(*)(
    void* dstp, const int dpitch, float* emaskp, const int epitch,
    const int width, const int height, const float tmin, const float tmax,
    const float maxval);


struct Buffer;

class TCannyMod : public GenericVideoFilter {
    int mode;
    float tmin;
    float tmax;
    float scale;
    arch_t arch;
    int align;

    int numPlanes;
    int bits;
    int bytes;
    float maxval;
    int radius;
    std::vector<float> gbweights;
    operator_t opr;
    std::vector<double> dbgweights;
    std::string opt;

    int hbPitch;
    int hbPad;
    int blPitch;
    int emPitch;
    int dirPitch;
    size_t hbSize;
    size_t blSize;
    size_t emSize;
    size_t dirSize;

    gblur_t gaussianBlur;
    edgemask_t edgeMask;
    write_direction_t writeDirections;
    nms_t nonMaximumSuppression;
    hysteresis_t hysteresis;

    void generateWeights(float sigma);
    void mainLoop(PVideoFrame& src, PVideoFrame& dst, Buffer& b, ise_t* env);
    PVideoFrame getFrameDebug(int n, ise_t* env);

public:
    TCannyMod(PClip c, float _tmin, float _tmax, float _scale, operator_t& opr,
        float sigma, int mode, arch_t arch);
    ~TCannyMod(){}
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    int __stdcall SetCacheHints(int hints, int)
    {
        return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};

template <typename T>
static inline void validate(bool cond, const T message)
{
    if (cond)
        throw std::runtime_error(message);
}

constexpr auto a2s(arch_t arch)
{
    if (arch == arch_t::NO_SIMD) return "NO_SIMD";
    if (arch == arch_t::USE_SSE4) return "SSE4";
    if (arch == arch_t::USE_AVX2)  return "AVX2";
    return "AVX512";
}



gblur_t get_gblur(int bytes, arch_t arch, int radius, int mode);

edgemask_t get_emask(int bytes, arch_t arch, int mode);

write_direction_t get_write_dir(int bytes);

nms_t get_nms(arch_t arch);

hysteresis_t get_hysteresis(int bytes);


#endif // TCANNY_M_HPP
