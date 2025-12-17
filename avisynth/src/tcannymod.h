/*
  tcannymod.h

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


#ifndef TCANNY_MOD_H
#define TCANNY_MOD_H

#include <cstdint>
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#define NOGDI
#include <windows.h>
#include <avisynth.h>

#define TCANNY_M_VERSION "1.3.0"


typedef IScriptEnvironment ise_t;


typedef void(__stdcall *gaussian_blur_t)(
    const int radius, const float* kernel, float* buffp, float* blurp,
    const size_t blur_pitch, const uint8_t* srcp, const size_t src_pitch,
    const size_t width, const size_t height) noexcept;


typedef void(__stdcall *edge_detection_t)(
    float* blurp, const size_t blur_pitch, float* emaskp,
    const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
    const size_t width, const size_t height);


typedef void (__stdcall *non_max_suppress_t)(
    const float* emaskp, const size_t em_pitch, const int32_t* dirp,
    const size_t dir_pitch, float* blurp, const size_t blr_pitch,
    const size_t width, const size_t height);


typedef void(__stdcall *write_gradient_mask_t)(
    const float* srcp, uint8_t* dstp, const size_t width,
    const size_t height, const size_t dst_pitch, const size_t src_pitch,
    const float scale);


typedef void(__stdcall *write_gradient_direction_t)(
    const int32_t* dirp, uint8_t* dstp, const size_t dir_pitch,
    const size_t dst_pitch, const size_t width, const size_t height);


typedef void (__stdcall *write_edge_direction_t)(
    const int32_t* dirp, const uint8_t* hystp, uint8_t* dstp,
    const size_t dir_pitch, const size_t hyst_pitch, const size_t dst_pitch,
    const size_t width, const size_t height);


enum arch_t {
    HAS_SSE2,
    HAS_SSE41,
    HAS_AVX2,
    HAS_AVX512,
};


constexpr size_t GB_MAX_LENGTH = 17;


class Buffers {
    ise_t* env;
    bool isV8;
public:
    uint8_t* orig;
    float* buffp;
    float* blurp;
    float* emaskp;
    int32_t* dirp;
    uint8_t* hystp;
    Buffers(size_t bufsize, size_t blsize, size_t emsize, size_t dirsize,
            size_t hystsize, size_t align, bool ip, ise_t* e);
    ~Buffers();
};

class TCannyM : public GenericVideoFilter {
    const char* name;
    int numPlanes;
    size_t align;
    bool isV8;
    int mode;
    int chroma;
    float th_min;
    float th_max;
    float scale;
    bool calc_dir;
    int gbRadius; // max: 8
    float gbKernel[GB_MAX_LENGTH];
    //float* horizontalKernel;
    //float* verticalKernel;
    Buffers* buff;
    size_t blurPitch;
    size_t emaskPitch;
    size_t dirPitch;
    size_t hystPitch;
    size_t buffSize;
    size_t blurSize;
    size_t emaskSize;
    size_t dirSize;
    size_t hystSize;

    gaussian_blur_t gaussianBlur;
    edge_detection_t edgeDetection;
    non_max_suppress_t nonMaximumSuppression;
    write_gradient_mask_t writeBluredFrame;
    write_gradient_mask_t writeGradientMask;
    write_gradient_direction_t writeGradientDirection;
    write_edge_direction_t writeEdgeDirection;

public:
    TCannyM(PClip child, int mode, float sigma, float th_min, float th_max,
            int chroma, bool sobel, float scale, arch_t arch, const char* name,
            bool is_plus, bool use_cache);
    ~TCannyM();
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
    int __stdcall SetCacheHints(int hints, int)
    {
        if (isV8) {
            return hints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
        } else {
            return hints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
        }
    }
};


gaussian_blur_t get_gaussian_blur(bool use_cache, arch_t arch) noexcept;

edge_detection_t
get_edge_detection(bool use_sobel, bool calc_dir, bool use_cache, arch_t arch) noexcept;

non_max_suppress_t get_non_max_suppress(arch_t arch) noexcept;

write_gradient_mask_t
get_write_gradient_mask(bool scale, bool use_cache, arch_t arch) noexcept;

write_gradient_direction_t
get_write_gradient_direction(bool use_cache, arch_t arch) noexcept;

write_edge_direction_t
get_write_edge_direction(bool use_cache, arch_t arch) noexcept;

void __stdcall
hysteresis(uint8_t* hystp, const size_t hpitch, float* blurp,
    const size_t bpitch, const int width, const int height,
    const float tmin, const float tmax) noexcept;

#endif
