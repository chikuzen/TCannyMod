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
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
#endif
#include <windows.h>
#include <avisynth.h>

#define TCANNY_M_VERSION "1.2.0"

constexpr size_t GB_MAX_LENGTH = 17;

typedef IScriptEnvironment ise_t;


using gaussian_blur_t = void(__stdcall *)(
    const int radius, const float* kernel, const float* hkernel, float* buffp,
    float* blurp, const size_t blur_pitch, const uint8_t* srcp,
    const size_t src_pitch, const size_t width, const size_t height);


using edge_detection_t = void(__stdcall *)(
    float* blurp, const size_t blur_pitch, float* emaskp,
    const size_t emask_pitch, int32_t* dirp, const size_t dir_pitch,
    const size_t width, const size_t height);


using non_max_suppress_t = void (__stdcall *)(
    const float* emaskp, const size_t em_pitch, const int32_t* dirp,
    const size_t dir_pitch, float* blurp, const size_t blr_pitch,
    const size_t width, const size_t height);


using write_gradient_mask_t = void(__stdcall *)(
    const float* srcp, uint8_t* dstp, const size_t width,
    const size_t height, const size_t dst_pitch, const size_t src_pitch,
    const float scale);


using write_gradient_direction_t = void(__stdcall *)(
    const int32_t* dirp, uint8_t* dstp, const size_t dir_pitch,
    const size_t dst_pitch, const size_t width, const size_t height);


using write_edge_direction_t = void (__stdcall *)(
    const int32_t* dirp, const uint8_t* hystp, uint8_t* dstp,
    const size_t dir_pitch, const size_t hyst_pitch, const size_t dst_pitch,
    const size_t width, const size_t height);


class Buffers {
    ise_t* env;
    bool isPlus;
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
    bool isPlus;
    int mode;
    int chroma;
    float th_min;
    float th_max;
    float scale;
    bool calc_dir;
    int gbRadius; // max: 8
    float gbKernel[GB_MAX_LENGTH];
    float* horizontalKernel;
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
            int chroma, bool sobel, float scale, int opt, const char* name,
            bool is_plus);
    ~TCannyM();
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
};

extern bool has_sse2();
extern bool has_sse41();
extern bool has_avx();
extern bool has_avx2();
#endif
