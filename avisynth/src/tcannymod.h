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
#include "gaussian_blur.h"
#include "edge_detection.h"
#include "write_frame.h"


#define TCANNY_M_VERSION "1.0.0"

constexpr size_t GB_MAX_LENGTH = 17;

typedef IScriptEnvironment ise_t;


class TCannyM : public GenericVideoFilter {
    const char* name;
    int numPlanes;
    size_t align;
    int mode;
    int chroma;
    float th_min;
    float th_max;
    float scale;
    bool calc_dir;
    int gbRadius; // max: 8
    float gbKernel[GB_MAX_LENGTH];
    size_t blurPitch;
    size_t emaskPitch;
    size_t dirHystPitch;
    size_t buffSize;
    size_t blurSize;
    size_t emaskSize;
    size_t dirSize;
    size_t hystSize;

    gaussian_blur_t gaussianBlur;
    edge_detection_t edgeDetection;
    write_gradient_mask_t writeBluredFrame;
    write_gradient_mask_t writeGradientMask;
    write_direction_map_t writeDirectionMap;

public:
    TCannyM(PClip child, int mode, float sigma, float th_min, float th_max,
            int chroma, bool sobel, float scale, int opt, const char* name,
            ise_t* env);
    ~TCannyM() {}
    PVideoFrame __stdcall GetFrame(int n, ise_t* env);
};

extern int has_sse2();
extern int has_sse41();
extern int has_avx();
extern int has_avx2();
#endif
