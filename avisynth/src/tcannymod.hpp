/*
  tcannymod.hpp

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


#ifndef TCANNY_MOD_HPP
#define TCANNY_MOD_HPP

#include <stdint.h>
#include <windows.h>
#include "avisynth.h"

#define TCANNY_M_VERSION "0.1.1"

#define GB_MAX_LENGTH 17


class TCannyM : public GenericVideoFilter {
    const char* name;
    int mode;
    int chroma;
    float th_min;
    float th_max;
    int gb_radius; // max: 8
    float gb_kernel[GB_MAX_LENGTH];
    float *buff;
    int buff_pitch;
    float *blur_frame;
    float *edge_mask;
    uint8_t *direction;
    int frame_pitch;
    uint8_t* hysteresiss_map;

    void __stdcall gaussian_blur(const uint8_t* srcp, int src_pitch, int width,
                                 int height);
    void __stdcall standerd_operator(int width, int height);
    void __stdcall sobel_operator(int width, int height);
    void __stdcall non_max_suppress(int width, int height);
    void __stdcall hysteresiss(int width, int height);
    void __stdcall write_dst_frame(const float* srcp, uint8_t* dstp, int width,
                                   int height, int dst_pitch);
    void __stdcall write_edge_direction(int width, int height, uint8_t* dstp,
                                        int dst_pitch);
    void __stdcall write_binary_mask(int width, int height, uint8_t* dstp,
                                     int dst_pitch);
    void (__stdcall TCannyM::*edge_detect)(int width, int height);

public:
    TCannyM(PClip child, int mode, float sigma, float th_min, float th_max,
            int chroma, bool sobel, const char* name, IScriptEnvironment* env);
    ~TCannyM();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};

#endif
