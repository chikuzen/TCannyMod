/*
  tcannymod.cpp

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


#include <malloc.h>
#include <math.h>
#include "tcannymod.hpp"


static const AVS_Linkage* AVS_linkage = 0;


static void __stdcall
set_gb_kernel(float sigma, int& radius, float* kernel)
{
    radius = max((int)(sigma * 3.0f + 0.5f), 1);
    int length = radius * 2 + 1;
    if (length > GB_MAX_LENGTH) {
        radius = 0;
        return;
    }

    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float weight = expf((-1 * i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = weight;
        sum += weight;
    }
    for (int i = 0; i < length; kernel[i++] /= sum);
}


TCannyM::TCannyM(PClip ch, int m, float sigma, float tmin, float tmax,
                 int c, bool sobel, float _scale, const char* n,
                 IScriptEnvironment* env)
: GenericVideoFilter(ch), mode(m), gb_radius(0), th_min(tmin), th_max(tmax),
 chroma(c), edge_mask(0), direction(0), hysteresiss_map(0), name(n),
 scale(_scale)
{
    if (!vi.IsPlanar()) {
        env->ThrowError("%s: Planar format only.", name);
    }

    if (vi.IsY8()) {
        chroma = 0;
    }

    if (vi.width > 65535 || vi.height > 65535) {
        env->ThrowError("%s: width/height must be smaller than 65536.", name);
    }

    if (sigma > 0.0f) {
        set_gb_kernel(sigma, gb_radius, gb_kernel);
        if (gb_radius == 0) {
            env->ThrowError("%s: sigma is too large.", name);
        }
    }

    buff_pitch = ((vi.width + 16 + 15) / 16) * 16;
    buff = (float*)_aligned_malloc(buff_pitch * sizeof(float) * 3, 16);

    frame_pitch = ((vi.width + 15) / 16) * 16;
    blur_frame = (float*)_aligned_malloc(frame_pitch * sizeof(float) * vi.height, 16);

    if (!blur_frame || !buff) {
        env->ThrowError("%s: failed to allocate temporal buffer.", name);
    }

    if (mode < 4) {
        edge_mask = (float*)_aligned_malloc(frame_pitch * sizeof(float) * vi.height, 16);
        direction = (uint8_t*)_aligned_malloc(frame_pitch * vi.height, 16);
        hysteresiss_map = (uint8_t*)malloc(vi.width * vi.height);
        if (!edge_mask || !direction || !hysteresiss_map) {
            env->ThrowError("TCannyMod: failed to allocate temporal buffer.");
        }
    }
    
    edge_detect = sobel ? &TCannyM::sobel_operator : &TCannyM::standerd_operator;
    write_gblur_frame = get_write_dst_frame(false);
    write_gradient_mask = get_write_dst_frame(scale != 1.0);
}


TCannyM::~TCannyM()
{
    _aligned_free(blur_frame);
    _aligned_free(buff);
    _aligned_free(edge_mask);
    _aligned_free(direction);
    free(hysteresiss_map);
}


PVideoFrame __stdcall TCannyM::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    for (int i = 0; i < 3; i++) {
        if (i > 0 && chroma == 0) {
            break;
        }

        int p = planes[i];
        int width = src->GetRowSize(p);
        int height = src->GetHeight(p);
        int src_pitch = src->GetPitch(p);
        const uint8_t* srcp = src->GetReadPtr(p);
        uint8_t *dstp = dst->GetWritePtr(p);
        int dst_pitch = dst->GetPitch(p);

        if (i > 0 && chroma > 1) {
            if (chroma == 2) {
                env->BitBlt(dstp, dst_pitch, srcp, src_pitch, width, height);
            } else {
                memset(dstp, 0x80, dst_pitch * height);
            }
            continue;
        }

        if ((intptr_t)srcp & 15) {
            env->ThrowError("%s: Invalid memory alignment", name);
        }

        gaussian_blur(srcp, src_pitch, width, height);
        if (mode == 4) {
            write_gblur_frame(blur_frame, dstp, width, height, dst_pitch, frame_pitch, 1.0);
            continue;
        }

        (this->*edge_detect)(width, height);
        if (mode == 1) {
            write_gradient_mask(edge_mask, dstp, width, height, dst_pitch, frame_pitch, scale);
            continue;
        }
        if (mode == 3) {
            env->BitBlt(dstp, dst_pitch, direction, frame_pitch, width, height);
            continue;
        }

        non_max_suppress(width, height);
        hysteresiss(width, height);
        if (mode == 2) {
            write_edge_direction(blur_frame, direction, th_max, width, height,
                                 frame_pitch, dstp, dst_pitch);
            continue;
        }
        write_binary_mask(blur_frame, th_max, width, height, frame_pitch,
                          dstp, dst_pitch);
    }

    return dst;
}

template <typename T>
static inline T minmax(T val, T min, T max)
{
    if (val < min) {
        val = min;
    }
    if (val > max) {
        val = max;
    }
    return val;
}

static AVSValue __cdecl
create_tcannymod(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    int mode = args[1].AsInt(0);
    if (mode < 0 || mode > 4) {
        env->ThrowError("TCannyMod: mode must be between 0 and 4.");
    }

    float sigma = (float)args[2].AsFloat(1.5);
    if (sigma < 0.0f) {
        env->ThrowError("TCannyMod: sigma must be greater than zero.");
    }

    float min = (float)args[4].AsFloat(0.1f);
    if (min < 0.0f) {
        env->ThrowError("TCannyMod: t_l must be greater than zero.");
    }

    float max = (float)args[3].AsFloat(8.0);
    if (max < min) {
        env->ThrowError("TCannyMod: t_h must be greater than t_l.");
    }

    int chroma = args[6].AsInt(0);
    if (chroma < 0 || chroma > 3) {
        env->ThrowError("TCannyMod: chroma must be set to 0, 1, 2 or 3.");
    }

    float scale = (float)(255.0 / minmax(args[7].AsFloat(255.0), 1.0, 255.0));

    return new TCannyM(args[0].AsClip(), mode, sigma, min, max, chroma,
                       args[5].AsBool(false), scale, "TCannyMod", env);
}


static AVSValue __cdecl
create_gblur(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    float sigma = (float)args[1].AsFloat(0.5);
    if (sigma < 0.0f) {
        env->ThrowError("GBlur: sigma must be greater than zero.");
    }
    int chroma = args[2].AsInt(1);
    if (chroma < 0 || chroma > 3) {
        env->ThrowError("GBlur: chroma must be set to 0, 1, 2 or 3.");
    }

    return new TCannyM(args[0].AsClip(), 4, sigma, 1.0f, 1.0f, chroma, false,
                       1.0f, "GBlur", env);
}


extern "C" __declspec(dllexport) const char * __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("TCannyMod",
             /*0*/   "c"
             /*1*/   "[mode]i"
             /*2*/   "[sigma]f"
             /*3*/   "[t_h]f"
             /*4*/   "[t_l]f"
             /*5*/   "[sobel]b"
             /*6*/   "[chroma]i"
             /*7*/   "[gmmax]f",
                     create_tcannymod, 0);
    env->AddFunction("GBlur", "c[sigma]f[chroma]i", create_gblur, 0);
    return "Canny edge detection filter for Avisynth2.6 ver." TCANNY_M_VERSION;
}
