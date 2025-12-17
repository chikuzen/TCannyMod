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


#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "tcannymod.h"
#include <avs/alignment.h>



template <typename T>
static inline T
my_malloc(size_t size, size_t align, bool is_v8, AvsAllocType at,
          ise_t* env) noexcept
{
    void* p = is_v8 ? env->Allocate(size, align, at) : avs_malloc(size, align);
    return reinterpret_cast<T>(p);
}


static inline void my_free(void* p, bool is_v8, ise_t* env) noexcept
{
    if (is_v8) {
        env->Free(p);
    } else {
        avs_free(p);
    }
    p = nullptr;
}


Buffers::Buffers(size_t bufsize, size_t blsize, size_t emsize, size_t dirsize,
    size_t hystsize, size_t align, bool ip, ise_t* e) :
    env(e), isV8(ip)
{
    size_t total_size = bufsize + blsize + emsize + dirsize + hystsize;
    orig = my_malloc<uint8_t*>(
        total_size, align, isV8, AVS_POOLED_ALLOC, env);

    memset(orig, 0, total_size);
    buffp = reinterpret_cast<float*>(orig) + 8;
    blurp = reinterpret_cast<float*>(orig + bufsize + align);
    emaskp = reinterpret_cast<float*>(orig + bufsize + blsize);
    dirp = reinterpret_cast<int32_t*>(orig + bufsize + blsize + emsize);
    hystp = orig + total_size - hystsize;
};


Buffers::~Buffers()
{
    my_free(orig, isV8, env);
    env = nullptr;
};


static inline void validate(bool cond, const char* msg)
{
    if (cond)
        throw std::runtime_error(msg);
}


static void __stdcall
set_gb_kernel(float sigma, int& radius, float* kernel)
{
    radius = std::max(static_cast<int>(sigma * 3.0f + 0.5f), 1);
    int length = radius * 2 + 1;
    validate(length > GB_MAX_LENGTH, "sigma is too large.");

    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float weight = std::exp((-1 * i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = weight;
        sum += weight;
    }
    for (int i = 0; i < length; kernel[i++] /= sum);
}


TCannyM::TCannyM(PClip ch, int m, float sigma, float tmin, float tmax, int c,
                 bool sobel, float s, arch_t arch, const char* n, bool is_v8) :
    GenericVideoFilter(ch), mode(m), gbRadius(0), th_min(tmin), th_max(tmax),
    chroma(c), name(n), scale(s), isV8(is_v8), buff(nullptr), align(32),
    calc_dir(true)
{
    validate(!vi.IsPlanar(), "Planar format only.");
    memset(gbKernel, 0, sizeof(gbKernel));

    numPlanes = (vi.IsY8() || chroma == 0) ? 1 : 3;

    if (sigma > 0.0f) {
        set_gb_kernel(sigma, gbRadius, gbKernel);

        size_t length = (gbRadius * 2 + 1);
    }

    blurPitch = ((align + (vi.width + 1) * sizeof(float)) + align - 1) & ~(align - 1);
    emaskPitch = (vi.width * sizeof(float) + align - 1) & ~(align - 1);
    dirPitch = (vi.width * sizeof(int32_t) + align - 1) & ~(align - 1);
    hystPitch = (vi.width + align - 1) & ~(align - 1);

    buffSize = ((8 + vi.width + 8) * sizeof(float) + align - 1) & ~(align - 1);
    blurSize = blurPitch * (vi.height + 1);
    emaskSize = mode == 4 ? 0 : emaskPitch * (vi.height + 1);
    dirSize = (mode == 1 || mode == 4) ? 0 : dirPitch * (vi.height + 1);
    hystSize = (mode == 0 || mode == 2) ? hystPitch * vi.height : 0;

    blurPitch /= sizeof(float);
    emaskPitch /= sizeof(float);
    dirPitch /= sizeof(int32_t);

    gaussianBlur = get_gaussian_blur(arch);

    edgeDetection = get_edge_detection(sobel, (mode != 1 && mode != 4), arch);

    nonMaximumSuppression = get_non_max_suppress(arch);

    writeBluredFrame = get_write_gradient_mask(false, arch);

    writeGradientMask = get_write_gradient_mask(scale != 1.0f, arch);

    writeGradientDirection = get_write_gradient_direction(arch);

    writeEdgeDirection = get_write_edge_direction(arch);

    if (!isV8) {
        buff = new Buffers(buffSize, blurSize, emaskSize, dirSize, hystSize,
                           align, false, nullptr);
    }
}


TCannyM::~TCannyM()
{
    if (!isV8) {
        delete buff;
    }
}


PVideoFrame __stdcall TCannyM::GetFrame(int n, ise_t* env)
{
    PVideoFrame src = child->GetFrame(n, env);

    PVideoFrame dst;
    Buffers* b = buff;

    if (isV8) {
        b = new Buffers(buffSize, blurSize, emaskSize, dirSize, hystSize, align,
                        true, env);
        if (!b || !b->orig) {
            env->ThrowError("%s: failed to allocate buffer.", name);
        }
        dst = env->NewVideoFrameP(vi, &src);
    } else {
        dst = env->NewVideoFrame(vi, align);
    }

    static const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int i = 0; i < numPlanes; i++) {

        const int p = planes[i];
        const int width = src->GetRowSize(p);
        const int height = src->GetHeight(p);
        const int src_pitch = src->GetPitch(p);
        const uint8_t* srcp = src->GetReadPtr(p);
        uint8_t *dstp = dst->GetWritePtr(p);
        const int dst_pitch = dst->GetPitch(p);

        if (i > 0 && chroma > 1) {
            if (chroma == 2) {
                env->BitBlt(dstp, dst_pitch, srcp, src_pitch, width, height);
            } else {
                memset(dstp, chroma == 3 ? 0x80 : 0x00, dst_pitch * height);
            }
            continue;
        }

        gaussianBlur(gbRadius, gbKernel, b->buffp, b->blurp, blurPitch, srcp,
            src_pitch, width, height);
        if (mode == 4) {
            writeBluredFrame(b->blurp, dstp, width, height, dst_pitch,
                             blurPitch, 1.0);
            continue;
        }

        edgeDetection(b->blurp, blurPitch, b->emaskp, emaskPitch, b->dirp,
                      dirPitch, width, height);

        if (mode == 1) {
            writeGradientMask(b->emaskp, dstp, width, height, dst_pitch,
                              emaskPitch, scale);
            continue;
        }
        if (mode == 3) {
            writeGradientDirection(b->dirp, dstp, dirPitch, dst_pitch, width,
                                   height);
            continue;
        }

        nonMaximumSuppression(b->emaskp, emaskPitch, b->dirp, dirPitch, b->blurp,
                              blurPitch, width, height);

        hysteresis(b->hystp, hystPitch, b->blurp, blurPitch, width, height,
                   th_min, th_max);

        if (mode == 2) {
            writeEdgeDirection(b->dirp, b->hystp, dstp, dirPitch, hystPitch,
                               dst_pitch, width, height);
            continue;
        }

        env->BitBlt(dstp, dst_pitch, b->hystp, hystPitch, width, height);
    }

    if (isV8) {
        delete b;
    }

    return dst;
}

static inline bool has_avx(ise_t* env) noexcept
{
    auto f = env->GetCPUFlags();
    return f & ::CPUF_AVX;
}


static arch_t get_arch(int opt, ise_t* env) noexcept
{
    auto f = env->GetCPUFlags();
    if (opt == 0 || opt == 1) {
        return HAS_SSE41;
    }
    if (f & (CPUF_AVX2 | CPUF_FMA3))
        return HAS_AVX2;
    else
        return HAS_SSE41;
}


static float calc_scale(double gmmax) noexcept
{
    return static_cast<float>(255.0 / std::min(std::max(gmmax, 1.0), 255.0));
}


static bool is_v8orgreater(ise_t* env)
{
    try {
        env->CheckVersion(8);
        return true;
    } catch (const AvisynthError&) {
        return false;
    }
}

static AVSValue __cdecl
create_tcannymod(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_avx(env), "This filter requires AVX.");

        int mode = args[1].AsInt(0);
        validate(mode < 0 || mode > 4, "mode must be between 0 and 4.");

        float sigma = static_cast<float>(args[2].AsFloat(1.5f));
        validate(sigma < 0.0f, "sigma must be greater than zero.");

        float tmin = static_cast<float>(args[4].AsFloat(0.1f));
        validate(tmin < 0.0f, "t_l must be greater than zero.");

        float tmax = static_cast<float>(args[3].AsFloat(8.0f));
        validate(tmax < tmin, "t_h must be greater than t_l.");

        int chroma = args[6].AsInt(0);
        validate(chroma < 0 || chroma > 4,
                 "chroma must be set to 0, 1, 2, 3 or 4.");

        float scale = calc_scale(args[7].AsFloat(255.0));

        bool is_v8 = is_v8orgreater(env);

        arch_t arch = get_arch(args[8].AsInt(-1), env);

        return new TCannyM(args[0].AsClip(), mode, sigma, tmin, tmax, chroma,
                           args[5].AsBool(false), scale, arch, "TCannyMod",
                           is_v8);

    } catch (std::runtime_error& e) {
        env->ThrowError("TCannyMod: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_gblur(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_avx(env), "This filter requires SSE2.");

        float sigma = (float)args[1].AsFloat(0.5);
        validate(sigma < 0.0f, "sigma must be greater than zero.");

        int chroma = args[2].AsInt(1);
        validate(chroma < 0 || chroma > 4,
                 "chroma must be set to 0, 1, 2, 3 or 4.");

        bool is_v8 = is_v8orgreater(env);

        arch_t arch = get_arch(args[3].AsInt(-1), env);

        return new TCannyM(args[0].AsClip(), 4, sigma, 1.0f, 1.0f, chroma,
                           false, 1.0f, arch, "GBlur", is_v8);

    } catch (std::runtime_error& e) {
        env->ThrowError("GBlur: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_emask(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_avx(env), "This filter requires SSE2.");

        float sigma = (float)args[1].AsFloat(1.5);
        validate(sigma < 0.0f, "sigma must be greater than zero.");

        int chroma = args[2].AsInt(0);
        validate(chroma < 0 || chroma > 4,
                 "chroma must be set to 0, 1, 2, 3 or 4.");

        float scale = calc_scale(args[2].AsFloat(50.0));

        bool is_v8 = is_v8orgreater(env);

        arch_t arch = get_arch(args[3].AsInt(-1), env);

        return new TCannyM(args[0].AsClip(), 1, sigma, 1.0f, 1.0f, chroma,
                           args[5].AsBool(false), scale, arch, "EMask", is_v8);

    } catch (std::runtime_error& e) {
        env->ThrowError("EMask: %s", e.what());
    }
    return 0;
}


const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char * __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
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
             /*7*/   "[gmmax]f"
             /*8*/   "[opt]i", create_tcannymod, nullptr);

    env->AddFunction("GBlur", "c[sigma]f[chroma]i[opt]i",
                     create_gblur, nullptr);

    env->AddFunction("EMask", "c[sigma]f[gmmax]f[chroma]i[sobel]b[opt]i",
                     create_emask, nullptr);

    return "Canny edge detection filter for Avisynth2.6/Avisynth+ ver."
        TCANNY_M_VERSION;
}
