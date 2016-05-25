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
#include <cmath>
#include <algorithm>
#include <map>
#include <tuple>
#include <stdexcept>
#include "tcannymod.h"
#include "gaussian_blur.h"
#include "edge_detection.h"
#include "write_frame.h"


static edge_detection_t
get_edge_detection(bool use_sobel, bool calc_dir, arch_t arch) noexcept
{
    using std::make_tuple;
    std::map<std::tuple<bool, bool, arch_t>, edge_detection_t> func;

    func[make_tuple(false, false, HAS_SSE2)] = standard<__m128, __m128i, false>;
    func[make_tuple(false, true, HAS_SSE2)] = standard<__m128, __m128i, true>;
    func[make_tuple(true, false, HAS_SSE2)] = sobel<__m128, __m128i, false>;
    func[make_tuple(true, true, HAS_SSE2)] = sobel<__m128, __m128i, true>;
#if defined(__AVX2__)
    func[make_tuple(false, false, HAS_AVX2)] = standard<__m256, __m256i, false>;
    func[make_tuple(false, true, HAS_AVX2)] = standard<__m256, __m256i, true>;
    func[make_tuple(true, false, HAS_AVX2)] = sobel<__m256, __m256i, false>;
    func[make_tuple(true, true, HAS_AVX2)] = sobel<__m256, __m256i, true>;
#endif

    arch_t a = arch == HAS_SSE41 ? HAS_SSE2 : arch;

    return func[make_tuple(use_sobel, calc_dir, a)];
}



static write_gradient_mask_t
get_write_gradient_mask(bool scale, arch_t arch) noexcept
{
#if defined(__AVX2__)
    if (arch == HAS_AVX2) {
        return scale ? write_gradient_mask<__m256, __m256i, true>
                     : write_gradient_mask<__m256, __m256i, false>;
    }
#endif
    return scale ? write_gradient_mask<__m128, __m128i, true>
                 : write_gradient_mask<__m128, __m128i, false>; 

}


static inline void validate(bool cond, const char* msg)
{
    if (cond)
        throw std::runtime_error(msg);
}


template <typename T>
static inline T
my_malloc(size_t size, size_t align, bool is_plus, AvsAllocType at,
          ise_t* env) noexcept
{
    void* p;
    if (is_plus) {
        p = static_cast<IScriptEnvironment2*>(env)->Allocate(size, align, at);
    } else {
        p = _aligned_malloc(size, align);
    }
    return reinterpret_cast<T>(p);
}


static inline void my_free(void* p, bool is_plus, ise_t* env) noexcept
{
    if (is_plus) {
        static_cast<IScriptEnvironment2*>(env)->Free(p);
    } else {
        _aligned_free(p);
    }
    p = nullptr;
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


static arch_t get_arch(int opt, bool is_plus) noexcept
{
    if (opt == 0 || !has_sse41()) {
        return HAS_SSE2;
    }
#if !defined(__AVX2__)
    return HAS_SSE41;
#else
    if (opt == 1 || !has_avx2()) {
        return HAS_SSE41;
    }
    return HAS_AVX2;
#endif
}


TCannyM::TCannyM(PClip ch, int m, float sigma, float tmin, float tmax, int c,
                 bool sobel, float s, int opt, const char* n, bool is_plus) :
    GenericVideoFilter(ch), mode(m), gbRadius(0), th_min(tmin), th_max(tmax),
    chroma(c), name(n), scale(s), isPlus(is_plus)
{
    validate(!vi.IsPlanar(), "Planar format only.");

    numPlanes = (vi.IsY8() || chroma == 0) ? 1 : 3;

    arch_t arch = get_arch(opt, isPlus);

    align = (arch < HAS_AVX2) ? 16 : 32;

    if (sigma > 0.0f) {
        set_gb_kernel(sigma, gbRadius, gbKernel);

        size_t length = (gbRadius * 2 + 1);
        horizontalKernel = my_malloc<float*>(
            length * align, align, false, AVS_NORMAL_ALLOC, nullptr);
        validate(!horizontalKernel, "failed to allocate memory.");

        size_t step = align / sizeof(float);
        for (size_t i = 0; i < length; ++i) {
            for (size_t j = 0; j < step; ++j) {
                horizontalKernel[i * step + j] = gbKernel[i];
            }
        }
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

    switch (arch) {
#if defined(__AVX2__)
    case HAS_AVX2:
        gaussianBlur = gaussian_blur<__m256, GB_MAX_LENGTH, HAS_AVX2>;
        nonMaximumSuppression = non_max_suppress<__m256, __m256i>;
        writeGradientDirection = write_gradient_direction<__m256i>;
        writeEdgeDirection = write_edge_direction<__m256i>;
        break;
#endif
    case HAS_SSE41:
        gaussianBlur = gaussian_blur<__m128, GB_MAX_LENGTH, HAS_SSE41>;
        nonMaximumSuppression = non_max_suppress<__m128, __m128i>;
        writeGradientDirection = write_gradient_direction<__m128i>;
        writeEdgeDirection = write_edge_direction<__m128i>;
        break;
    default:
        gaussianBlur = gaussian_blur<__m128, GB_MAX_LENGTH, HAS_SSE2>;
        nonMaximumSuppression = non_max_suppress<__m128, __m128i>;
        writeGradientDirection = write_gradient_direction<__m128i>;
        writeEdgeDirection = write_edge_direction<__m128i>;
    }

    edgeDetection = get_edge_detection(sobel, (mode != 1 && mode != 4), arch);

    writeBluredFrame = get_write_gradient_mask(false, arch);

    writeGradientMask = get_write_gradient_mask(scale != 1.0f, arch);
}


TCannyM::~TCannyM()
{
    my_free(horizontalKernel, false, nullptr);
}


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
            size_t hystsize, size_t align, bool ip, ise_t* e) :
        env(e), isPlus(ip)
    {
        size_t total_size = bufsize + blsize + emsize + dirsize + hystsize;
        orig = my_malloc<uint8_t*>(
            total_size, align, isPlus, AVS_POOLED_ALLOC, env);

        buffp = reinterpret_cast<float*>(orig) + 8;
        blurp = reinterpret_cast<float*>(orig + bufsize + align);
        emaskp = reinterpret_cast<float*>(orig + bufsize + blsize);
        dirp = reinterpret_cast<int32_t*>(orig + bufsize + blsize + emsize);
        hystp = orig + total_size - hystsize;
    };
    ~Buffers()
    {
        my_free(orig, isPlus, env);
    };
};


PVideoFrame __stdcall TCannyM::GetFrame(int n, ise_t* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi, align);

    auto b = Buffers(buffSize, blurSize, emaskSize, dirSize, hystSize, align,
                     isPlus, env);
    if (b.orig == nullptr) {
        env->ThrowError("%s: failed to allocate buffer.", name);
    }

    const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

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

        gaussianBlur(gbRadius, gbKernel, horizontalKernel, b.buffp, b.blurp,
                     blurPitch, srcp, src_pitch, width, height);
        if (mode == 4) {
            writeBluredFrame(b.blurp, dstp, width, height, dst_pitch,
                             blurPitch, 1.0);
            continue;
        }

        edgeDetection(b.blurp, blurPitch, b.emaskp, emaskPitch, b.dirp,
                      dirPitch, width, height);

        if (mode == 1) {
            writeGradientMask(b.emaskp, dstp, width, height, dst_pitch,
                              emaskPitch, scale);
            continue;
        }
        if (mode == 3) {
            writeGradientDirection(b.dirp, dstp, dirPitch, dst_pitch, width,
                                   height);
            continue;
        }

        nonMaximumSuppression(b.emaskp, emaskPitch, b.dirp, dirPitch, b.blurp,
                              blurPitch, width, height);

        hysteresis(b.hystp, hystPitch, b.blurp, blurPitch, width, height,
                   th_min, th_max);

        if (mode == 2) {
            writeEdgeDirection(b.dirp, b.hystp, dstp, dirPitch, hystPitch,
                               dst_pitch, width, height);
            continue;
        }

        env->BitBlt(dstp, dst_pitch, b.hystp, hystPitch, width, height);
    }

    return dst;
}



static float calc_scale(double gmmax)
{
    return static_cast<float>(255.0 / std::min(std::max(gmmax, 1.0), 255.0));
}


static AVSValue __cdecl
create_tcannymod(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_sse2(), "This filter requires SSE2.");
    
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
    
        bool is_plus = user_data != nullptr;
    
        return new TCannyM(args[0].AsClip(), mode, sigma, tmin, tmax, chroma,
                           args[5].AsBool(false), scale, args[8].AsInt(HAS_AVX2),
                           "TCannyMod", is_plus);
    } catch (std::runtime_error& e) {
        env->ThrowError("TCannyMod: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_gblur(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_sse2(), "This filter requires SSE2.");
    
        float sigma = (float)args[1].AsFloat(0.5);
        validate(sigma < 0.0f, "sigma must be greater than zero.");
    
        int chroma = args[2].AsInt(1);
        validate(chroma < 0 || chroma > 4,
                 "chroma must be set to 0, 1, 2, 3 or 4.");
    
        bool is_plus = user_data != nullptr;
    
        return new TCannyM(args[0].AsClip(), 4, sigma, 1.0f, 1.0f, chroma, false,
                        1.0f, args[3].AsInt(HAS_AVX2), "GBlur", is_plus);
    } catch (std::runtime_error& e) {
        env->ThrowError("GBlur: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_emask(AVSValue args, void* user_data, ise_t* env)
{
    try {
        validate(!has_sse2(), "This filter requires SSE2.");
    
        float sigma = (float)args[1].AsFloat(1.5);
        validate(sigma < 0.0f, "sigma must be greater than zero.");
    
        int chroma = args[2].AsInt(0);
        validate(chroma < 0 || chroma > 4,
                 "chroma must be set to 0, 1, 2, 3 or 4.");
    
        float scale = calc_scale(args[2].AsFloat(50.0));
    
        bool is_plus = user_data != nullptr;
    
        return new TCannyM(args[0].AsClip(), 1, sigma, 1.0f, 1.0f, chroma,
                           args[5].AsBool(false), scale, args[3].AsInt(HAS_AVX2),
                           "EMask", is_plus);
    } catch (std::runtime_error& e) {
        env->ThrowError("EMask: %s", e.what());
    }
    return 0;
}


static const AVS_Linkage* AVS_linkage = nullptr;


extern "C" __declspec(dllexport) const char * __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    void* is_plus = env->FunctionExists("SetFilterMTMode") ? "true" : nullptr;

    env->AddFunction("TCannyMod",
             /*0*/   "c"
             /*1*/   "[mode]i"
             /*2*/   "[sigma]f"
             /*3*/   "[t_h]f"
             /*4*/   "[t_l]f"
             /*5*/   "[sobel]b"
             /*6*/   "[chroma]i"
             /*7*/   "[gmmax]f"
             /*8*/   "[opt]i", create_tcannymod, is_plus);

    env->AddFunction("GBlur", "c[sigma]f[chroma]i[opt]i",
                     create_gblur, is_plus);
    env->AddFunction("EMask", "c[sigma]f[gmmax]f[chroma]i[sobel]b[opt]i",
                     create_emask, is_plus);

    if (is_plus != nullptr) {
        auto env2 = static_cast<IScriptEnvironment2*>(env);
        env2->SetFilterMTMode("TCannyMod", MT_NICE_FILTER, true);
        env2->SetFilterMTMode("GBlur", MT_NICE_FILTER, true);
        env2->SetFilterMTMode("EMask", MT_NICE_FILTER, true);
    }

    return "Canny edge detection filter for Avisynth2.6/Avisynth+ ver."
        TCANNY_M_VERSION;
}
