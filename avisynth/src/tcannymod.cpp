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
#include "tcannymod.h"


static gaussian_blur_t get_gaussian_blur(arch_t arch)
{
    switch (arch) {
    case HAS_SSE2:
        return gaussian_blur<__m128, GB_MAX_LENGTH, HAS_SSE2>;
    case HAS_SSE41:
        return gaussian_blur<__m128, GB_MAX_LENGTH, HAS_SSE41>;
    default:
        break;
    }
    return gaussian_blur<__m256, GB_MAX_LENGTH, HAS_AVX2>;
}


static edge_detection_t
get_edge_detection(bool use_sobel, bool calc_dir, arch_t arch)
{
    using std::make_tuple;
    std::map<std::tuple<bool, bool, arch_t>, edge_detection_t> func;

    func[make_tuple(false, false, HAS_SSE2)] = standerd<__m128, __m128i, false>;
    func[make_tuple(false, false, HAS_AVX2)] = standerd<__m256, __m256i, false>;
    func[make_tuple(false, true, HAS_SSE2)] = standerd<__m128, __m128i, true>;
    func[make_tuple(false, true, HAS_AVX2)] = standerd<__m256, __m256i, true>;

    func[make_tuple(true, false, HAS_SSE2)] = sobel<__m128, __m128i, false>;
    func[make_tuple(true, false, HAS_AVX2)] = sobel<__m256, __m256i, false>;
    func[make_tuple(true, true, HAS_SSE2)] = sobel<__m128, __m128i, true>;
    func[make_tuple(true, true, HAS_AVX2)] = sobel<__m256, __m256i, true>;

    arch_t a = arch == HAS_SSE41 ? HAS_SSE2 : arch;

    return func[make_tuple(use_sobel, calc_dir, a)];
}

static write_gradient_mask_t get_write_gradient_mask(bool scale, arch_t arch)
{
    if (arch < HAS_AVX2) {
        return scale ? write_gradient_mask<__m128, __m128i, true>
                     : write_gradient_mask<__m128, __m128i, false>; 
    }
    return scale ? write_gradient_mask<__m256, __m256i, true>
                 : write_gradient_mask<__m256, __m256i, false>;
}


static write_direction_map_t get_write_direction_map(arch_t arch)
{
    if (arch < HAS_AVX2) {
        return write_direction_map<__m128, __m128i>;
    }
    return write_direction_map<__m256, __m256i>;
}


static void __stdcall
set_gb_kernel(float sigma, int& radius, float* kernel)
{
    radius = std::max(static_cast<int>(sigma * 3.0f + 0.5f), 1);
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


static arch_t get_arch(int opt)
{
    // on 32bit with /arch:AVX outputs weird result.
    // I don't know why...
    if (opt == 0 || !has_sse41()) {
        return HAS_SSE2;
    }
#if !defined(_WIN64)
    return HAS_SSE41;
#else
    if (opt == 1 || !has_avx2()) {
        return HAS_SSE41;
    }
    return HAS_AVX2;
#endif
}

TCannyM::TCannyM(PClip ch, int m, float sigma, float tmin, float tmax, int c,
                 bool sobel, float s, int opt, const char* n, ise_t* env) :
    GenericVideoFilter(ch), mode(m), gbRadius(0), th_min(tmin), th_max(tmax),
    chroma(c), name(n), scale(s)
{
    if (!vi.IsPlanar()) {
        env->ThrowError("%s: Planar format only.", name);
    }

    if (vi.width > 65535 || vi.height > 65535) {
        env->ThrowError("%s: width/height must be smaller than 65536.", name);
    }

    numPlanes = (vi.IsY8() || chroma == 0) ? 1 : 3;

    if (sigma > 0.0f) {
        set_gb_kernel(sigma, gbRadius, gbKernel);
        if (gbRadius == 0) {
            env->ThrowError("%s: sigma is too large.", name);
        }
    }

    arch_t arch = get_arch(opt);

    align = (arch < HAS_AVX2) ? 16 : 32;

    blurPitch = ((align + (vi.width + 1) * sizeof(float)) + align - 1) & ~(align - 1);
    emaskPitch = (vi.width * sizeof(float) + align - 1) & ~(align - 1);
    dirHystPitch = (vi.width + align - 1) & ~(align - 1);

    buffSize = ((8 + vi.width + 8) * sizeof(float) + align - 1) & ~(align - 1);
    blurSize = blurPitch * (vi.height + 1);
    emaskSize = mode == 4 ? 0 : emaskPitch * (vi.height + 1);
    dirSize = (mode == 1 || mode == 4) ? 0 : dirHystPitch * (vi.height + 1);
    hystSize = (mode == 0 || mode == 2) ? dirHystPitch * vi.height : 0;

    blurPitch /= sizeof(float);
    emaskPitch /= sizeof(float);

    gaussianBlur = get_gaussian_blur(arch);
    edgeDetection = get_edge_detection(sobel, (mode != 1 && mode != 4), arch);
    writeBluredFrame = get_write_gradient_mask(false, arch);
    writeGradientMask = get_write_gradient_mask(scale != 1.0f, arch);
    writeDirectionMap = get_write_direction_map(arch);
}


class Buffers {
    uint8_t* orig;
public:
    float* buffp;
    float* blurp;
    float* emaskp;
    uint8_t* dirp;
    uint8_t* hystp;
    Buffers(size_t bufsize, size_t blsize, size_t emsize, size_t dirsize,
            size_t hystsize, size_t align, ise_t* env, const char* name) {
        size_t all_size = bufsize + blsize + emsize + dirsize + hystsize;
        orig = static_cast<uint8_t*>(_aligned_malloc(all_size, align));
        if (!orig) {
            env->ThrowError("%s: failed to alocate buffers.", name);
        }
        buffp = reinterpret_cast<float*>(orig) + 8;
        blurp = reinterpret_cast<float*>(orig + bufsize + align);
        emaskp = reinterpret_cast<float*>(orig + bufsize + blsize);
        dirp = orig + bufsize + blsize + emsize;
        hystp = dirp + dirsize;
    };
    ~Buffers() {
        if (orig) {
            _aligned_free(orig);
            orig = nullptr;
        }
    };
};


PVideoFrame __stdcall TCannyM::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi, align);

    auto b = Buffers(buffSize, blurSize, emaskSize, dirSize, hystSize, align,
                     env, name);

    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

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

        if ((reinterpret_cast<uintptr_t>(srcp) & (align - 1)) ||
                (src_pitch | dst_pitch) & (align - 1)) {
            env->ThrowError("%s: Invalid memory alignment", name);
        }

        gaussianBlur(gbRadius, gbKernel, b.buffp, b.blurp, blurPitch, srcp,
                     src_pitch, width, height);
        if (mode == 4) {
            writeBluredFrame(b.blurp, dstp, width, height, dst_pitch,
                             blurPitch, 1.0);
            continue;
        }

        edgeDetection(b.blurp, blurPitch, b.emaskp, emaskPitch, b.dirp,
                      dirHystPitch, width, height);

        if (mode == 1) {
            writeGradientMask(b.emaskp, dstp, width, height, dst_pitch,
                              emaskPitch, scale);
            continue;
        }
        if (mode == 3) {
            env->BitBlt(dstp, dst_pitch, b.dirp, dirHystPitch, width, height);
            continue;
        }

        non_max_suppress(b.emaskp, emaskPitch, b.dirp, dirHystPitch, b.blurp,
                         blurPitch, width, height);

        hysteresis(b.hystp, dirHystPitch, b.blurp, blurPitch, width, height,
                   th_min, th_max);

        if (mode == 2) {
            writeDirectionMap(b.hystp, b.dirp, dirHystPitch, dstp,
                              dst_pitch, width, height);
            continue;
        }

        env->BitBlt(dstp, dst_pitch, b.hystp, dirHystPitch, width, height);
    }

    return dst;
}


static AVSValue __cdecl
create_tcannymod(AVSValue args, void* user_data, ise_t* env)
{
    if (!has_sse2()) {
        env->ThrowError("TCannyMod: This filter requires SSE2.");
    }
    int mode = args[1].AsInt(0);
    if (mode < 0 || mode > 4) {
        env->ThrowError("TCannyMod: mode must be between 0 and 4.");
    }

    float sigma = static_cast<float>(args[2].AsFloat(1.5f));
    if (sigma < 0.0f) {
        env->ThrowError("TCannyMod: sigma must be greater than zero.");
    }

    float tmin = static_cast<float>(args[4].AsFloat(0.1f));
    if (tmin < 0.0f) {
        env->ThrowError("TCannyMod: t_l must be greater than zero.");
    }

    float tmax = static_cast<float>(args[3].AsFloat(8.0f));
    if (tmax < tmin) {
        env->ThrowError("TCannyMod: t_h must be greater than t_l.");
    }

    int chroma = args[6].AsInt(0);
    if (chroma < 0 || chroma > 4) {
        env->ThrowError("TCannyMod: chroma must be set to 0, 1, 2, 3 or 4.");
    }

    float scale = static_cast<float>(
        255.0 / std::min(std::max(args[7].AsFloat(255.0), 1.0), 255.0));

    return new TCannyM(args[0].AsClip(), mode, sigma, tmin, tmax, chroma,
                       args[5].AsBool(false), scale, args[8].AsInt(HAS_AVX2),
                       "TCannyMod", env);
}


static AVSValue __cdecl
create_gblur(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    if (!has_sse2()) {
        env->ThrowError("GBlur: This filter requires SSE2.");
    }
    float sigma = (float)args[1].AsFloat(0.5);
    if (sigma < 0.0f) {
        env->ThrowError("GBlur: sigma must be greater than zero.");
    }
    int chroma = args[2].AsInt(1);
    if (chroma < 0 || chroma > 4) {
        env->ThrowError("GBlur: chroma must be set to 0, 1, 2, 3 or 4.");
    }

    return new TCannyM(args[0].AsClip(), 4, sigma, 1.0f, 1.0f, chroma, false,
                       1.0f, args[3].AsInt(HAS_AVX2), "GBlur", env);
}


static const AVS_Linkage* AVS_linkage = nullptr;


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
             /*7*/   "[gmmax]f"
             /*8*/   "[opt]i", create_tcannymod, nullptr);
    env->AddFunction("GBlur", "c[sigma]f[chroma]i[opt]i", create_gblur, nullptr);
    return "Canny edge detection filter for Avisynth2.6 ver." TCANNY_M_VERSION;
}
