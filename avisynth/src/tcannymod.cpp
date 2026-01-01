/*
  tcannymod.cpp

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


#include <chrono>
#include <format>
#include <algorithm>
#include "tcannymod.hpp"
#include "utils.hpp"


struct Buffer {
    ise_t* env;
    bool isV8;
    size_t size;
    uint8_t* orig;
    float* hbuff;
    float* blurp;
    float* emaskp;
    int32_t* dirp;
    Buffer(size_t hbsize, size_t blsize, size_t emsize, size_t dirsize,
        size_t align, int hbpad, bool v8, ise_t* e) : env(e), isV8(v8),
        size(emsize)
    {
        size_t total = hbsize + blsize + emsize + dirsize;
        void* p = isV8 ? env->Allocate(total, align, AVS_POOLED_ALLOC)
            : avs_malloc(total, align);
        validate(!p, "failed to allocate temporal memory.");

        orig = reinterpret_cast<uint8_t*>(p);
        hbuff = reinterpret_cast<float*>(orig + hbpad);
        blurp = reinterpret_cast<float*>(orig + hbsize);
        emaskp = reinterpret_cast<float*>(orig + hbsize + blsize);
        dirp = reinterpret_cast<int32_t*>(orig + hbsize + blsize + emsize);
    }
    ~Buffer()
    {
        if (isV8) {
            env->Free(orig);
        } else {
            avs_free(orig);
        }
        orig = nullptr;
        env = nullptr;
    }
};


PVideoFrame __stdcall TCannyMod::getFrameDebug(int n, ise_t* env)
{
    using namespace std::chrono;

    Buffer buff(hbSize, blSize, emSize, dirSize, align, hbPad, true, env);
    auto src = child->GetFrame(n, env);
    auto dst = env->NewVideoFrameP(vi, &src);

    auto start = system_clock::now();

    mainLoop(src, dst, buff, env);

    auto end = system_clock::now();
    auto pt = duration_cast<microseconds>(end - start).count();


    auto map = env->getFramePropsRW(dst);
    env->propSetInt(map, "TCM_gbradius", radius, PROPAPPENDMODE_APPEND);
    env->propSetFloatArray(map, "TCM_gbkernel", dbgweights.data(),
        static_cast<int>(dbgweights.size()));
    env->propSetDataH(map, "TCM_opt", opt.c_str(), int(opt.length()),
        PROPDATATYPEHINT_UTF8, PROPAPPENDMODE_APPEND);
    env->propSetInt(map, "GB_procTime", pt, PROPAPPENDMODE_APPEND);

    return dst;
}


void TCannyMod::mainLoop(PVideoFrame& src, PVideoFrame& dst, Buffer& buff,
    ise_t* env)
{
    const int p[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int i = 0; i < numPlanes; ++i) {
        auto plane = p[i];
        auto srcp = src->GetReadPtr(plane);
        auto spitch = src->GetPitch(plane) / bytes;
        auto width = src->GetRowSize(plane) / bytes;
        auto height = src->GetHeight(plane);
        auto dstp = dst->GetWritePtr(plane);
        auto dpitch = dst->GetPitch(plane) / bytes;

        if (i > 0) {
            if (mode & mode_t::COPY_CHROMA) {
                env->BitBlt(dstp, dpitch * bytes, srcp, spitch * bytes,
                    width * bytes, height);
                continue;
            } else if (mode & mode_t::FILL_HALF_CHROMA) {
                uint32_t* d = reinterpret_cast<uint32_t*>(dstp);
                std::fill_n(d, dpitch * bytes * height / sizeof(uint32_t),
                    get_halfvalue(bits));
                continue;
            } else if (mode & mode_t::FILL_ZERO_CHROMA) {
                memset(dstp, dpitch * bytes * height, 0);
                continue;
            }
        }
        if (mode & mode_t::DO_BLUR_ONLY) {
            gaussianBlur(srcp, spitch, buff.hbuff, hbPitch, dstp, dpitch,
                width, height, radius, gbweights.data(), maxval);
            continue;
        }
        gaussianBlur(srcp, spitch, buff.hbuff, hbPitch, buff.blurp,
            blPitch, width, height, radius, gbweights.data(), maxval);

        if ((mode & mode_t::CALC_DIRECTION) == 0) {
            edgeMask(buff.blurp, blPitch, dstp, dpitch, opr, scale,
                width, height, maxval, nullptr, 0);
            continue;
        }

        edgeMask(buff.blurp, blPitch, buff.emaskp, emPitch, opr, scale, width,
            height, maxval, buff.dirp, dirPitch);

        if ((mode & mode_t::SHOW_DIRECTION)) {
            writeDirections(buff.dirp, dirPitch, dstp, dpitch, width, height);
            continue;
        }

        nonMaximumSuppression(buff.emaskp, emPitch, buff.dirp, dirPitch,
            buff.blurp, blPitch, width, height);

        hysteresis(dstp, dpitch, buff.blurp, blPitch, width, height, tmin, tmax,
            maxval);
    }
}


PVideoFrame __stdcall TCannyMod::GetFrame(int n, ise_t* env)
{
    if (mode & mode_t::SET_DEBUG_INFO) {
        return getFrameDebug(n, env);
    }

    bool isV8 = mode & mode_t::AT_LEAST_V8;
    Buffer buff(hbSize, blSize, emSize, dirSize, align, hbPad, isV8, env);
    auto src = child->GetFrame(n, env);
    auto dst = isV8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    mainLoop(src, dst, buff, env);

    return dst;
}


void TCannyMod::generateWeights(float sigma)
{
    int w = vi.width, h = vi.height;
    if ((mode & mode_t::PROC_CHROMA) && numPlanes > 1) {
        if (vi.IsYV411()) w /= 4;
        if (vi.Is422() || vi.Is420()) w /= 2;
        if (vi.Is420()) h /= 2;
    }

    auto t = static_cast<int>(sigma * 3 + 0.5f);
    radius = std::max(1, t);
    validate(std::min(w, h) < radius, "sigma is too large");

    int length = radius * 2 + 1;
    gbweights.resize(length, 0.0f);
    dbgweights.resize(length, 0.0);

    float sum = 0.0f;
    for (int r = -radius; r <= radius; ++r) {
        float weight = std::exp((-1 * r * r) / (2.0f * sigma * sigma));
        gbweights[r + radius] = weight;
        sum += weight;
    }
    for (int i = 0; i < length; ++i) {
        gbweights[i] /= sum;
        dbgweights[i] = gbweights[i];
    }
}


TCannyMod::TCannyMod(PClip c, float _tmin, float _tmax, float _sc,
    operator_t& _o, float sigma, int _m, arch_t _a) :
    GenericVideoFilter(c), tmin(_tmin), tmax(_tmax), scale(_sc), opr(_o),
    mode(_m), arch(_a), radius(0), hbPitch(0), hbPad(0), blPitch(0),
    emPitch(0), dirPitch(0), hbSize(0), blSize(0), emSize(0), dirSize(0),
    edgeMask(nullptr), writeDirections(nullptr), hysteresis(nullptr),
    nonMaximumSuppression(nullptr)
{
    validate(!vi.IsPlanar(), "Planar format only.");
    bits = vi.BitsPerComponent();
    bytes = (bits + 7) / 8;
    numPlanes = (vi.IsY() || mode & mode_t::DO_NOT_TOUCH_CHROMA) ? 1 : 3;
    maxval = bits == 32 ? 1.0f :
        vi.IsRGB() ? 1.0f * ((1 << bits) - 1) : 1.0f * (0xFF << (bits - 8));

    opt = a2s(arch);

    align = 64;
    int bm = align - 1;

    if ((mode & mode_t::DO_NOT_BLUR) == 0) {
        generateWeights(sigma);
        hbPad = (radius * sizeof(float) + bm) & ~bm;
        hbPitch = (2 * hbPad + (vi.width * sizeof(float) + bm)) & ~bm;
        hbSize = hbPitch * (arch == USE_AVX512 ? 6 : 4);
        hbPitch /= sizeof(float);
    }

    if (mode & mode_t::DETECT_EDGE) {
        blPitch = (vi.width * sizeof(float) + bm) & ~bm;
        blSize = blPitch * vi.height;
        blPitch /= sizeof(float);
    }

    if (mode & mode_t::CALC_DIRECTION) {
        dirPitch = blPitch;
        dirSize = blSize;
    }

    if (mode & GENERATE_CANNY_IMAGE) {
        emPitch = blPitch;
        emSize = blSize;
    }

    gaussianBlur = get_gblur(bytes, arch, radius, mode);

    edgeMask = get_emask(bytes, arch, mode);

    writeDirections = get_write_dir(bytes);

    nonMaximumSuppression = get_nms(arch);

    hysteresis = get_hysteresis(bytes);

}


static void set_chroma_mode(int chroma, int& mode)
{
    int ret = 0;
    switch (chroma) {
    case 0: mode |= mode_t::DO_NOT_TOUCH_CHROMA; return;
    case 1: mode |= mode_t::PROC_CHROMA; return;
    case 2: mode |= mode_t::COPY_CHROMA; return;
    case 3: mode |= mode_t::FILL_HALF_CHROMA; return;
    case 4: mode |= mode_t::FILL_ZERO_CHROMA; return;
    }
}


arch_t get_arch(int opt)
{
    if (opt == 0) {
        return arch_t::NO_SIMD;
    }
    if (opt == 1) {
        if (has_sse41()) return arch_t::USE_SSE4;
        return arch_t::NO_SIMD;
    }
    if (opt == 2) {
        if (has_avx2()) return arch_t::USE_AVX2;
        if (has_sse41()) return arch_t::USE_SSE4;
        return arch_t::NO_SIMD;
    }
    if (has_avx512()) return arch_t::USE_AVX512;
    if (has_avx2()) return arch_t::USE_AVX2;
    if (has_sse41()) return arch_t::USE_SSE4;
    return arch_t::NO_SIMD;
}


static operator_t parse_operator(const char* o, int& mode)
{
    std::string ostring(o);
    if (ostring == "standard") {
        mode |= mode_t::USE_STANDARD_OPERATOR;
        return operator_t{ 0.0f, 1.0f, 0.0f };
    } else if (ostring == "sobel") {
        mode |= mode_t::USE_SOBEL_OPERATOR;
        return operator_t{ 1.0f, 2.0f, 1.0f };
    } else if (ostring == "prewitt") {
        mode |= mode_t::USE_CUSTOM_OPERATOR;
        return operator_t{ 1.0f, 1.0f, 1.0f };
    }

    try {
        auto v = split(ostring, " ");
        validate(v.size() != 3, nullptr);

        operator_t opr;
        opr[0] = std::stof(v[0]);
        opr[1] = std::stof(v[1]);
        opr[2] = std::stof(v[2]);
        if (opr[0] == 0.0f && opr[1] == 1.0f && opr[2] == 0.0f) {
            mode |= mode_t::USE_STANDARD_OPERATOR;
        } else if (opr[0] == 1.0f && opr[1] == 2.0f && opr[2] == 1.0f) {
            mode |= mode_t::USE_SOBEL_OPERATOR;
        } else {
            mode |= mode_t::USE_CUSTOM_OPERATOR;
        }
        return opr;
    } catch (std::exception&) {
        throw std::runtime_error("invalid operator is set.");
    }
}


static AVSValue __cdecl
create_gblur(AVSValue args, void* user_data, ise_t* env)
{
    try {
        int mode = mode_t::DO_BLUR_ONLY;
        if (user_data != nullptr) {
            mode |= mode_t::AT_LEAST_V8;
        }

        auto clip = args[0].AsClip();

        float sigma = static_cast<float>(args[1].AsFloat(1.50));
        validate(sigma <= 0.0f, "sigma must be greater than zero.");

        auto chroma = args[2].AsInt(0);
        validate(chroma < 0 || chroma > 4, "chroma must be 0, 1, 2, 3 or 4");
        set_chroma_mode(chroma, mode);

        auto arch = get_arch(args[3].AsInt(-1));

        if (args[4].AsBool(false) && user_data != nullptr) {
            mode |= mode_t::SET_DEBUG_INFO;
        }

        operator_t o = parse_operator("standard", mode);

        return new TCannyMod(clip, 0.0f, 0.0f, 1.0f, o, sigma, mode, arch);

    } catch (std::exception& e) {
        env->ThrowError("GBlur2: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_emask(AVSValue args, void* user_data, ise_t* env)
{
    try {
        int mode = mode_t::DETECT_EDGE;
        if (user_data != nullptr) {
            mode |= mode_t::AT_LEAST_V8;
        }

        auto clip = args[0].AsClip();

        auto opr = parse_operator(args[1].AsString("standard"), mode);

        float scale = static_cast<float>(args[2].AsFloat(5.1));
        validate(scale <= 0.0f, "scale must be greater than zero.");
        if (scale != 1.0f) {
            mode |= mode_t::SCALE_MAGNITUDE;
        }

        float sigma = static_cast<float>(args[3].AsFloat(0.50));
        validate(sigma < 0.0f, "sigma must be greater than or equal to zero.");
        if (sigma == 0.0f) {
            mode |= mode_t::DO_NOT_BLUR;
        }

        if (args[4].AsBool(false)) {
            mode |= mode_t::STRICT_MAGNITUDE;
        }

        auto chroma = args[5].AsInt(0);
        validate(chroma < 0 || chroma > 4, "chroma must be 0, 1, 2, 3 or 4");
        set_chroma_mode(chroma, mode);

        auto arch = get_arch(args[6].AsInt(-1));

        if (args[7].AsBool(false) && user_data != nullptr) {
            mode |= mode_t::SET_DEBUG_INFO;
        }

        return new TCannyMod(clip, 0, 0, scale, opr, sigma, mode, arch);

    } catch (std::exception& e) {
        env->ThrowError("EMask: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_dirmap(AVSValue args, void* user_data, ise_t* env)
{
    try {
        int mode = mode_t::DETECT_EDGE | mode_t::CALC_DIRECTION
            | mode_t::SHOW_DIRECTION;

        if (user_data != nullptr) {
            mode |= mode_t::AT_LEAST_V8;
        }

        auto clip = args[0].AsClip();

        auto opr = parse_operator(args[1].AsString("standard"), mode);

        float sigma = static_cast<float>(args[2].AsFloat(1.50));
        validate(sigma < 0.0f, "sigma must be greater than or equal to zero.");
        if (sigma == 0.0f) {
            mode |= mode_t::DO_NOT_BLUR;
        }

        auto chroma = args[3].AsInt(0);
        validate(chroma < 0 || chroma > 4, "chroma must be 0, 1, 2, 3 or 4");
        set_chroma_mode(chroma, mode);

        auto arch = get_arch(args[4].AsInt(-1));

        if (args[5].AsBool(false) && user_data != nullptr) {
            mode |= mode_t::SET_DEBUG_INFO;
        }

        return new TCannyMod(clip, 0, 0, 1.0f, opr, sigma, mode, arch);

    } catch (std::exception& e) {
        env->ThrowError("DirMap: %s", e.what());
    }
    return 0;
}


static AVSValue __cdecl
create_canny(AVSValue args, void* user_data, ise_t* env)
{
    try {
        int mode = mode_t::DETECT_EDGE | mode_t::CALC_DIRECTION
            | mode_t::GENERATE_CANNY_IMAGE;

        if (user_data != nullptr) {
            mode |= mode_t::AT_LEAST_V8;
        }

        auto clip = args[0].AsClip();

        auto tmin = static_cast<float>(args[1].AsFloat(1.0));
        validate(tmin <= 0.0f, "t_l must be greater than 0.");

        auto tmax = static_cast<float>(args[2].AsFloat(8.0));
        validate(tmax <= tmin, "t_h must be greater than t_l.");

        auto opr = parse_operator(args[3].AsString("standard"), mode);

        float scale = static_cast<float>(args[4].AsFloat(1.0));
        validate(scale <= 0.0f, "scale must be greater than zero.");
        if (scale != 1.0f) {
            mode |= mode_t::SCALE_MAGNITUDE;
        }

        float sigma = static_cast<float>(args[5].AsFloat(1.50));
        validate(sigma < 0.0f, "sigma must be greater than or equal to zero.");
        if (sigma == 0.0f) {
            mode |= mode_t::DO_NOT_BLUR;
        }

        if (args[6].AsBool(true)) {
            mode |= mode_t::STRICT_MAGNITUDE;
        }

        auto chroma = args[7].AsInt(0);
        validate(chroma < 0 || chroma > 4, "chroma must be 0, 1, 2, 3 or 4");
        set_chroma_mode(chroma, mode);

        auto arch = get_arch(args[8].AsInt(-1));

        if (args[9].AsBool(false) && user_data != nullptr) {
            mode |= mode_t::SET_DEBUG_INFO;
        }

        return new TCannyMod(clip, tmin, tmax, scale, opr, sigma, mode, arch);

    } catch (std::exception& e) {
        env->ThrowError("TCannyMod: %s", e.what());
    }
    return 0;
}


static const AVS_Linkage* AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(ise_t* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    bool isV8 = false;
    try {
        env->CheckVersion(8);
        isV8 = true;
    } catch (...) {
        isV8 = false;
    }

    env->AddFunction("GBlur2",
        /*0*/   "c"
        /*1*/   "[sigma]f"
        /*2*/   "[chroma]i"
        /*3*/   "[opt]i"
        /*4*/   "[debug]b", create_gblur, isV8 ? &isV8 : nullptr);

    env->AddFunction("Emask",
        /*0*/   "c"
        /*1*/   "[operator]s"
        /*2*/   "[scale]f"
        /*3*/   "[sigma]f"
        /*4*/   "[strict]b"
        /*5*/   "[chroma]i"
        /*6*/   "[opt]i"
        /*7*/   "[debug]b", create_emask, isV8 ? &isV8 : nullptr);

    env->AddFunction("DirMap",
        /*0*/   "c"
        /*1*/   "[operator]s"
        /*2*/   "[sigma]f"
        /*3*/   "[chroma]i"
        /*4*/   "[opt]i"
        /*5*/   "[debug]b", create_dirmap, isV8 ? &isV8 : nullptr);

    env->AddFunction("TCannyMod",
        /*0*/   "c"
        /*1*/   "[t_l]f"
        /*2*/   "[t_h]f"
        /*3*/   "[operator]s"
        /*4*/   "[scale]f"
        /*5*/   "[sigma]f"
        /*6*/   "[strict]b"
        /*7*/   "[chroma]i"
        /*8*/   "[opt]i"
        /*9*/   "[debug]b", create_canny, isV8 ? &isV8 : nullptr);

    return "Canny Edge Detection Filter for avisynth+ ver." TCANNY_M_VERSION;
}
