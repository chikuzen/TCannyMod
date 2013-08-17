#include <malloc.h>
#include <math.h>
#include "tcannymod.hpp"


static const AVS_Linkage* AVS_linkage = 0;


static void __stdcall
set_gb_kernel(float sigma, int& radius, float* kernel)
{
    int length = max((int)(sigma * 3.0f + 0.5f), 1) * 2 + 1;
    if (length > GB_MAX_LENGTH) {
        return;
    }
    radius = length / 2;

    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float weight = expf((-1 * i * i) / (2.0f * sigma * sigma));
        kernel[i + radius] = weight;
        sum += weight;
    }
    for (int i = 0; i < length; kernel[i++] /= sum);
}


TCannyM::TCannyM(PClip ch, int m, float sigma, float tmin, float tmax,
                 int c, IScriptEnvironment* env)
    : GenericVideoFilter(ch), mode(m), gb_radius(0), th_min(tmin), th_max(tmax),
      chroma(c)
{
    if (!vi.IsPlanar()) {
        env->ThrowError("TCanny2: Planar format only.");
    }

    if (vi.IsY8()) {
        chroma = 0;
    }

    set_gb_kernel(sigma, gb_radius, gb_kernel);
    if (gb_radius == 0) {
        env->ThrowError("TCanny2: sigma is too large.");
    }

    frame_pitch = ((vi.width + 15) / 16) * 16;
    blur_frame = (float*)_aligned_malloc(frame_pitch * sizeof(float) * vi.height, 16);
    edge_mask = (float*)_aligned_malloc(frame_pitch * sizeof(float) * vi.height, 16);
    direction = (uint8_t*)_aligned_malloc(frame_pitch * vi.height, 16);

    buff_pitch = ((vi.width + 16 + 15) / 16) * 16;
    buff = (float*)_aligned_malloc(buff_pitch * sizeof(float) * 3, 16);

    hysteresiss_map = (uint8_t*)malloc(vi.width * vi.height);

    if (!blur_frame || !edge_mask || !direction || !buff || !hysteresiss_map) {
        env->ThrowError("TCanny2: failed to allocate temporal buffer.");
    }
}


TCannyM::~TCannyM()
{
    _aligned_free(blur_frame);
    _aligned_free(edge_mask);
    _aligned_free(direction);
    _aligned_free(buff);
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

        gaussian_blur(srcp, src_pitch, width, height);
        if (mode == 4) {
            write_dst_frame(blur_frame, dstp, width, height, dst_pitch);
            continue;
        }

        edge_detect(width, height);
        if (mode == 1) {
            write_dst_frame(edge_mask, dstp, width, height, dst_pitch);
            continue;
        }
        if (mode == 3) {
            env->BitBlt(dstp, dst_pitch, direction, frame_pitch, width, height);
            continue;
        }

        non_max_suppress(width, height);
        hysteresiss(width, height);
        if (mode == 2) {
            write_edge_direction(width, height, dstp, dst_pitch);
            continue;
        }
        write_binary_mask(width, height, dstp, dst_pitch);
    }

    return dst;
}


static AVSValue __cdecl
create_tcannymod(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    int mode = args[1].AsInt(0);
    if (mode < 0 || mode > 4) {
        env->ThrowError("TCannyMod: mode must be between 0 and 4.");
    }

    float sigma = args[2].AsFloat(1.5);
    if (sigma <= 0.0f) {
        env->ThrowError("TCannyMod: sigma must be greater than zero.");
    }

    float min = args[4].AsFloat(0.1f);
    if (min < 0.0f) {
        env->ThrowError("TCannyMod: t_l must be greater than zero.");
    }

    float max = args[3].AsFloat(8.0);
    if (max < min) {
        env->ThrowError("TCannyMod: t_h must be greater than t_l.");
    }

    int chroma = args[5].AsInt(0);
    if (chroma < 0 || chroma > 3) {
        env->ThrowError("TCannyMod: chroma must be set to 0, 1, 2 or 3.");
    }

    return new TCannyM(args[0].AsClip(), mode, sigma, min, max, chroma,
                       env);
}

extern "C" __declspec(dllexport) const char * __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("TCannyMod", "c[mode]i[sigma]f[t_h]f[t_l]f[chroma]i",
                     create_tcannymod, 0);
    return "Canny edge detection filter for Avisynth2.6 ver." TCANNY_M_VERSION;
}
