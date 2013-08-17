#ifndef TCANNY_MOD_HPP
#define TCANNY_MOD_HPP

#include <stdint.h>
#include <windows.h>
#include "avisynth.h"

#define TCANNY_M_VERSION "0.0.1"

#define GB_MAX_LENGTH 17


class TCannyM : public GenericVideoFilter {
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
    void __stdcall edge_detect(int width, int height);
    void __stdcall non_max_suppress(int width, int height);
    void __stdcall hysteresiss(int width, int height);
    void __stdcall write_dst_frame(const float* srcp, uint8_t* dstp, int width, int height, int dst_pitch);
    void __stdcall write_edge_direction(int width, int height, uint8_t* dstp, int dst_pitch);
    void __stdcall write_binary_mask(int width, int height, uint8_t* dstp, int dst_pitch);

public:
    TCannyM(PClip child, int mode, float sigma, float th_min, float th_max,
            int chroma, IScriptEnvironment* env);
    ~TCannyM();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};

#endif
