#ifndef WRITE_FRAME_H
#define WRITE_FRAME_H

#include <cstdint>


using write_dst_frame_t = void(_stdcall *)(
    const float* srcp, uint8_t* dstp, int width, int height, int dst_pitch,
    int src_pitch, float scale);

write_dst_frame_t get_write_dst_frame(bool scale);

void __stdcall write_edge_direction(
    const float* edgep, const uint8_t* dir, float th_max, int width,
    int height, const int frame_pitch, uint8_t* dstp, int dst_pitch);

void __stdcall write_binary_mask(
    const float* srcp, float th_max, int width, int height, int src_pitch,
    uint8_t* dstp, int dst_pitch);


#endif
