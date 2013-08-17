/*
  write_frame.cpp

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


#include <emmintrin.h>
#include "tcannymod.hpp"


void __stdcall TCannyM::
write_dst_frame(const float* srcp, uint8_t* dstp, int width, int height,
                int dst_pitch)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_cvtps_epi32(_mm_load_ps(srcp + x));
            __m128i xmm1 = _mm_cvtps_epi32(_mm_load_ps(srcp + x + 4));
            __m128i xmm2 = _mm_cvtps_epi32(_mm_load_ps(srcp + x + 8));
            __m128i xmm3 = _mm_cvtps_epi32(_mm_load_ps(srcp + x + 12));
            xmm0 = _mm_packs_epi32(xmm0, xmm1);
            xmm1 = _mm_packs_epi32(xmm2, xmm3);
            xmm0 = _mm_packus_epi16(xmm0, xmm1);
            _mm_store_si128((__m128i*)(dstp + x), xmm0);
        }
        srcp += frame_pitch;
        dstp += dst_pitch;
    }
}


void __stdcall TCannyM::
write_edge_direction(int width, int height, uint8_t* dstp, int dst_pitch)
{
    const float* edgep = blur_frame;
    const uint8_t* dir = direction;
    __m128 tmax = _mm_set1_ps(th_max);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(edgep + x), tmax));
            __m128i xmm1 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(edgep + x + 4), tmax));
            __m128i xmm2 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(edgep + x + 8), tmax));
            __m128i xmm3 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(edgep + x + 12), tmax));
            xmm0 = _mm_packs_epi16(_mm_packs_epi32(xmm0, xmm1), _mm_packs_epi32(xmm2, xmm3));
            xmm1 = _mm_load_si128((__m128i*)(dir + x));
            xmm0 = _mm_and_si128(xmm0, xmm1);
            _mm_store_si128((__m128i*)(dstp + x), xmm0);
        }
        edgep += frame_pitch;
        dir += frame_pitch;
        dstp += dst_pitch;
    }
}

void __stdcall TCannyM::
write_binary_mask(int width, int height, uint8_t* dstp, int dst_pitch)
{
    const float* srcp = blur_frame;
    __m128 tmax = _mm_set1_ps(th_max);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128i xmm0 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(srcp + x), tmax));
            __m128i xmm1 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(srcp + x + 4), tmax));
            __m128i xmm2 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(srcp + x + 8), tmax));
            __m128i xmm3 = _mm_castps_si128(_mm_cmpge_ps(_mm_load_ps(srcp + x + 12), tmax));
            xmm0 = _mm_packs_epi16(_mm_packs_epi32(xmm0, xmm1), _mm_packs_epi32(xmm2, xmm3));
            _mm_store_si128((__m128i*)(dstp + x), xmm0);
        }
        srcp += frame_pitch;
        dstp += dst_pitch;
    }
}
