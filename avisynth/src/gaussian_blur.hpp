/*
  gaussian_blur.hpp

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

#ifndef GAUSSIAN_BLUR_HPP
#define GAUSSIAN_BLUR_HPP


void cvt2flt_sse4_u8(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_sse4_u16(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_sse4_flt(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void gblur_sse4_u8_r1_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u8_r2_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u8_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u8_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u16_r1_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u16_r2_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u16_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_u16_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_flt_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_sse4_flt_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void cvt2flt_avx2_u8(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_avx2_u16(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_avx2_flt(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void gblur_avx2_u8_r1_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u8_r2_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u8_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u8_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u16_r1_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u16_r2_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u16_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_u16_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_flt_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx2_flt_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void cvt2flt_avx512_u8(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_avx512_u16(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void cvt2flt_avx512_flt(const void* srcp, int spitch, float*, int, void* dstp,
    int dpitch, int width, int height, int, const float*, const float);

void gblur_avx512_u8_r1_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u8_r2_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u8_r3_u8(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u8_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u8_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u8_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r1_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r2_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r3_u16(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_u16_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_flt_r1_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_flt_r2_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);

void gblur_avx512_flt_r3_flt(const void* srcp, int spitch, float* hbuffp,
    int fbpitch, void* dstp, int dpitch, int width, int height, int radius,
    const float* weights, float);


#endif // GAUSSIAN_BLUR_HPP
