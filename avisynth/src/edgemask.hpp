/*
  edgemask.hpp

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

#ifndef EDGEMASK_HPP
#define EDGEMASK_HPP


void emask_sse4_u8_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u8_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_u16_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_ns_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_sse4_flt_sc_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u8_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_u16_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_ns_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx2_flt_sc_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u8_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_u16_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_std_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_sobel_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_custom_fast(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_std_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_sobel_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_custom_strict(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_std_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_sobel_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_custom_fast_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_std_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_sobel_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_ns_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);

void emask_avx512_flt_sc_custom_strict_dir(const float* blurp, int blpitch, void* dstp, int dpitch, std::array<float, 3>& opr,
    float scale, int width, int height, float maxval, int32_t* dirp,
    int dirpitch);


void nms_sse4(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height);

void nms_avx2(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height);

void nms_avx512(float* emaskp, const int epitch, const int32_t* dirp, int dirpitch,
    float* dstp, int dpitch, const int width, const int height);

#endif //  EDGEMASK_HPP