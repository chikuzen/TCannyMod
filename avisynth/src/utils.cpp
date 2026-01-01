/*
    utils.cpp

    This file is a part of TCannyMod.

    Copyright (C) 2026 OKA Motofumi

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

#if defined(_MSC_VER)
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif
#include "utils.hpp"


enum : uint32_t {
    CPU_NO_X86_SIMD             = 0x00000000,
    CPU_SSE2_SUPPORT            = 0x00000001,
    CPU_SSE3_SUPPORT            = 0x00000002,
    CPU_SSSE3_SUPPORT           = 0x00000004,
    CPU_SSE4_1_SUPPORT          = 0x00000008,
    CPU_SSE4_2_SUPPORT          = 0x00000010,
    CPU_SSE4_A_SUPPORT          = 0x00000020,
    CPU_FMA4_SUPPORT            = 0x00000040,
    CPU_FMA3_SUPPORT            = 0x00000080,
    CPU_AVX_SUPPORT             = 0x00000100,
    CPU_AVX2_SUPPORT            = 0x00000200,
    CPU_AVX512F_SUPPORT         = 0x00000400,
    CPU_AVX512DQ_SUPPORT        = 0x00000800,
    CPU_AVX512IFMA52_SUPPORT    = 0x00001000,
    CPU_AVX512PF_SUPPORT        = 0x00002000,
    CPU_AVX512ER_SUPPORT        = 0x00004000,
    CPU_AVX512CD_SUPPORT        = 0x00008000,
    CPU_AVX512BW_SUPPORT        = 0x00010000,
    CPU_AVX512VL_SUPPORT        = 0x00020000,
    CPU_AVX512VBMI_SUPPORT      = 0x00040000,
    CPU_AVX512VBMI2_SUPPORT     = 0x00080000,
    CPU_AVX512VNNI_SUPPORT      = 0x00100000,
    CPU_AVX512BITALG_SUPPORT    = 0x00200000,
    CPU_AVX512VPOPCNTDQ_SUPPORT = 0x00400000,
    CPU_AVX512FP16_SUPPORT      = 0x00800000,
    CPU_AVX512BF16_SUPPORT      = 0x01000000,

};


static inline void get_cpuid(int *array, int info_type)
{
#if defined(_MSC_VER)
    __cpuid(array, info_type);
#else
    __cpuid(info_type, array[0], array[1], array[2], array[3]);
#endif
}


static inline void get_cpuid2(int *array, int info_type, int ecx)
{
#if defined(_MSC_VER)
    __cpuidex(array, info_type, ecx);
#else
    __cpuid_count(info_type, ecx, array[0], array[1], array[2], array[3]);
#endif
}

static inline int is_bit_set(int bitfield, int bit)  noexcept
{
    return bitfield & (1 << bit);
}

static uint32_t get_simd_support_info(void) noexcept
{
    uint32_t ret = 0;
    int regs[4] = {0};

    get_cpuid(regs, 0x00000001);
    if (is_bit_set(regs[3], 26)) {
        ret |= CPU_SSE2_SUPPORT;
    }
    if (is_bit_set(regs[2], 0)) {
        ret |= CPU_SSE3_SUPPORT;
    }
    if (is_bit_set(regs[2], 9)) {
        ret |= CPU_SSSE3_SUPPORT;
    }
    if (is_bit_set(regs[2], 19)) {
        ret |= CPU_SSE4_1_SUPPORT;
    }
    if (is_bit_set(regs[2], 26)) {
        ret |= CPU_SSE4_2_SUPPORT;
    }
    if (is_bit_set(regs[2], 27)) {
        if (is_bit_set(regs[2], 28)) {
            ret |= CPU_AVX_SUPPORT;
        }
        if (is_bit_set(regs[2], 12)) {
            ret |= CPU_FMA3_SUPPORT;
        }
    }

    regs[3] = 0;
    get_cpuid(regs, 0x80000001);
    if (is_bit_set(regs[3], 6)) {
        ret |= CPU_SSE4_A_SUPPORT;
    }
    if (is_bit_set(regs[3], 16)) {
        ret |= CPU_FMA4_SUPPORT;
    }

    get_cpuid(regs, 0x00000000);
    if (regs[0] < 7) {
        return ret;
    }

    get_cpuid2(regs, 0x00000007, 0);
    if (is_bit_set(regs[1], 5)) {
        ret |= CPU_AVX2_SUPPORT;
    }
    if (!is_bit_set(regs[1], 16)) {
        return ret;
    }
    else {
        ret |= CPU_AVX512F_SUPPORT;
    }
    if (is_bit_set(regs[1], 17)) {
        ret |= CPU_AVX512DQ_SUPPORT;
    }
    if (is_bit_set(regs[1], 21)) {
        ret |= CPU_AVX512IFMA52_SUPPORT;
    }
    if (is_bit_set(regs[1], 26)) {
        ret |= CPU_AVX512PF_SUPPORT;
    }
    if (is_bit_set(regs[1], 27)) {
        ret |= CPU_AVX512ER_SUPPORT;
    }
    if (is_bit_set(regs[1], 28)) {
        ret |= CPU_AVX512CD_SUPPORT;
    }
    if (is_bit_set(regs[1], 30)) {
        ret |= CPU_AVX512BW_SUPPORT;
    }
    if (is_bit_set(regs[1], 31)) {
        ret |= CPU_AVX512VL_SUPPORT;
    }
    if (is_bit_set(regs[2], 1)) {
        ret |= CPU_AVX512VBMI_SUPPORT;
    }
    if (is_bit_set(regs[2], 6)) {
        ret |= CPU_AVX512VBMI2_SUPPORT;
    }
    if (is_bit_set(regs[2], 11)) {
        ret |= CPU_AVX512VNNI_SUPPORT;
    }
    if (is_bit_set(regs[2], 12)) {
        ret |= CPU_AVX512BITALG_SUPPORT;
    }
    if (is_bit_set(regs[2], 14)) {
        ret |= CPU_AVX512VPOPCNTDQ_SUPPORT;
    }
    if (is_bit_set(regs[3], 23)) {
        ret |= CPU_AVX512FP16_SUPPORT;
    }
    if (!is_bit_set(regs[2], 1)) {
        return ret;
    }

    if (is_bit_set(regs[0], 5)) {
        ret |= CPU_AVX512BF16_SUPPORT;
    }

    return ret;
}


bool has_sse41(uint32_t info) noexcept
{
    if (info == 0) info = get_simd_support_info();
    auto requirement
        = CPU_SSE2_SUPPORT | CPU_SSE3_SUPPORT | CPU_SSSE3_SUPPORT
        | CPU_SSE4_1_SUPPORT;
    return (info & requirement) == requirement;
}

bool has_avx(uint32_t info) noexcept
{
    if (info == 0) info = get_simd_support_info();
    if (!has_sse41(info)) return false;
    return (info & CPU_AVX_SUPPORT) == CPU_AVX_SUPPORT;
}

bool has_avx2(uint32_t info) noexcept
{
    if (info == 0) info = get_simd_support_info();
    if (!has_avx(info)) return false;

    auto requirements = CPU_AVX2_SUPPORT | CPU_FMA3_SUPPORT;
    return (info & requirements) == requirements;
}

bool has_avx512(uint32_t info) noexcept
{
    if (info == 0) info = get_simd_support_info();
    if (!has_avx2(info)) return false;

    auto requirements
        = CPU_AVX512F_SUPPORT | CPU_AVX512VL_SUPPORT | CPU_AVX512BW_SUPPORT;

    return (info & requirements) == requirements;
}

bool has_avx512fp16(uint32_t info) noexcept
{
    if (info == 0) info = get_simd_support_info();
    if (!has_avx512(info)) return false;

    auto requirements = CPU_AVX512FP16_SUPPORT;

    return (info & requirements) == requirements;
}


std::vector<std::string>
split(const std::string& str, const char* separator) noexcept
{
    std::vector<std::string> dst;
    size_t offset = 0;
    std::string sep(separator);
    size_t length = sep.length();
    while (true) {
        auto pos = str.find(sep, offset);
        if (pos == std::string::npos) {
            auto t = str.substr(offset);
            if (t != "") {
                dst.push_back(t);
            }
            break;
        }
        auto t = str.substr(offset, pos - offset);
        if (t != "") {
            dst.push_back(t);
        }
        offset = pos + length;
    }
    return dst;
}

uint32_t get_halfvalue(int bits)
{
    switch (bits) {
    case 8:
        return 0x80808080;
    case 10:
        return 0x02000200;
    case 12:
        return 0x08000800;
    case 14:
        return 0x20002000;
    case 16:
        return 0x80008000;
    }
    return 0x3F000000;  // 0.5f
}


