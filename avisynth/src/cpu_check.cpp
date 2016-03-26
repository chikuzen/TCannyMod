/*
    cpu_check.cpp
    
    This file is a part of TCannyMod.
    
    Copyright (C) 2016 OKA Motofumi
    
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

#include <cstdint>
#if defined(_MSC_VER)
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif


enum {
    CPU_NO_X86_SIMD          = 0x000000,
    CPU_SSE2_SUPPORT         = 0x000001,
    CPU_SSE3_SUPPORT         = 0x000002,
    CPU_SSSE3_SUPPORT        = 0x000004,
    CPU_SSE4_1_SUPPORT       = 0x000008,
    CPU_SSE4_2_SUPPORT       = 0x000010,
    CPU_SSE4_A_SUPPORT       = 0x000020,
    CPU_FMA4_SUPPORT         = 0x000040,
    CPU_FMA3_SUPPORT         = 0x000080,
    CPU_AVX_SUPPORT          = 0x000100,
    CPU_AVX2_SUPPORT         = 0x000200,
    CPU_AVX512F_SUPPORT      = 0x000400,
    CPU_AVX512DQ_SUPPORT     = 0x000800,
    CPU_AVX512IFMA52_SUPPORT = 0x001000,
    CPU_AVX512PF_SUPPORT     = 0x002000,
    CPU_AVX512ER_SUPPORT     = 0x004000,
    CPU_AVX512CD_SUPPORT     = 0x008000,
    CPU_AVX512BW_SUPPORT     = 0x010000,
    CPU_AVX512VL_SUPPORT     = 0x020000,
    CPU_AVX512VBMI_SUPPORT   = 0x040000,
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

static inline int is_bit_set(int bitfield, int bit)
{
    return bitfield & (1 << bit);
}

static uint32_t get_simd_support_info(void)
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

    ret |= CPU_AVX512F_SUPPORT;
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

    return ret;
}

int has_sse2()
{
    return !!(get_simd_support_info() & CPU_SSE2_SUPPORT);
}

int has_ssse3()
{
    return !!(get_simd_support_info() & CPU_SSSE3_SUPPORT);
}

int has_sse41()
{
    return !!(get_simd_support_info() & CPU_SSE4_1_SUPPORT);
}

int has_avx()
{
    return !!(get_simd_support_info() & CPU_AVX_SUPPORT);
}

int has_avx2()
{
    return !!(get_simd_support_info() & CPU_AVX2_SUPPORT);
}
