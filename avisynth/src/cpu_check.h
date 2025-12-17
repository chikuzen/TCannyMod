#ifndef CPU_CHECK_H
#define CPU_CHECK_H

#include <cstdint>

bool has_sse41(uint32_t info = 0) noexcept;

bool has_avx(uint32_t info = 0) noexcept;

bool has_avx2(uint32_t info = 0) noexcept;

bool has_avx512(uint32_t info = 0) noexcept;

#endif // CPU_CHECK_H

