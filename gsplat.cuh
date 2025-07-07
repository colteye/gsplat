#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_SIZE 16
#define SH_COEFFS 9

struct float2x2 {
    float2 row[2];
};

struct GaussianInstance {
    float2 uv;
    float depth;
    float alpha;
    float3 sh[SH_COEFFS];
    float2x2 inv_cov;
    int tile_id;
    uint64_t sort_key;
};

__device__ inline uint32_t float_to_sortable_uint(float f) {
    uint32_t x = __float_as_uint(f);
    return (x & 0x80000000) ? ~x : x ^ 0x80000000;
}