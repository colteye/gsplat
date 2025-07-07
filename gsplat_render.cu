#include "gaussian_kernel.cuh"

__device__ float3 eval_sh_color(const float3 sh[SH_COEFFS]) {
    return sh[0];  // Placeholder: use constant term only
}

__global__ void splat_tiles_kernel(
    const GaussianInstance* instances,
    const int* tile_offsets,
    float3* image_out,
    float* alpha_out,
    int W, int H,
    int num_tiles_x
) {
    int tile_id = blockIdx.x;
    int start = tile_offsets[tile_id];
    int end = tile_offsets[tile_id + 1];
    if (start == end) return;

    int tid = threadIdx.x;
    int px = (tile_id % num_tiles_x) * TILE_SIZE + (tid % TILE_SIZE);
    int py = (tile_id / num_tiles_x) * TILE_SIZE + (tid / TILE_SIZE);
    if (px >= W || py >= H) return;

    float2 p_uv = make_float2(px + 0.5f, py + 0.5f);
    float3 accum_color = make_float3(0, 0, 0);
    float accum_alpha = 0;

    for (int i = start; i < end; ++i) {
        const GaussianInstance& g = instances[i];
        float2 d = make_float2(p_uv.x - g.uv.x, p_uv.y - g.uv.y);
        float dist =
            d.x * (g.inv_cov.row[0].x * d.x + g.inv_cov.row[0].y * d.y) +
            d.y * (g.inv_cov.row[1].x * d.x + g.inv_cov.row[1].y * d.y);
        if (dist > 9.f) continue;

        float weight = expf(-0.5f * dist);
        float a = g.alpha * weight * (1 - accum_alpha);
        if (a < 1e-4f) continue;
        float3 col = eval_sh_color(g.sh);
        accum_color += a * col;
        accum_alpha += a;
        if (accum_alpha > 0.995f) break;
    }

    int idx = py * W + px;
    image_out[idx] = accum_color;
    alpha_out[idx] = accum_alpha;
}

extern "C"
void launch_render(
    const GaussianInstance* instances,
    const int* tile_offsets,
    float3* image_out,
    float* alpha_out,
    int W, int H,
    int num_tiles
) {
    dim3 blocks(num_tiles);
    dim3 threads(TILE_SIZE * TILE_SIZE);
    splat_tiles_kernel<<<blocks, threads>>>(instances, tile_offsets, image_out, alpha_out, W, H, W / TILE_SIZE);
}
