#include "gaussian_kernel.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

__global__ void expand_gaussians_to_tiles(
    const float2* uv,
    const float* depth,
    const float* alpha,
    const float3* sh,
    const float2x2* inv_cov,
    int num_gaussians,
    int W, int H,
    GaussianInstance* output_instances,
    int* instance_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_gaussians) return;

    float2 center = uv[i];
    float d = depth[i];
    float a = alpha[i];
    const float3* sh_ptr = &sh[i * SH_COEFFS];
    float2x2 icov = inv_cov[i];

    float a11 = icov.row[0].x, a12 = icov.row[0].y;
    float a21 = icov.row[1].x, a22 = icov.row[1].y;
    float det = a11 * a22 - a12 * a21;
    if (det == 0) return;

    float cov11 = a22 / det;
    float cov22 = a11 / det;

    float radius_x = 3.f * sqrtf(fabsf(cov11));
    float radius_y = 3.f * sqrtf(fabsf(cov22));

    int tile_x_min = max(0, int((center.x - radius_x) / TILE_SIZE));
    int tile_x_max = min(W / TILE_SIZE - 1, int((center.x + radius_x) / TILE_SIZE));
    int tile_y_min = max(0, int((center.y - radius_y) / TILE_SIZE));
    int tile_y_max = min(H / TILE_SIZE - 1, int((center.y + radius_y) / TILE_SIZE));

    for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
        for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
            int tile_id = ty * (W / TILE_SIZE) + tx;
            int idx = atomicAdd(instance_count, 1);
            GaussianInstance& out = output_instances[idx];
            out.uv = center;
            out.depth = d;
            out.alpha = a;
            out.inv_cov = icov;
            out.tile_id = tile_id;
            for (int j = 0; j < SH_COEFFS; ++j)
                out.sh[j] = sh_ptr[j];
            uint32_t depth_bits = float_to_sortable_uint(d);
            out.sort_key = ((uint64_t)tile_id << 32) | depth_bits;
        }
    }
}

extern "C"
void launch_expand_and_sort(
    const float2* d_uv,
    const float* d_depth,
    const float* d_alpha,
    const float3* d_sh,
    const float2x2* d_inv_cov,
    int num_gaussians,
    int W,
    int H,
    GaussianInstance* d_output_instances,
    int* d_instance_count
) {
    cudaMemset(d_instance_count, 0, sizeof(int));
    int threads = 128;
    int blocks = (num_gaussians + threads - 1) / threads;

    expand_gaussians_to_tiles<<<blocks, threads>>>(
        d_uv, d_depth, d_alpha, d_sh, d_inv_cov,
        num_gaussians, W, H,
        d_output_instances, d_instance_count
    );
    cudaDeviceSynchronize();

    int h_count;
    cudaMemcpy(&h_count, d_instance_count, sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_ptr<GaussianInstance> instances_ptr(d_output_instances);
    thrust::sort(instances_ptr, instances_ptr + h_count,
        [] __device__ (const GaussianInstance& a, const GaussianInstance& b) {
            return a.sort_key < b.sort_key;
        });
}
