#include <torch/extension.h>
#include "gaussian_kernel.cuh"

void launch_expand_and_sort(
    const float2*, const float*, const float*,
    const float3*, const float2x2*, int, int, int,
    GaussianInstance*, int*
);
void launch_render(
    const GaussianInstance*, const int*, float3*, float*, int, int, int
);

void expand_and_sort(
    at::Tensor uv, at::Tensor depth, at::Tensor alpha, at::Tensor sh, at::Tensor inv_cov,
    at::Tensor output, at::Tensor counter, int W, int H
) {
    launch_expand_and_sort(
        reinterpret_cast<float2*>(uv.data_ptr<float>()),
        depth.data_ptr<float>(), alpha.data_ptr<float>(),
        reinterpret_cast<float3*>(sh.data_ptr<float>()),
        reinterpret_cast<float2x2*>(inv_cov.data_ptr<float>()),
        uv.size(0), W, H,
        reinterpret_cast<GaussianInstance*>(output.data_ptr<uint8_t>()),
        counter.data_ptr<int>()
    );
}

void render_splatted(
    at::Tensor sorted_instances,
    at::Tensor tile_offsets,
    at::Tensor image_out,
    at::Tensor alpha_out,
    int W, int H
) {
    int num_tiles = tile_offsets.size(0) - 1;
    launch_render(
        reinterpret_cast<const GaussianInstance*>(sorted_instances.data_ptr<uint8_t>()),
        tile_offsets.data_ptr<int>(),
        reinterpret_cast<float3*>(image_out.data_ptr<float>()),
        alpha_out.data_ptr<float>(),
        W, H, num_tiles
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expand_and_sort", &expand_and_sort, "Expand + Sort");
    m.def("render_splatted", &render_splatted, "Render Tiles");
}