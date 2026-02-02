// MPS Conv3D - Metal implementation of 3D convolution
// Strategy: im2col + PyTorch matmul (leverage Apple's optimized GEMM)

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <mutex>
#include <atomic>

// Thread-safe Metal state
static std::mutex g_init_mutex;
static std::atomic<bool> g_initialized{false};

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_im2col3d_fp32 = nil;
static id<MTLComputePipelineState> g_im2col3d_fp16 = nil;
static id<MTLComputePipelineState> g_im2col3d_bf16 = nil;
static id<MTLComputePipelineState> g_col2im3d_fp32 = nil;

// Helper to check for integer overflow
static void check_size_overflow(int64_t value, const char* name) {
    constexpr int64_t INT32_LIMIT = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    TORCH_CHECK(value <= INT32_LIMIT,
        name, " (", value, ") exceeds int32 limit for Metal kernel dispatch");
}

static const char* METAL_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

// im2col3d: Extract 3D patches into column matrix for GEMM
// Input: (N, C_in, D, H, W)
// Output: (N * D_out * H_out * W_out, C_in * kD * kH * kW)
// Each thread handles one output spatial position
kernel void im2col3d_fp32(
    device const float* input [[buffer(0)]],
    device float* col [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& in_channels [[buffer(3)]],
    constant int& in_depth [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_depth [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_d [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_d [[buffer(13)]],
    constant int& stride_h [[buffer(14)]],
    constant int& stride_w [[buffer(15)]],
    constant int& pad_d [[buffer(16)]],
    constant int& pad_h [[buffer(17)]],
    constant int& pad_w [[buffer(18)]],
    constant int& dilation_d [[buffer(19)]],
    constant int& dilation_h [[buffer(20)]],
    constant int& dilation_w [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid = linear index into (batch, out_d, out_h, out_w)
    int out_spatial = out_depth * out_height * out_width;
    int total = batch * out_spatial;
    if (int(gid) >= total) return;

    int b = gid / out_spatial;
    int spatial_idx = gid % out_spatial;
    int od = spatial_idx / (out_height * out_width);
    int oh = (spatial_idx / out_width) % out_height;
    int ow = spatial_idx % out_width;

    int col_width = in_channels * kernel_d * kernel_h * kernel_w;
    int col_row = gid;  // Row in column matrix

    // Input base for this batch
    int input_batch_offset = b * in_channels * in_depth * in_height * in_width;

    // Fill one row of column matrix
    int col_idx = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;

                    float val = 0.0f;
                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int input_idx = input_batch_offset +
                                       ic * in_depth * in_height * in_width +
                                       id * in_height * in_width +
                                       ih * in_width + iw;
                        val = input[input_idx];
                    }

                    col[col_row * col_width + col_idx] = val;
                    col_idx++;
                }
            }
        }
    }
}

kernel void im2col3d_fp16(
    device const half* input [[buffer(0)]],
    device half* col [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& in_channels [[buffer(3)]],
    constant int& in_depth [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_depth [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_d [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_d [[buffer(13)]],
    constant int& stride_h [[buffer(14)]],
    constant int& stride_w [[buffer(15)]],
    constant int& pad_d [[buffer(16)]],
    constant int& pad_h [[buffer(17)]],
    constant int& pad_w [[buffer(18)]],
    constant int& dilation_d [[buffer(19)]],
    constant int& dilation_h [[buffer(20)]],
    constant int& dilation_w [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    int out_spatial = out_depth * out_height * out_width;
    int total = batch * out_spatial;
    if (int(gid) >= total) return;

    int b = gid / out_spatial;
    int spatial_idx = gid % out_spatial;
    int od = spatial_idx / (out_height * out_width);
    int oh = (spatial_idx / out_width) % out_height;
    int ow = spatial_idx % out_width;

    int col_width = in_channels * kernel_d * kernel_h * kernel_w;
    int col_row = gid;

    int input_batch_offset = b * in_channels * in_depth * in_height * in_width;

    int col_idx = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;

                    half val = 0.0h;
                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int input_idx = input_batch_offset +
                                       ic * in_depth * in_height * in_width +
                                       id * in_height * in_width +
                                       ih * in_width + iw;
                        val = input[input_idx];
                    }

                    col[col_row * col_width + col_idx] = val;
                    col_idx++;
                }
            }
        }
    }
}

// im2col3d BF16 - native bfloat16 with FP32 accumulation for precision
kernel void im2col3d_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* col [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& in_channels [[buffer(3)]],
    constant int& in_depth [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_depth [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_d [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_d [[buffer(13)]],
    constant int& stride_h [[buffer(14)]],
    constant int& stride_w [[buffer(15)]],
    constant int& pad_d [[buffer(16)]],
    constant int& pad_h [[buffer(17)]],
    constant int& pad_w [[buffer(18)]],
    constant int& dilation_d [[buffer(19)]],
    constant int& dilation_h [[buffer(20)]],
    constant int& dilation_w [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    int out_spatial = out_depth * out_height * out_width;
    int total = batch * out_spatial;
    if (int(gid) >= total) return;

    int b = gid / out_spatial;
    int spatial_idx = gid % out_spatial;
    int od = spatial_idx / (out_height * out_width);
    int oh = (spatial_idx / out_width) % out_height;
    int ow = spatial_idx % out_width;

    int col_width = in_channels * kernel_d * kernel_h * kernel_w;
    int col_row = gid;

    int input_batch_offset = b * in_channels * in_depth * in_height * in_width;

    int col_idx = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;

                    bfloat val = bfloat(0);
                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int input_idx = input_batch_offset +
                                       ic * in_depth * in_height * in_width +
                                       id * in_height * in_width +
                                       ih * in_width + iw;
                        val = input[input_idx];
                    }

                    col[col_row * col_width + col_idx] = val;
                    col_idx++;
                }
            }
        }
    }
}

// col2im3d: Scatter column matrix back to image (for backward)
// Uses native atomic_float for optimal performance on Apple Silicon
kernel void col2im3d_fp32(
    device const float* col [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& in_channels [[buffer(3)]],
    constant int& in_depth [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_depth [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_d [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_d [[buffer(13)]],
    constant int& stride_h [[buffer(14)]],
    constant int& stride_w [[buffer(15)]],
    constant int& pad_d [[buffer(16)]],
    constant int& pad_h [[buffer(17)]],
    constant int& pad_w [[buffer(18)]],
    constant int& dilation_d [[buffer(19)]],
    constant int& dilation_h [[buffer(20)]],
    constant int& dilation_w [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    int out_spatial = out_depth * out_height * out_width;
    int total = batch * out_spatial;
    if (int(gid) >= total) return;

    int b = gid / out_spatial;
    int spatial_idx = gid % out_spatial;
    int od = spatial_idx / (out_height * out_width);
    int oh = (spatial_idx / out_width) % out_height;
    int ow = spatial_idx % out_width;

    int col_width = in_channels * kernel_d * kernel_h * kernel_w;
    int col_row = gid;

    int output_batch_offset = b * in_channels * in_depth * in_height * in_width;

    int col_idx = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;

                    if (id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        int output_idx = output_batch_offset +
                                        ic * in_depth * in_height * in_width +
                                        id * in_height * in_width +
                                        ih * in_width + iw;
                        float val = col[col_row * col_width + col_idx];
                        atomic_fetch_add_explicit(&output[output_idx], val, memory_order_relaxed);
                    }
                    col_idx++;
                }
            }
        }
    }
}
)";

static void ensure_initialized() {
    // Fast path: already initialized
    if (g_initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Slow path: need to initialize
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Double-check after acquiring lock
    if (g_initialized.load(std::memory_order_relaxed)) {
        return;
    }

    g_device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(g_device != nil, "Failed to create Metal device");

    NSError* error = nil;

    NSString* source = [NSString stringWithUTF8String:METAL_SHADER];
    g_library = [g_device newLibraryWithSource:source options:nil error:&error];

    if (error) {
        NSLog(@"Failed to compile Metal shader: %@", error);
        throw std::runtime_error("Failed to compile Metal shader");
    }

    id<MTLFunction> im2col_fp32 = [g_library newFunctionWithName:@"im2col3d_fp32"];
    id<MTLFunction> im2col_fp16 = [g_library newFunctionWithName:@"im2col3d_fp16"];
    id<MTLFunction> im2col_bf16 = [g_library newFunctionWithName:@"im2col3d_bf16"];
    id<MTLFunction> col2im_fp32 = [g_library newFunctionWithName:@"col2im3d_fp32"];

    g_im2col3d_fp32 = [g_device newComputePipelineStateWithFunction:im2col_fp32 error:&error];
    g_im2col3d_fp16 = [g_device newComputePipelineStateWithFunction:im2col_fp16 error:&error];
    g_im2col3d_bf16 = [g_device newComputePipelineStateWithFunction:im2col_bf16 error:&error];
    g_col2im3d_fp32 = [g_device newComputePipelineStateWithFunction:col2im_fp32 error:&error];

    TORCH_CHECK(g_im2col3d_fp32 != nil, "Failed to create im2col3d_fp32 pipeline");
    TORCH_CHECK(g_im2col3d_fp16 != nil, "Failed to create im2col3d_fp16 pipeline");
    TORCH_CHECK(g_im2col3d_bf16 != nil, "Failed to create im2col3d_bf16 pipeline");
    TORCH_CHECK(g_col2im3d_fp32 != nil, "Failed to create col2im3d_fp32 pipeline");

    g_initialized.store(true, std::memory_order_release);
}

// im2col3d: Metal kernel to extract patches
torch::Tensor im2col3d_mps(
    const torch::Tensor& input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    ensure_initialized();

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    int col_height = batch * out_depth * out_height * out_width;
    int col_width = in_channels * kernel_d * kernel_h * kernel_w;

    // Native support for FP32, FP16, BF16 - no conversion needed
    auto input_work = input.contiguous();
    auto col = torch::empty({col_height, col_width}, input_work.options());

    id<MTLBuffer> input_buf = at::native::mps::getMTLBufferStorage(input_work);
    id<MTLBuffer> col_buf = at::native::mps::getMTLBufferStorage(col);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Select kernel based on dtype - native support for all types
        id<MTLComputePipelineState> pso;
        if (input_work.scalar_type() == at::kHalf) {
            pso = g_im2col3d_fp16;
        } else if (input_work.scalar_type() == at::kBFloat16) {
            pso = g_im2col3d_bf16;
        } else {
            pso = g_im2col3d_fp32;
        }

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:input_buf offset:input_work.storage_offset() * input_work.element_size() atIndex:0];
        [encoder setBuffer:col_buf offset:0 atIndex:1];

        [encoder setBytes:&batch length:sizeof(int) atIndex:2];
        [encoder setBytes:&in_channels length:sizeof(int) atIndex:3];
        [encoder setBytes:&in_depth length:sizeof(int) atIndex:4];
        [encoder setBytes:&in_height length:sizeof(int) atIndex:5];
        [encoder setBytes:&in_width length:sizeof(int) atIndex:6];
        [encoder setBytes:&out_depth length:sizeof(int) atIndex:7];
        [encoder setBytes:&out_height length:sizeof(int) atIndex:8];
        [encoder setBytes:&out_width length:sizeof(int) atIndex:9];
        [encoder setBytes:&kernel_d length:sizeof(int) atIndex:10];
        [encoder setBytes:&kernel_h length:sizeof(int) atIndex:11];
        [encoder setBytes:&kernel_w length:sizeof(int) atIndex:12];
        [encoder setBytes:&stride_d length:sizeof(int) atIndex:13];
        [encoder setBytes:&stride_h length:sizeof(int) atIndex:14];
        [encoder setBytes:&stride_w length:sizeof(int) atIndex:15];
        [encoder setBytes:&pad_d length:sizeof(int) atIndex:16];
        [encoder setBytes:&pad_h length:sizeof(int) atIndex:17];
        [encoder setBytes:&pad_w length:sizeof(int) atIndex:18];
        [encoder setBytes:&dilation_d length:sizeof(int) atIndex:19];
        [encoder setBytes:&dilation_h length:sizeof(int) atIndex:20];
        [encoder setBytes:&dilation_w length:sizeof(int) atIndex:21];

        MTLSize gridSize = MTLSizeMake(col_height, 1, 1);
        NSUInteger threadGroupSize = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }

    return col;
}

// Forward: im2col + matmul
torch::Tensor conv3d_forward_mps(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    TORCH_CHECK(input.device().type() == torch::kMPS, "input must be on MPS");
    TORCH_CHECK(weight.device().type() == torch::kMPS, "weight must be on MPS");
    TORCH_CHECK(groups == 1, "groups > 1 not yet supported in im2col path");

    // Parameter validation
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0,
        "stride must be positive, got (", stride_d, ", ", stride_h, ", ", stride_w, ")");
    TORCH_CHECK(dilation_d > 0 && dilation_h > 0 && dilation_w > 0,
        "dilation must be positive, got (", dilation_d, ", ", dilation_h, ", ", dilation_w, ")");
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0,
        "padding must be non-negative, got (", pad_d, ", ", pad_h, ", ", pad_w, ")");

    int batch = input.size(0);
    int out_channels = weight.size(0);
    int in_channels = weight.size(1);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // im2col: (B*D_out*H_out*W_out, C_in*kD*kH*kW)
    auto col = im2col3d_mps(input, kernel_d, kernel_h, kernel_w,
                            stride_d, stride_h, stride_w,
                            pad_d, pad_h, pad_w,
                            dilation_d, dilation_h, dilation_w);

    // Weight: (C_out, C_in*kD*kH*kW) -> transpose for matmul
    auto weight_col = weight.view({out_channels, -1});  // (C_out, C_in*k*k*k)

    // Matmul: (B*D*H*W, C_in*k*k*k) @ (C_in*k*k*k, C_out) = (B*D*H*W, C_out)
    auto output_col = torch::mm(col, weight_col.t());

    // Reshape: (B, D_out, H_out, W_out, C_out) -> (B, C_out, D_out, H_out, W_out)
    auto output = output_col.view({batch, out_depth, out_height, out_width, out_channels});
    output = output.permute({0, 4, 1, 2, 3}).contiguous();

    return output;
}

// Backward input: matmul + col2im
torch::Tensor conv3d_backward_input_mps(
    const torch::Tensor& grad_output,
    const torch::Tensor& weight,
    std::vector<int64_t> input_shape,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    ensure_initialized();
    TORCH_CHECK(groups == 1, "groups > 1 not yet supported");

    int batch = input_shape[0];
    int in_channels = input_shape[1];
    int in_depth = input_shape[2];
    int in_height = input_shape[3];
    int in_width = input_shape[4];

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = grad_output.size(2);
    int out_height = grad_output.size(3);
    int out_width = grad_output.size(4);

    // grad_output: (B, C_out, D_out, H_out, W_out) -> (B*D*H*W, C_out)
    auto grad_out_col = grad_output.permute({0, 2, 3, 4, 1}).contiguous();
    grad_out_col = grad_out_col.view({-1, out_channels});

    // Weight: (C_out, C_in*k*k*k)
    auto weight_col = weight.view({out_channels, -1});

    // grad_col = grad_out_col @ weight_col: (B*D*H*W, C_in*k*k*k)
    auto grad_col = torch::mm(grad_out_col.to(at::kFloat), weight_col.to(at::kFloat));

    // col2im: scatter back to input shape
    auto grad_input = torch::zeros(input_shape, grad_col.options());

    int col_height = batch * out_depth * out_height * out_width;

    id<MTLBuffer> col_buf = at::native::mps::getMTLBufferStorage(grad_col);
    id<MTLBuffer> output_buf = at::native::mps::getMTLBufferStorage(grad_input);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:g_col2im3d_fp32];
        [encoder setBuffer:col_buf offset:0 atIndex:0];
        [encoder setBuffer:output_buf offset:0 atIndex:1];

        [encoder setBytes:&batch length:sizeof(int) atIndex:2];
        [encoder setBytes:&in_channels length:sizeof(int) atIndex:3];
        [encoder setBytes:&in_depth length:sizeof(int) atIndex:4];
        [encoder setBytes:&in_height length:sizeof(int) atIndex:5];
        [encoder setBytes:&in_width length:sizeof(int) atIndex:6];
        [encoder setBytes:&out_depth length:sizeof(int) atIndex:7];
        [encoder setBytes:&out_height length:sizeof(int) atIndex:8];
        [encoder setBytes:&out_width length:sizeof(int) atIndex:9];
        [encoder setBytes:&kernel_d length:sizeof(int) atIndex:10];
        [encoder setBytes:&kernel_h length:sizeof(int) atIndex:11];
        [encoder setBytes:&kernel_w length:sizeof(int) atIndex:12];
        [encoder setBytes:&stride_d length:sizeof(int) atIndex:13];
        [encoder setBytes:&stride_h length:sizeof(int) atIndex:14];
        [encoder setBytes:&stride_w length:sizeof(int) atIndex:15];
        [encoder setBytes:&pad_d length:sizeof(int) atIndex:16];
        [encoder setBytes:&pad_h length:sizeof(int) atIndex:17];
        [encoder setBytes:&pad_w length:sizeof(int) atIndex:18];
        [encoder setBytes:&dilation_d length:sizeof(int) atIndex:19];
        [encoder setBytes:&dilation_h length:sizeof(int) atIndex:20];
        [encoder setBytes:&dilation_w length:sizeof(int) atIndex:21];

        MTLSize gridSize = MTLSizeMake(col_height, 1, 1);
        MTLSize tgSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    }

    return grad_input.to(grad_output.scalar_type());
}

// Backward weight: im2col.T @ grad_output
torch::Tensor conv3d_backward_weight_mps(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    std::vector<int64_t> weight_shape,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    TORCH_CHECK(groups == 1, "groups > 1 not yet supported");

    int out_channels = weight_shape[0];
    int kernel_d = weight_shape[2];
    int kernel_h = weight_shape[3];
    int kernel_w = weight_shape[4];

    // im2col on input
    auto col = im2col3d_mps(input.to(at::kFloat), kernel_d, kernel_h, kernel_w,
                            stride_d, stride_h, stride_w,
                            pad_d, pad_h, pad_w,
                            dilation_d, dilation_h, dilation_w);

    // grad_output: (B, C_out, D_out, H_out, W_out) -> (B*D*H*W, C_out)
    auto grad_out_col = grad_output.permute({0, 2, 3, 4, 1}).contiguous();
    grad_out_col = grad_out_col.view({-1, out_channels}).to(at::kFloat);

    // grad_weight = grad_out_col.T @ col: (C_out, C_in*k*k*k)
    auto grad_weight_col = torch::mm(grad_out_col.t(), col);

    // Reshape to weight shape
    auto grad_weight = grad_weight_col.view(weight_shape);

    return grad_weight.to(grad_output.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward", &conv3d_forward_mps, "Conv3D forward (MPS) - im2col + GEMM");
    m.def("conv3d_backward_input", &conv3d_backward_input_mps, "Conv3D backward input (MPS)");
    m.def("conv3d_backward_weight", &conv3d_backward_weight_mps, "Conv3D backward weight (MPS)");
    m.def("im2col3d", &im2col3d_mps, "im2col3d (MPS)");
}
