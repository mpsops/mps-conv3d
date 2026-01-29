// MPS Conv3D - Metal implementation of 3D convolution
// Used in video models: Synchformer, I3D, SlowFast, C3D, etc.

#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_conv3d_forward_fp32 = nil;
static id<MTLComputePipelineState> g_conv3d_forward_fp16 = nil;
static id<MTLComputePipelineState> g_conv3d_backward_input_fp32 = nil;
static id<MTLComputePipelineState> g_conv3d_backward_weight_fp32 = nil;

static const char* METAL_SHADER = R"(
#include <metal_stdlib>
using namespace metal;

// Conv3D forward kernel
// input: (N, C_in, D, H, W)
// weight: (C_out, C_in/groups, kD, kH, kW)
// output: (N, C_out, D_out, H_out, W_out)
kernel void conv3d_forward_fp32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& in_channels [[buffer(4)]],
    constant int& in_depth [[buffer(5)]],
    constant int& in_height [[buffer(6)]],
    constant int& in_width [[buffer(7)]],
    constant int& out_channels [[buffer(8)]],
    constant int& out_depth [[buffer(9)]],
    constant int& out_height [[buffer(10)]],
    constant int& out_width [[buffer(11)]],
    constant int& kernel_d [[buffer(12)]],
    constant int& kernel_h [[buffer(13)]],
    constant int& kernel_w [[buffer(14)]],
    constant int& stride_d [[buffer(15)]],
    constant int& stride_h [[buffer(16)]],
    constant int& stride_w [[buffer(17)]],
    constant int& pad_d [[buffer(18)]],
    constant int& pad_h [[buffer(19)]],
    constant int& pad_w [[buffer(20)]],
    constant int& dilation_d [[buffer(21)]],
    constant int& dilation_h [[buffer(22)]],
    constant int& dilation_w [[buffer(23)]],
    constant int& groups [[buffer(24)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // gid.x = output width position
    // gid.y = output height position
    // gid.z = batch * out_channels * out_depth

    int ow = gid.x;
    int oh = gid.y;
    int bcd = gid.z;

    int od = bcd % out_depth;
    int bc = bcd / out_depth;
    int oc = bc % out_channels;
    int b = bc / out_channels;

    if (ow >= out_width || oh >= out_height || b >= batch) return;

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = oc / group_out_channels;
    int oc_in_group = oc % group_out_channels;

    float sum = 0.0f;

    for (int ic = 0; ic < group_in_channels; ic++) {
        int actual_ic = g * group_in_channels + ic;

        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            if (id < 0 || id >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                if (ih < 0 || ih >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;
                    if (iw < 0 || iw >= in_width) continue;

                    int input_idx = ((b * in_channels + actual_ic) * in_depth + id) * in_height * in_width +
                                   ih * in_width + iw;

                    int weight_idx = ((oc * group_in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width +
                     oh * out_width + ow;
    output[output_idx] = sum;
}

kernel void conv3d_forward_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& in_channels [[buffer(4)]],
    constant int& in_depth [[buffer(5)]],
    constant int& in_height [[buffer(6)]],
    constant int& in_width [[buffer(7)]],
    constant int& out_channels [[buffer(8)]],
    constant int& out_depth [[buffer(9)]],
    constant int& out_height [[buffer(10)]],
    constant int& out_width [[buffer(11)]],
    constant int& kernel_d [[buffer(12)]],
    constant int& kernel_h [[buffer(13)]],
    constant int& kernel_w [[buffer(14)]],
    constant int& stride_d [[buffer(15)]],
    constant int& stride_h [[buffer(16)]],
    constant int& stride_w [[buffer(17)]],
    constant int& pad_d [[buffer(18)]],
    constant int& pad_h [[buffer(19)]],
    constant int& pad_w [[buffer(20)]],
    constant int& dilation_d [[buffer(21)]],
    constant int& dilation_h [[buffer(22)]],
    constant int& dilation_w [[buffer(23)]],
    constant int& groups [[buffer(24)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int ow = gid.x;
    int oh = gid.y;
    int bcd = gid.z;

    int od = bcd % out_depth;
    int bc = bcd / out_depth;
    int oc = bc % out_channels;
    int b = bc / out_channels;

    if (ow >= out_width || oh >= out_height || b >= batch) return;

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = oc / group_out_channels;
    int oc_in_group = oc % group_out_channels;

    float sum = 0.0f;

    for (int ic = 0; ic < group_in_channels; ic++) {
        int actual_ic = g * group_in_channels + ic;

        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            if (id < 0 || id >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                if (ih < 0 || ih >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;
                    if (iw < 0 || iw >= in_width) continue;

                    int input_idx = ((b * in_channels + actual_ic) * in_depth + id) * in_height * in_width +
                                   ih * in_width + iw;

                    int weight_idx = ((oc * group_in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    sum += float(input[input_idx]) * float(weight[weight_idx]);
                }
            }
        }
    }

    int output_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width +
                     oh * out_width + ow;
    output[output_idx] = half(sum);
}

// Backward for input - transposed convolution
kernel void conv3d_backward_input_fp32(
    device const float* grad_output [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device atomic_float* grad_input [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& in_channels [[buffer(4)]],
    constant int& in_depth [[buffer(5)]],
    constant int& in_height [[buffer(6)]],
    constant int& in_width [[buffer(7)]],
    constant int& out_channels [[buffer(8)]],
    constant int& out_depth [[buffer(9)]],
    constant int& out_height [[buffer(10)]],
    constant int& out_width [[buffer(11)]],
    constant int& kernel_d [[buffer(12)]],
    constant int& kernel_h [[buffer(13)]],
    constant int& kernel_w [[buffer(14)]],
    constant int& stride_d [[buffer(15)]],
    constant int& stride_h [[buffer(16)]],
    constant int& stride_w [[buffer(17)]],
    constant int& pad_d [[buffer(18)]],
    constant int& pad_h [[buffer(19)]],
    constant int& pad_w [[buffer(20)]],
    constant int& dilation_d [[buffer(21)]],
    constant int& dilation_h [[buffer(22)]],
    constant int& dilation_w [[buffer(23)]],
    constant int& groups [[buffer(24)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Iterate over output positions and scatter gradients to input
    int ow = gid.x;
    int oh = gid.y;
    int bcd = gid.z;

    int od = bcd % out_depth;
    int bc = bcd / out_depth;
    int oc = bc % out_channels;
    int b = bc / out_channels;

    if (ow >= out_width || oh >= out_height || b >= batch) return;

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = oc / group_out_channels;

    int grad_out_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width +
                       oh * out_width + ow;
    float grad_out_val = grad_output[grad_out_idx];

    for (int ic = 0; ic < group_in_channels; ic++) {
        int actual_ic = g * group_in_channels + ic;

        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            if (id < 0 || id >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                if (ih < 0 || ih >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;
                    if (iw < 0 || iw >= in_width) continue;

                    int weight_idx = ((oc * group_in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    int input_idx = ((b * in_channels + actual_ic) * in_depth + id) * in_height * in_width +
                                   ih * in_width + iw;

                    atomic_fetch_add_explicit(&grad_input[input_idx], grad_out_val * weight[weight_idx], memory_order_relaxed);
                }
            }
        }
    }
}

// Backward for weight
kernel void conv3d_backward_weight_fp32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device atomic_float* grad_weight [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& in_channels [[buffer(4)]],
    constant int& in_depth [[buffer(5)]],
    constant int& in_height [[buffer(6)]],
    constant int& in_width [[buffer(7)]],
    constant int& out_channels [[buffer(8)]],
    constant int& out_depth [[buffer(9)]],
    constant int& out_height [[buffer(10)]],
    constant int& out_width [[buffer(11)]],
    constant int& kernel_d [[buffer(12)]],
    constant int& kernel_h [[buffer(13)]],
    constant int& kernel_w [[buffer(14)]],
    constant int& stride_d [[buffer(15)]],
    constant int& stride_h [[buffer(16)]],
    constant int& stride_w [[buffer(17)]],
    constant int& pad_d [[buffer(18)]],
    constant int& pad_h [[buffer(19)]],
    constant int& pad_w [[buffer(20)]],
    constant int& dilation_d [[buffer(21)]],
    constant int& dilation_h [[buffer(22)]],
    constant int& dilation_w [[buffer(23)]],
    constant int& groups [[buffer(24)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Iterate over output positions and accumulate weight gradients
    int ow = gid.x;
    int oh = gid.y;
    int bcd = gid.z;

    int od = bcd % out_depth;
    int bc = bcd / out_depth;
    int oc = bc % out_channels;
    int b = bc / out_channels;

    if (ow >= out_width || oh >= out_height || b >= batch) return;

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = oc / group_out_channels;

    int grad_out_idx = ((b * out_channels + oc) * out_depth + od) * out_height * out_width +
                       oh * out_width + ow;
    float grad_out_val = grad_output[grad_out_idx];

    for (int ic = 0; ic < group_in_channels; ic++) {
        int actual_ic = g * group_in_channels + ic;

        for (int kd = 0; kd < kernel_d; kd++) {
            int id = od * stride_d - pad_d + kd * dilation_d;
            if (id < 0 || id >= in_depth) continue;

            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                if (ih < 0 || ih >= in_height) continue;

                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw * dilation_w;
                    if (iw < 0 || iw >= in_width) continue;

                    int input_idx = ((b * in_channels + actual_ic) * in_depth + id) * in_height * in_width +
                                   ih * in_width + iw;

                    int weight_idx = ((oc * group_in_channels + ic) * kernel_d + kd) * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                    atomic_fetch_add_explicit(&grad_weight[weight_idx], grad_out_val * input[input_idx], memory_order_relaxed);
                }
            }
        }
    }
}
)";

static void ensure_initialized() {
    if (g_device != nil) return;

    g_device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;

    NSString* source = [NSString stringWithUTF8String:METAL_SHADER];
    g_library = [g_device newLibraryWithSource:source options:nil error:&error];

    if (error) {
        NSLog(@"Failed to compile Metal shader: %@", error);
        throw std::runtime_error("Failed to compile Metal shader");
    }

    id<MTLFunction> fwd_fp32 = [g_library newFunctionWithName:@"conv3d_forward_fp32"];
    id<MTLFunction> fwd_fp16 = [g_library newFunctionWithName:@"conv3d_forward_fp16"];
    id<MTLFunction> bwd_input = [g_library newFunctionWithName:@"conv3d_backward_input_fp32"];
    id<MTLFunction> bwd_weight = [g_library newFunctionWithName:@"conv3d_backward_weight_fp32"];

    g_conv3d_forward_fp32 = [g_device newComputePipelineStateWithFunction:fwd_fp32 error:&error];
    g_conv3d_forward_fp16 = [g_device newComputePipelineStateWithFunction:fwd_fp16 error:&error];
    g_conv3d_backward_input_fp32 = [g_device newComputePipelineStateWithFunction:bwd_input error:&error];
    g_conv3d_backward_weight_fp32 = [g_device newComputePipelineStateWithFunction:bwd_weight error:&error];
}

torch::Tensor conv3d_forward_mps(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    ensure_initialized();

    TORCH_CHECK(input.device().type() == torch::kMPS, "input must be on MPS");
    TORCH_CHECK(weight.device().type() == torch::kMPS, "weight must be on MPS");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int out_depth = (in_depth + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch, out_channels, out_depth, out_height, out_width}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        id<MTLCommandBuffer> cmdBuf = [stream->commandQueue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        bool is_fp16 = input_contig.scalar_type() == torch::kHalf;
        id<MTLComputePipelineState> pso = is_fp16 ? g_conv3d_forward_fp16 : g_conv3d_forward_fp32;

        [encoder setComputePipelineState:pso];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_contig) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_contig) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:2];

        [encoder setBytes:&batch length:sizeof(int) atIndex:3];
        [encoder setBytes:&in_channels length:sizeof(int) atIndex:4];
        [encoder setBytes:&in_depth length:sizeof(int) atIndex:5];
        [encoder setBytes:&in_height length:sizeof(int) atIndex:6];
        [encoder setBytes:&in_width length:sizeof(int) atIndex:7];
        [encoder setBytes:&out_channels length:sizeof(int) atIndex:8];
        [encoder setBytes:&out_depth length:sizeof(int) atIndex:9];
        [encoder setBytes:&out_height length:sizeof(int) atIndex:10];
        [encoder setBytes:&out_width length:sizeof(int) atIndex:11];
        [encoder setBytes:&kernel_d length:sizeof(int) atIndex:12];
        [encoder setBytes:&kernel_h length:sizeof(int) atIndex:13];
        [encoder setBytes:&kernel_w length:sizeof(int) atIndex:14];
        [encoder setBytes:&stride_d length:sizeof(int) atIndex:15];
        [encoder setBytes:&stride_h length:sizeof(int) atIndex:16];
        [encoder setBytes:&stride_w length:sizeof(int) atIndex:17];
        [encoder setBytes:&pad_d length:sizeof(int) atIndex:18];
        [encoder setBytes:&pad_h length:sizeof(int) atIndex:19];
        [encoder setBytes:&pad_w length:sizeof(int) atIndex:20];
        [encoder setBytes:&dilation_d length:sizeof(int) atIndex:21];
        [encoder setBytes:&dilation_h length:sizeof(int) atIndex:22];
        [encoder setBytes:&dilation_w length:sizeof(int) atIndex:23];
        [encoder setBytes:&groups length:sizeof(int) atIndex:24];

        MTLSize gridSize = MTLSizeMake(out_width, out_height, batch * out_channels * out_depth);
        MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    return output;
}

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

    int batch = input_shape[0];
    int in_channels = input_shape[1];
    int in_depth = input_shape[2];
    int in_height = input_shape[3];
    int in_width = input_shape[4];

    int out_channels = weight.size(0);
    int out_depth = grad_output.size(2);
    int out_height = grad_output.size(3);
    int out_width = grad_output.size(4);

    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    auto grad_output_f = grad_output.to(torch::kFloat32).contiguous();
    auto weight_f = weight.to(torch::kFloat32).contiguous();
    auto grad_input = torch::zeros({batch, in_channels, in_depth, in_height, in_width}, grad_output_f.options());

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        id<MTLCommandBuffer> cmdBuf = [stream->commandQueue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:g_conv3d_backward_input_fp32];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_output_f) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_f) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_input) offset:0 atIndex:2];

        [encoder setBytes:&batch length:sizeof(int) atIndex:3];
        [encoder setBytes:&in_channels length:sizeof(int) atIndex:4];
        [encoder setBytes:&in_depth length:sizeof(int) atIndex:5];
        [encoder setBytes:&in_height length:sizeof(int) atIndex:6];
        [encoder setBytes:&in_width length:sizeof(int) atIndex:7];
        [encoder setBytes:&out_channels length:sizeof(int) atIndex:8];
        [encoder setBytes:&out_depth length:sizeof(int) atIndex:9];
        [encoder setBytes:&out_height length:sizeof(int) atIndex:10];
        [encoder setBytes:&out_width length:sizeof(int) atIndex:11];
        [encoder setBytes:&kernel_d length:sizeof(int) atIndex:12];
        [encoder setBytes:&kernel_h length:sizeof(int) atIndex:13];
        [encoder setBytes:&kernel_w length:sizeof(int) atIndex:14];
        [encoder setBytes:&stride_d length:sizeof(int) atIndex:15];
        [encoder setBytes:&stride_h length:sizeof(int) atIndex:16];
        [encoder setBytes:&stride_w length:sizeof(int) atIndex:17];
        [encoder setBytes:&pad_d length:sizeof(int) atIndex:18];
        [encoder setBytes:&pad_h length:sizeof(int) atIndex:19];
        [encoder setBytes:&pad_w length:sizeof(int) atIndex:20];
        [encoder setBytes:&dilation_d length:sizeof(int) atIndex:21];
        [encoder setBytes:&dilation_h length:sizeof(int) atIndex:22];
        [encoder setBytes:&dilation_w length:sizeof(int) atIndex:23];
        [encoder setBytes:&groups length:sizeof(int) atIndex:24];

        MTLSize gridSize = MTLSizeMake(out_width, out_height, batch * out_channels * out_depth);
        MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    return grad_input.to(grad_output.scalar_type());
}

torch::Tensor conv3d_backward_weight_mps(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    std::vector<int64_t> weight_shape,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    ensure_initialized();

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight_shape[0];
    int out_depth = grad_output.size(2);
    int out_height = grad_output.size(3);
    int out_width = grad_output.size(4);

    int kernel_d = weight_shape[2];
    int kernel_h = weight_shape[3];
    int kernel_w = weight_shape[4];

    auto grad_output_f = grad_output.to(torch::kFloat32).contiguous();
    auto input_f = input.to(torch::kFloat32).contiguous();
    auto grad_weight = torch::zeros(weight_shape, grad_output_f.options());

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        id<MTLCommandBuffer> cmdBuf = [stream->commandQueue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:g_conv3d_backward_weight_fp32];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_output_f) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_f) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_weight) offset:0 atIndex:2];

        [encoder setBytes:&batch length:sizeof(int) atIndex:3];
        [encoder setBytes:&in_channels length:sizeof(int) atIndex:4];
        [encoder setBytes:&in_depth length:sizeof(int) atIndex:5];
        [encoder setBytes:&in_height length:sizeof(int) atIndex:6];
        [encoder setBytes:&in_width length:sizeof(int) atIndex:7];
        [encoder setBytes:&out_channels length:sizeof(int) atIndex:8];
        [encoder setBytes:&out_depth length:sizeof(int) atIndex:9];
        [encoder setBytes:&out_height length:sizeof(int) atIndex:10];
        [encoder setBytes:&out_width length:sizeof(int) atIndex:11];
        [encoder setBytes:&kernel_d length:sizeof(int) atIndex:12];
        [encoder setBytes:&kernel_h length:sizeof(int) atIndex:13];
        [encoder setBytes:&kernel_w length:sizeof(int) atIndex:14];
        [encoder setBytes:&stride_d length:sizeof(int) atIndex:15];
        [encoder setBytes:&stride_h length:sizeof(int) atIndex:16];
        [encoder setBytes:&stride_w length:sizeof(int) atIndex:17];
        [encoder setBytes:&pad_d length:sizeof(int) atIndex:18];
        [encoder setBytes:&pad_h length:sizeof(int) atIndex:19];
        [encoder setBytes:&pad_w length:sizeof(int) atIndex:20];
        [encoder setBytes:&dilation_d length:sizeof(int) atIndex:21];
        [encoder setBytes:&dilation_h length:sizeof(int) atIndex:22];
        [encoder setBytes:&dilation_w length:sizeof(int) atIndex:23];
        [encoder setBytes:&groups length:sizeof(int) atIndex:24];

        MTLSize gridSize = MTLSizeMake(out_width, out_height, batch * out_channels * out_depth);
        MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    return grad_weight.to(grad_output.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_forward", &conv3d_forward_mps, "Conv3D forward (MPS)");
    m.def("conv3d_backward_input", &conv3d_backward_input_mps, "Conv3D backward input (MPS)");
    m.def("conv3d_backward_weight", &conv3d_backward_weight_mps, "Conv3D backward weight (MPS)");
}
