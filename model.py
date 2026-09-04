"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // Butterfly (xor) tree reduction across the warp
    // Each step adds values from lanes with differing bits
    
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // Butterfly (xor) tree reduction across the warp for maximum
    // Each step takes the max between current value and partner lane's value
    
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // Compute lane ID and warp ID
    int lane = threadIdx.x & 31;  // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;  // threadIdx.x / 32
    
    // Step 1: Reduce within each warp
    float warp_sum = warp_reduce_sum(val);
    
    // Step 2: Each warp's lane 0 writes its partial sum to shared memory
    if (lane == 0) {
        shared[warp_id] = warp_sum;
    }
    
    // Step 3: Synchronize to ensure all warp sums are written
    __syncthreads();
    
    // Step 4: Only warp 0 reduces the partial sums
    float block_sum = 0.0f;
    if (warp_id == 0) {
        // Number of warps in the block
        int num_warps = (blockDim.x + 31) >> 5;  // ceil(blockDim.x / 32)
        
        // Each lane of warp 0 loads one partial sum (if within bounds)
        float partial = (lane < num_warps) ? shared[lane] : 0.0f;
        
        // Reduce all partial sums across warp 0
        block_sum = warp_reduce_sum(partial);
    }
    
    return block_sum;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    // Compute lane ID and warp ID
    int lane = threadIdx.x & 31;  // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;  // threadIdx.x / 32
    
    // Step 1: Reduce within each warp
    float warp_max = warp_reduce_max(val);
    
    // Step 2: Each warp's lane 0 writes its max to shared memory
    if (lane == 0) {
        shared[warp_id] = warp_max;
    }
    
    // Step 3: Synchronize to ensure all warp maxima are written
    __syncthreads();
    
    // Step 4: Only warp 0 reduces the partial maxima
    float block_max = -INFINITY;
    if (warp_id == 0) {
        // Number of warps in the block
        int num_warps = (blockDim.x + 31) >> 5;  // ceil(blockDim.x / 32)
        
        // Each lane of warp 0 loads one partial max (if within bounds)
        // Use -INFINITY for inactive lanes so they don't affect the max
        float partial = (lane < num_warps) ? shared[lane] : -INFINITY;
        
        // Reduce all partial maxima across warp 0
        block_max = warp_reduce_max(partial);
    }
    
    return block_max;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
    // Compute global index from launch configuration
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to cover all elements regardless of n
    // This handles any n, even if it's larger than the total threads launched
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + residual[i];
    }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to prevent out-of-bounds access
    if (idx < n) {
        // Load input value
        float v = x[idx];
        
        // Compute cubic term: v^3
        float v3 = v * v * v;
        
        // GELU tanh approximation:
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
        const float cubic_coeff = 0.044715f;
        
        float tanh_arg = sqrt_2_over_pi * (v + cubic_coeff * v3);
        float gelu = 0.5f * v * (1.0f + tanhf(tanh_arg));
        
        // Store result
        out[idx] = gelu;
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to prevent out-of-bounds access
    if (idx < n) {
        float v = x[idx];
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        out[idx] = v / (1.0f + expf(-v));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to prevent out-of-bounds access
    if (idx < n) {
        // Load gate and up values
        float g = gate[idx];
        float u = up[idx];
        
        // Compute SiLU(gate): g / (1 + exp(-g))
        float silu_g = g / (1.0f + expf(-g));
        
        // SwiGLU: SiLU(gate) * up
        out[idx] = silu_g * u;
    }
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    // Static shared memory: 32 floats covers up to 32 warps (1024 threads)
    __shared__ float shared[32];

    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * n;
    float* out_row = out + (size_t)row * n;

    // Step 1: each thread accumulates partial sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x_row[i];
        sum_sq += v * v;
    }

    // Step 2: warp-level reduction using XOR butterfly
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    // Step 3: each warp writes its sum to shared memory
    if (lane == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // Step 4: warp 0 reduces all warp sums
    float block_sum_sq = 0.0f;
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        block_sum_sq = partial;

        // Compute inverse RMS and broadcast via shared[0]
        float mean_sq = block_sum_sq / n;
        float rms = sqrtf(mean_sq + eps);
        shared[0] = 1.0f / rms;
    }
    __syncthreads();

    // Step 5: all threads read inv_rms and apply normalization
    float inv_rms = shared[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // Static shared memory: 32 floats covers up to 32 warps (1024 threads)
    __shared__ float shared[32];
    
    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * n;
    float* out_row = out + (size_t)row * n;
    
    // ============ PASS 1: Compute mean ============
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x_row[i];
    }
    
    // Reduce to block-wide sum
    float block_sum = block_reduce_sum(sum, shared);
    
    // Compute mean and broadcast to all threads
    if (threadIdx.x == 0) {
        shared[0] = block_sum / n;  // mean
    }
    __syncthreads();
    
    float mean = shared[0];
    
    // ============ PASS 2: Compute variance ============
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }
    
    // Reduce to block-wide sum of squared differences
    float block_var_sum = block_reduce_sum(var_sum, shared);
    
    // Compute inverse standard deviation and broadcast
    if (threadIdx.x == 0) {
        float var = block_var_sum / n + eps;
        shared[0] = rsqrtf(var);  // inv_std = 1/sqrt(var + eps)
    }
    __syncthreads();
    
    float inv_std = shared[0];
    
    // ============ PASS 3: Apply normalization ============
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float normalized = (x_row[i] - mean) * inv_std;
        out_row[i] = normalized * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // Static shared memory: 32 floats covers up to 32 warps (1024 threads)
    __shared__ float shared[32];
    
    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * n;
    const float* residual_row = residual + (size_t)row * n;
    float* out_row = out + (size_t)row * n;
    float* residual_out_row = residual_out + (size_t)row * n;
    
    // ============ PASS 1: Residual add + sum of squares ============
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float r = x_row[i] + residual_row[i];
        residual_out_row[i] = r;
        sum_sq += r * r;
    }
    
    // Reduce to block-wide sum of squares
    float block_sum_sq = block_reduce_sum(sum_sq, shared);
    
    // Compute inverse RMS and broadcast to all threads
    if (threadIdx.x == 0) {
        float rms = sqrtf(block_sum_sq / n + eps);
        shared[0] = 1.0f / rms;  // inv_rms
    }
    __syncthreads();
    
    float inv_rms = shared[0];
    
    // ============ PASS 2: Apply normalization and scaling ============
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = residual_out_row[i] * inv_rms * weight[i];
    }
}

# Step 12 - softmax_row_kernel
__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    // Dynamic shared memory for block reductions (one slot per warp)
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * cols;
    float* out_row = out + (size_t)row * cols;
    
    // ============ PASS 1: Find row max ============
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i];
        if (v > local_max) local_max = v;
    }
    
    // Reduce to block-wide max
    float row_max = block_reduce_max(local_max, shared);
    
    // Broadcast row_max to all threads
    if (threadIdx.x == 0) {
        shared[0] = row_max;
    }
    __syncthreads();
    row_max = shared[0];
    
    // ============ PASS 2: Compute sum of exponentials ============
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += expf(x_row[i] - row_max);
    }
    
    // Reduce to block-wide sum
    float row_sum = block_reduce_sum(local_sum, shared);
    
    // Broadcast row_sum to all threads
    if (threadIdx.x == 0) {
        shared[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared[0];
    
    // ============ PASS 3: Write softmax output ============
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

# Step 13 - causal_softmax_kernel
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    // Static shared memory: 32 floats covers up to 32 warps (1024 threads)
    __shared__ float shared[32];

    int row = blockIdx.x;
    const float* x_row = x + (size_t)row * cols;
    float* out_row = out + (size_t)row * cols;

    // Valid length: only columns <= row participate (causal mask)
    int valid_len = (row + 1 < cols) ? row + 1 : cols;

    // ---------- Step 1: find max over valid prefix ----------
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        float v = x_row[i];
        if (v > local_max) local_max = v;
    }

    // Warp-level max reduction (XOR butterfly)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    // Warp 0 collects partial maxima from all warps
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) {
        shared[warp_id] = local_max;
    }
    __syncthreads();

    // Warp 0 reduces all partial maxima
    float row_max = -INFINITY;
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? shared[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial = fmaxf(partial, __shfl_xor_sync(0xffffffff, partial, offset));
        }
        row_max = partial;
        // Broadcast row_max to all threads via shared[0]
        shared[0] = row_max;
    }
    __syncthreads();
    row_max = shared[0];

    // ---------- Step 2: sum of exponentials over valid prefix ----------
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        local_sum += expf(x_row[i] - row_max);
    }

    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    float row_sum = 0.0f;
    if (warp_id == 0) {
        float partial = (lane < num_warps) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        row_sum = partial;
        shared[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared[0];

    // ---------- Step 3: write outputs ----------
    // Write valid columns (c <= row) with softmax
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }

    // Write masked columns (c > row) as zero
    for (int i = threadIdx.x + valid_len; i < cols; i += blockDim.x) {
        out_row[i] = 0.0f;
    }
}

# Step 14 - embedding_lookup_kernel
__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, 
                                        float* out, int seq_len, int vocab_size, 
                                        int embed_dim) {
    // Total number of output elements
    int total = seq_len * embed_dim;
    
    // Grid-stride loop over all output elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total; 
         idx += blockDim.x * gridDim.x) {
        
        // Compute sequence position and feature dimension
        int pos = idx / embed_dim;          // which token in the sequence
        int dim = idx % embed_dim;          // which embedding element
        
        // Load token ID for this position
        int token = token_ids[pos];
        
        // Copy the embedding vector element from the weight table
        // weight is row-major: [vocab_size][embed_dim]
        out[idx] = weight[(size_t)token * embed_dim + dim];
    }
}

# Step 15 - rope_kernel
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    int half = head_dim / 2;
    int total = seq_len * n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Recover position (t), head (h), and pair index (i)
    int pos = idx / (n_heads * half);
    int rem = idx - pos * (n_heads * half);
    int head = rem / half;
    int pair = rem - head * half;

    // Base offset for this (pos, head) in the feature dimension
    size_t base = ((size_t)pos * n_heads + head) * head_dim;
    size_t even = base + 2 * pair;
    size_t odd = even + 1;

    // Load cosine and sine for this position and pair
    size_t table_idx = (size_t)pos * half + pair;
    float cos_val = cos_table[table_idx];
    float sin_val = sin_table[table_idx];

    // ---- Apply rotation to query ----
    float q0 = q[even];
    float q1 = q[odd];
    q[even] = q0 * cos_val - q1 * sin_val;
    q[odd]  = q0 * sin_val + q1 * cos_val;

    // ---- Apply the identical rotation to key ----
    float k0 = k[even];
    float k1 = k[odd];
    k[even] = k0 * cos_val - k1 * sin_val;
    k[odd]  = k0 * sin_val + k1 * cos_val;
}

# Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // Total number of output elements (M * N)
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Determine row (m) and column (n) in the output matrix
    int m = idx / N;       // which batch element
    int n = idx % N;       // which output feature

    // Compute dot product: sum over K
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += x[(size_t)m * K + k] * weight[(size_t)n * K + k];
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[n];
    }

    // Write result
    out[idx] = sum;
}

# Step 17 - fused_linear_bias_gelu_kernel
__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    
    // Total output elements (M * N)
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode row (m) and column (n) of the output matrix
    int m = idx / N;
    int n = idx % N;

    // Compute dot product: sum over K
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += x[(size_t)m * K + k] * weight[(size_t)n * K + k];
    }

    // Add bias (non‑null in this fused kernel)
    sum += bias[n];

    // ----- Apply GELU (tanh approximation) -----
    // Constants for GELU: sqrt(2/pi) and cubic coefficient
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrtf(2.0f / M_PI)
    const float cubic_coeff = 0.044715f;

    float x_val = sum;
    float x3 = x_val * x_val * x_val;
    float tanh_arg = sqrt_2_over_pi * (x_val + cubic_coeff * x3);
    float gelu = 0.5f * x_val * (1.0f + tanhf(tanh_arg));

    // Write result
    out[idx] = gelu;
}

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

