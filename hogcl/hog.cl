#define CELL_WIDTH 8
#define CELL_HEIGHT 8
#define CELLS_PER_BLOCK_X 2
#define CELLS_PER_BLOCK_Y 2
#define NTHREADS 256
#define CV_PI_F M_PI_F

#ifdef INTEL_DEVICE
#define QANGLE_TYPE     int
#define QANGLE_TYPE2    int2
#else
#define QANGLE_TYPE     uchar
#define QANGLE_TYPE2    uchar2
#endif


//----------------------------------------------------------------------------
// Gradients computation


__kernel void compute_gradients_8UC1_kernel(
    const int height, const int width,
    const int img_step, const int grad_quadstep, const int qangle_step,
    __global const uchar * img, __global float * grad, __global unsigned char * qangle,
    const float angle_scale, const char correct_gamma, const int cnbins)
{
    
    
    const int x = get_global_id(0);
    const int tid = get_local_id(0);
    const int gSizeX = get_local_size(0);
    const int gidY = get_group_id(1);

    __global const uchar* row = img + gidY * img_step;

    __local float sh_row[NTHREADS + 2];

    if (x < width)
        sh_row[tid + 1] = row[x];
    else
        sh_row[tid + 1] = row[width - 2];

    if (tid == 0)
        sh_row[0] = row[max(x - 1, 1)];

    if (tid == gSizeX - 1)
        sh_row[gSizeX + 1] = row[min(x + 1, width - 2)];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (x < width)
    {
        float dx;

        if (correct_gamma == 1)
            dx = sqrt(sh_row[tid + 2]) - sqrt(sh_row[tid]);
        else
            dx = sh_row[tid + 2] - sh_row[tid];

        float dy = 0.f;
        if (gidY > 0 && gidY < height - 1)
        {
            float a = (float) img[ (gidY + 1) * img_step + x ];
            float b = (float) img[ (gidY - 1) * img_step + x ];
            if (correct_gamma == 1)
                dy = sqrt(a) - sqrt(b);
            else
                dy = a - b;
        }
        float mag = hypot(dx, dy);

        float ang = (atan2(dy, dx) + CV_PI_F) * angle_scale - 0.5f;
        int hidx = (int)floor(ang);
        ang -= hidx;
        hidx = (hidx + cnbins) % cnbins;

        qangle[ (gidY * qangle_step + x) << 1 ]     = hidx;
        qangle[ ((gidY * qangle_step + x) << 1) + 1 ] = (hidx + 1) % cnbins;
        grad[ (gidY * grad_quadstep + x) << 1 ]       = mag * (1.f - ang);
        grad[ ((gidY * grad_quadstep + x) << 1) + 1 ]   = mag * ang;
    }
    
    
}





__kernel void compute_hists_lut_kernel(
                                       const int cblock_stride_x, const int cblock_stride_y,
                                       const int cnbins, const int cblock_hist_size, const int img_block_width,
                                       const int blocks_in_group, const int blocks_total,
                                       const int grad_quadstep, const int qangle_step,
                                       __global const float* grad, __global const uchar* qangle,
                                       __global const float* gauss_w_lut,
                                       __global float* block_hists, __local float* smem)
{
    
    
    const int lx = get_local_id(0);
    const int lp = lx / 24; /* local group id */
    const int gid = get_group_id(0) * blocks_in_group + lp;/* global group id */
    const int gidY = gid / img_block_width;
    const int gidX = gid - gidY * img_block_width;
    
    const int lidX = lx - lp * 24;
    const int lidY = get_local_id(1);
    
    const int cell_x = lidX / 12;
    const int cell_y = lidY;
    const int cell_thread_x = lidX - cell_x * 12;
    
    __local float* hists = smem + lp * cnbins * (CELLS_PER_BLOCK_X *
                                                 CELLS_PER_BLOCK_Y * 12 + CELLS_PER_BLOCK_X * CELLS_PER_BLOCK_Y);
    __local float* final_hist = hists + cnbins *
    (CELLS_PER_BLOCK_X * CELLS_PER_BLOCK_Y * 12);
    
    const int offset_x = gidX * cblock_stride_x + (cell_x << 2) + cell_thread_x;
    const int offset_y = gidY * cblock_stride_y + (cell_y << 2);
    
    __global const float* grad_ptr = (gid < blocks_total) ?
    grad + offset_y * grad_quadstep + (offset_x << 1) : grad;
    __global const QANGLE_TYPE* qangle_ptr = (gid < blocks_total) ?
    qangle + offset_y * qangle_step + (offset_x << 1) : qangle;
    
    __local float* hist = hists + 12 * (cell_y * CELLS_PER_BLOCK_Y + cell_x) +
    cell_thread_x;
    for (int bin_id = 0; bin_id < cnbins; ++bin_id)
        hist[bin_id * 48] = 0.f;
    
    const int dist_x = -4 + cell_thread_x - 4 * cell_x;
    const int dist_center_x = dist_x - 4 * (1 - 2 * cell_x);
    
    const int dist_y_begin = -4 - 4 * lidY;
    for (int dist_y = dist_y_begin; dist_y < dist_y_begin + 12; ++dist_y)
    {
        float2 vote = (float2) (grad_ptr[0], grad_ptr[1]);
        QANGLE_TYPE2 bin = (QANGLE_TYPE2) (qangle_ptr[0], qangle_ptr[1]);
        
        grad_ptr += grad_quadstep;
        qangle_ptr += qangle_step;
        
        int dist_center_y = dist_y - 4 * (1 - 2 * cell_y);
        
        int idx = (dist_center_y + 8) * 16 + (dist_center_x + 8);
        float gaussian = gauss_w_lut[idx];
        idx = (dist_y + 8) * 16 + (dist_x + 8);
        float interp_weight = gauss_w_lut[256+idx];
        
        hist[bin.x * 48] += gaussian * interp_weight * vote.x;
        hist[bin.y * 48] += gaussian * interp_weight * vote.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    volatile __local float* hist_ = hist;
    for (int bin_id = 0; bin_id < cnbins; ++bin_id, hist_ += 48)
    {
        if (cell_thread_x < 6)
            hist_[0] += hist_[6];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (cell_thread_x < 3)
            hist_[0] += hist_[3];
#ifdef CPU
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        if (cell_thread_x == 0)
            final_hist[(cell_x * 2 + cell_y) * cnbins + bin_id] =
            hist_[0] + hist_[1] + hist_[2];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int tid = (cell_y * CELLS_PER_BLOCK_Y + cell_x) * 12 + cell_thread_x;
    if ((tid < cblock_hist_size) && (gid < blocks_total))
    {
        __global float* block_hist = block_hists +
        (gidY * img_block_width + gidX) * cblock_hist_size;
        block_hist[tid] = final_hist[tid];
    }
    
    
}




//  Normalization of histograms via L2Hys_norm
//  optimized for the case of 9 bins
__kernel void normalize_hists_36_kernel(__global float* block_hists,
                                        const float threshold, __local float *squares)
{
    const int tid = get_local_id(0);
    const int gid = get_global_id(0);
    const int bid = tid / 36;      /* block-hist id, (0 - 6) */
    const int boffset = bid * 36;  /* block-hist offset in the work-group */
    const int hid = tid - boffset; /* histogram bin id, (0 - 35) */

    float elem = block_hists[gid];
    squares[tid] = elem * elem;
    barrier(CLK_LOCAL_MEM_FENCE);

    __local float* smem = squares + boffset;
    float sum = smem[hid];
    if (hid < 18)
        smem[hid] = sum = sum + smem[hid + 18];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hid < 9)
        smem[hid] = sum = sum + smem[hid + 9];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hid < 4)
        smem[hid] = sum + smem[hid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    sum = smem[0] + smem[1] + smem[2] + smem[3] + smem[8];

    elem = elem / (sqrt(sum) + 3.6f);
    elem = min(elem, threshold);

    barrier(CLK_LOCAL_MEM_FENCE);
    squares[tid] = elem * elem;
    barrier(CLK_LOCAL_MEM_FENCE);

    sum = smem[hid];
    if (hid < 18)
      smem[hid] = sum = sum + smem[hid + 18];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hid < 9)
        smem[hid] = sum = sum + smem[hid + 9];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hid < 4)
        smem[hid] = sum + smem[hid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);
    sum = smem[0] + smem[1] + smem[2] + smem[3] + smem[8];

    block_hists[gid] = elem / (sqrt(sum) + 1e-3f);
}


__kernel void extract_descrs_by_cols_kernel(
    const int cblock_hist_size, const int descriptors_quadstep, const int cdescr_size,
    const int cnblocks_win_x, const int cnblocks_win_y, const int img_block_width,
    const int win_block_stride_x, const int win_block_stride_y,
    __global const float* block_hists, __global float* descriptors)
{
    int tid = get_local_id(0);
    int gidX = get_group_id(0);
    int gidY = get_group_id(1);

    // Get left top corner of the window in src
    __global const float* hist = block_hists +  (gidY * win_block_stride_y *
        img_block_width + gidX * win_block_stride_x) * cblock_hist_size;

    // Get left top corner of the window in dst
    __global float* descriptor = descriptors +
        (gidY * get_num_groups(0) + gidX) * descriptors_quadstep;

    // Copy elements from src to dst
    for (int i = tid; i < cdescr_size; i += NTHREADS)
    {
        int block_idx = i / cblock_hist_size;
        int idx_in_block = i - block_idx * cblock_hist_size;

        int y = block_idx / cnblocks_win_x;
        int x = block_idx - y * cnblocks_win_x;

        descriptor[(x * cnblocks_win_y + y) * cblock_hist_size + idx_in_block] =
            hist[(y * img_block_width  + x) * cblock_hist_size + idx_in_block];
    }
}

