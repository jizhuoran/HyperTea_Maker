class MIOBatchNorm(object):
    """docstring for MIOBatchNorm"""
    def __init__(self, batch_size, channels, spatial_dim):
        super(MIOBatchNorm, self).__init__()

        self.batch_size = batch_size
        self.channels = channels
        self.spatial_dim = spatial_dim

        self.in_cstride = spatial_dim
        self.in_nstride = channels * self.in_cstride
        self.in_nhw     = batch_size * self.in_cstride
        self.in_nchw    = batch_size * self.in_nstride
        # self.xlocalsize = 1024
        # ylocalsize = 1
        # zlocalsize = 1

        # xgridsize = self.channels * xlocalsize
        # ygridsize = 1
        # zgridsize = 1




        self.inhw = 1.0 / self.in_nhw

        # ldsgcn   = xlocalsize / 64
        # ldsnogcn = xlocalsize



        if self.in_nhw < 33554432 and self.in_cstride > 1024:  
            self.variant  = 1
            
            self.xlocalsize = 1024
            self.ylocalsize = 1
            self.zlocalsize = 1

            self.xgridsize = self.channels * self.xlocalsize
            self.ygridsize = 1
            self.zgridsize = 1

            self.ldsgcn   = self.xlocalsize // 64
            self.ldsnogcn = self.xlocalsize
        
            self.single       = True

        elif(self.in_nhw < 33554432 and self.in_cstride > 512):
            self.variant    = 3
            
            self.xlocalsize = 64 * ((in_cstride + 63) // 64)
            self.ylocalsize = 1
            self.zlocalsize = 1

            self.xgridsize = self.channels * self.xlocalsize
            self.ygridsize = 1
            self.zgridsize = 1

            self.ldsgcn     = self.xlocalsize // 64
            self.ldsnogcn   = self.xlocalsize

            self.single       = True

        elif(self.in_cstride <= 512):
            self.variant = 0

            self.xlocalsize = 1024
            self.ylocalsize = 1
            self.zlocalsize = 1

            self.xgridsize = self.channels * self.xlocalsize
            self.ygridsize = 1
            self.zgridsize = 1

            self.ldsgcn   = self.xlocalsize // 64
            self.ldsnogcn = self.xlocalsize

            self.single       = True


        else:
            self.variant      = 2

            self.xlocalsize   = 1
            self.ylocalsize   = 1024
            self.zlocalsize = 1

            self.xgridsize    = self.channels
            self.ygridsize    = math.ceil(self.in_cstride / self.ylocalsize) * self.ylocalsize
            self.zgridsize = 1

            self.ldsgcn       = self.ylocalsize // 64
            self.ldsnogcn     = self.ylocalsize
        
            self.single       = False




    def generate_macro_(self, precision = 'float'):

        f'''
#define MIO_BN_N {self.batch_size}
#define MIO_BN_C {self.channels}
#define {self.in_cstride} {self.in_cstride}
#define {self.in_nhw} {self.in_nhw}
#define MIO_BN_CHW {self.in_nstride}
#define MIO_BN_NCHW {self.in_nchw}
#define MIO_BN_LDS_SIZE {self.ldsnogcn}
#define MIO_BN_LDSGCN_SIZE {self.ldsgcn}
#define MIO_BN_VARIANT {self.variant}
#define {self.xlocalsize} {self.xlocalsize}
#define MIO_BN_GRP1 {self.ylocalsize}
#define MIO_BN_GRP2 {self.zlocalsize}


#define MIO_save_BN_N {self.batch_size}
#define MIO_save_BN_C {self.channels}
#define MIO_save_BN_HW {self.in_cstride}
#define MIO_save_BN_NHW {self.in_nhw}
#define MIO_save_BN_CHW {self.in_nstride}
#define MIO_save_BN_NCHW {self.in_nchw}
#define MIO_save_BN_LDS_SIZE {self.ldsnogcn}
#define MIO_save_BN_LDSGCN_SIZE {self.ldsgcn}
#define MIO_save_BN_VARIANT {self.variant}
#define MIO_save_BN_GRP0 {self.xlocalsize}
#define MIO_save_BN_GRP1 {self.ylocalsize}
#define MIO_save_BN_GRP2 {self.zlocalsize}
#define Dtype {precision}
#define Dtype4 {precision}4


static inline void ReduceKernel(__local Dtype* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{{
    Dtype sum              = (Dtype)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    for(unsigned int i = 0; i < unit_len; i += sum_stride) {{
        sum += lcl_mem[lcl_offset + i];
    }}
    lcl_mem[lcl_offset] = sum;
}}

static inline void
regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}}
        '''


    def generate_variant0_(self):





        # print(MIO_BN_NLOOP)
        # print(MIO_BN_NLOOPM)
        # print(MIO_BN_SEGIHW)
        # print(MIO_BN_SNHW)



        MIO_BN_SEGTMP = self.in_cstride * (self.xlocalsize // self.in_cstride)
        MIO_BN_N = self.batch_size
        MIO_BN_C = self.channels
        MIO_BN_CHW = self.in_nstride
        MIO_BN_NCHW = self.in_nchw
        MIO_BN_LDS_SIZE = self.ldsnogcn
        MIO_BN_LDSGCN_SIZE = self.ldsgcn
        MIO_BN_VARIANT = self.variant
        MIO_BN_GRP1 = self.ylocalsize
        MIO_BN_GRP2 = self.zlocalsize

        print(MIO_BN_SEGTMP)
        print(MIO_BN_N)
        print(MIO_BN_C)
        print(MIO_BN_CHW)
        print(MIO_BN_NCHW)
        print(MIO_BN_LDS_SIZE)
        print(MIO_BN_LDSGCN_SIZE)
        print(MIO_BN_VARIANT)
        print(MIO_BN_GRP1)
        print(MIO_BN_GRP2)


        MIO_BN_SEGMENT = min(self.in_nhw, MIO_BN_SEGTMP)

        print(MIO_BN_SEGMENT)


        MIO_BN_NLOOP = (self.in_nhw + MIO_BN_SEGMENT - 1) // MIO_BN_SEGMENT
        MIO_BN_NLOOPM = MIO_BN_NLOOP - 1
        MIO_BN_SEGIHW = MIO_BN_SEGMENT // self.in_cstride
        MIO_BN_SNHW = MIO_BN_NLOOPM * MIO_BN_SEGIHW
        f'''

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias,
                               Dtype INHW,
                               Dtype epsilon
                               )
{{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype pvscale     = (Dtype)0.;
    Dtype pvbias      = (Dtype)0.;
    Dtype batchvalues[{MIO_BN_NLOOP}];

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * {self.in_cstride} + (lid % {self.in_cstride});
    unsigned int lidihw = lid / {self.in_cstride};
    unsigned int nid    = 0;

    if(lid == 0)
    {{
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < {MIO_BN_SEGMENT})
    {{
        __attribute__((opencl_unroll_hint(2)))
        for(unsigned int n = 0; n < {MIO_BN_NLOOPM}; ++n)
        {{
            nid            = n * {MIO_BN_SEGIHW} + lidihw;
            index          = nid * {self.in_nstride} + chwid;
            batchvalues[n] = (Dtype)(*(in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }}
        nid   = {MIO_BN_SNHW} + lidihw;
        index = nid * {self.in_nstride} + chwid;
        batchvalues[{MIO_BN_NLOOPM}] =
            (index < {self.in_nchw}) ? (Dtype)(*(in + index)) : (Dtype)0.;
        mean += batchvalues[{MIO_BN_NLOOPM}];
        variance = mad(batchvalues[{MIO_BN_NLOOPM}], batchvalues[{MIO_BN_NLOOPM}], variance);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    __local Dtype lcl_data[{self.ldsnogcn}];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = ({self.xlocalsize} >> 1); red > 256; red >>= 1)
    {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce variance
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = ({self.xlocalsize} >> 1); red > 256; red >>= 1)
    {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);


    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (Dtype)epsilon);
    pvscale     = (Dtype)lcl_scale;
    pvbias      = (Dtype)lcl_bias;

    if(lid < {MIO_BN_SEGMENT})
    {{
        //==== CALC NORM =======================
        Dtype inhat = (Dtype)0.;

        __attribute__((opencl_unroll_hint(2)))
        for(unsigned int n = 0; n < {MIO_BN_NLOOPM}; n++)
        {{ // apply normalization
            inhat      = (batchvalues[n] - mean) * invVariance;
            nid        = n * {MIO_BN_SEGIHW} + lidihw;
            index      = nid * {self.in_nstride} + chwid;
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
        }} // end for

        // Tail of loop
        inhat = (batchvalues[{MIO_BN_NLOOPM}] - mean) * invVariance;
        nid   = {MIO_BN_SNHW} + lidihw;
        index = nid * {self.in_nstride} + chwid;
        if(index < {self.in_nchw})
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
    }}

}} // end spatial norm
        '''


temp = MIOBatchNorm(1, 3, 16)

print(temp.generate_macro_())
print(temp.generate_variant0_())
  