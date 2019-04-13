import math



class MIOBatchNormGenerator(object):
    """docstring for MIOBatchNormGenerator"""
    def __init__(self, precision, kernels):
        super(MIOBatchNormGenerator, self).__init__()
        self.precision = precision
        
        self.MIO_batchnorm_code = ''.join(kernels)

        self.header = f'''
#define Dtype {self.precision}
#define Dtype4 {self.precision}4


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
        '''


    def generate_bn_code(self):

        return f'''
std::string bn_opencl_funcs = R"(
{self.header}
{self.MIO_batchnorm_code}
)";
        '''




class MIOBatchNorm(object):
    """docstring for MIOBatchNorm"""
    def __init__(self, op_name, batch_size, channels, spatial_dim, epsilon):
        super(MIOBatchNorm, self).__init__()

        self.function_selector = [
            self.generate_variant0_,
            self.generate_variant1_,
            self.generate_variant2_,
            self.generate_variant3_
        ]

        self.unroll_hint = lambda x: '__attribute__((opencl_unroll_hint({})))'.format(x) if False else ' '

        self.op_name = op_name + '_'
        self.batch_size = batch_size
        self.channels = channels
        self.spatial_dim = spatial_dim

        self.in_cstride = spatial_dim
        self.in_nstride = channels * self.in_cstride
        self.in_nhw     = batch_size * self.in_cstride
        self.in_nchw    = batch_size * self.in_nstride

        self.epsilon = epsilon
        self.inhw = 1.0 / self.in_nhw


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
            
            self.xlocalsize = 64 * ((self.in_cstride + 63) // 64)
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


        else: #in_cstride > 512 && self.in_nhw >= 33554432
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




    def generate_MIO_BN_func(self):

        return self.generate_macro_() + self.function_selector[self.variant]()



    def local_shape(self):
      return [self.xlocalsize, self.ylocalsize, self.zlocalsize]

    def global_shape(self):
      return [self.xgridsize, self.ygridsize, self.zgridsize]



    def generate_macro_(self):

        MIO_BN_LDS_SIZE = self.ldsnogcn

        return f'''
static inline void
{self.op_name}regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < ({MIO_BN_LDS_SIZE} >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < ({MIO_BN_LDS_SIZE} >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, {MIO_BN_LDS_SIZE});
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}}
        '''


    def generate_variant0_(self):


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
        MIO_BN_SEGMENT = min(self.in_nhw, MIO_BN_SEGTMP)



        MIO_BN_NLOOP = (self.in_nhw + MIO_BN_SEGMENT - 1) // MIO_BN_SEGMENT
        MIO_BN_NLOOPM = MIO_BN_NLOOP - 1
        MIO_BN_SEGIHW = MIO_BN_SEGMENT // self.in_cstride
        MIO_BN_SNHW = MIO_BN_NLOOPM * MIO_BN_SEGIHW
        
        return f'''

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {{

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
        {self.unroll_hint(2)}
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
    {self.op_name}regLDSreduce(&mean, lcl_data, lid, (Dtype){self.inhw});
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
    {self.op_name}regLDSreduce(&variance, lcl_data, lid, (Dtype){self.inhw});
    barrier(CLK_LOCAL_MEM_FENCE);


    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (Dtype){self.epsilon});
    pvscale     = (Dtype)lcl_scale;
    pvbias      = (Dtype)lcl_bias;

    if(lid < {MIO_BN_SEGMENT})
    {{
        //==== CALC NORM =======================
        Dtype inhat = (Dtype)0.;

        {self.unroll_hint(2)}
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





    def generate_variant1_(self):


        # MIO_BN_SEGTMP = self.in_cstride * (self.xlocalsize // self.in_cstride)
        # MIO_BN_N = self.batch_size
        # MIO_BN_C = self.channels
        # MIO_BN_CHW = self.in_nstride
        # MIO_BN_NHW = self.in_nhw

        # MIO_BN_NCHW = self.in_nchw
        # MIO_BN_LDS_SIZE = self.ldsnogcn
        # MIO_BN_LDSGCN_SIZE = self.ldsgcn
        # MIO_BN_VARIANT = self.variant
        # MIO_BN_GRP0 = self.xlocalsize
        # MIO_BN_GRP1 = self.ylocalsize
        # MIO_BN_GRP2 = self.zlocalsize
        # MIO_BN_SEGMENT = min(self.in_nhw, MIO_BN_SEGTMP)

        MIO_MAX_READ = 2 if self.in_cstride < 4096 else 3
        RD_BLK = 1


        GRPRD = self.xlocalsize * RD_BLK * 4
        MIO_BN_REM4 = self.in_nhw - ((self.in_nhw // GRPRD) * GRPRD)
        MIO_BN_LESS4 = self.in_nhw - MIO_BN_REM4
        
        MIO_BN_CHUNK4 = MIO_MAX_READ * GRPRD
        MIO_BN_REMOUT4 = self.in_nhw - ((self.in_nhw // MIO_BN_CHUNK4) * MIO_BN_CHUNK4)
        MIO_BN_LESSOUT4 = self.in_nhw - MIO_BN_REMOUT4


        MIO_BN_REM = self.in_nhw - ((self.in_nhw // self.xlocalsize) * self.xlocalsize)
        MIO_BN_LESS = self.in_nhw - MIO_BN_REM

        MIO_BN_CHUNK = MIO_MAX_READ * self.xlocalsize
        MIO_BN_REMOUT = self.in_nhw - ((self.in_nhw // MIO_BN_CHUNK) * MIO_BN_CHUNK)
        MIO_BN_LESSOUT = self.in_nhw - MIO_BN_REMOUT
        
        if self.in_cstride >= 4096:
            read_input = f'''
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < {MIO_BN_LESS4};
                                               k += {GRPRD}) {{
        nidx  = k / {self.in_cstride};
        hwidx = k - (nidx * {self.in_cstride});
        index = nidx * {self.in_nstride} + chwid + hwidx;
        read4 = *((const global Dtype4*)(in + index));
        mean += (Dtype)read4.x;
        mean += (Dtype)read4.y;
        mean += (Dtype)read4.z;
        mean += (Dtype)read4.w;
        variance = mad((Dtype)read4.x, (Dtype)read4.x, variance);
        variance = mad((Dtype)read4.y, (Dtype)read4.y, variance);
        variance = mad((Dtype)read4.z, (Dtype)read4.z, variance);
        variance = mad((Dtype)read4.w, (Dtype)read4.w, variance);
    }}
    ''' + (f'''

    unsigned int remkey = (lid << 2) + {MIO_BN_LESS4};
    nidx                = remkey / {self.in_cstride};
    hwidx               = remkey - (nidx * {self.in_cstride});
    index               = nidx * {self.in_nstride} + chwid + hwidx;
    if(index < {self.in_nchw}) {{
        read4 = *((const global Dtype4*)(in + index));
        mean += (Dtype)read4.x;
        mean += (Dtype)read4.y;
        mean += (Dtype)read4.z;
        mean += (Dtype)read4.w;
        variance = mad((Dtype)read4.x, (Dtype)read4.x, variance);
        variance = mad((Dtype)read4.y, (Dtype)read4.y, variance);
        variance = mad((Dtype)read4.z, (Dtype)read4.z, variance);
        variance = mad((Dtype)read4.w, (Dtype)read4.w, variance);
    }}
    ''' if not MIO_BN_REM4 == 0 else ' ')


        else:
            read_input = f'''
    {self.unroll_hint(4)} for(unsigned int k = lid; k < {MIO_BN_LESS};
                                               k += {self.xlocalsize}) {{
        nidx            = k / {self.in_cstride};
        hwidx           = k - (nidx * {self.in_cstride});
        index           = nidx * {self.in_nstride} + chwid + hwidx;
        Dtype xin = (Dtype)(*(in + index));
        mean += xin;
        variance = mad(xin, xin, variance);
    }}
    ''' +  (f'''
    if(lid < MIO_BN_REM) {{
        unsigned int remkey = lid + {MIO_BN_LESS};
        nidx                = remkey / {self.in_cstride};
        hwidx               = remkey - (nidx * {self.in_cstride});
        index               = nidx * {self.in_nstride} + chwid + hwidx;
        Dtype xin = (index < {self.in_nchw}) ? (Dtype)(*(in + index)) : (Dtype)0.;
        mean += xin;
        variance = mad(xin, xin, variance);
    }}
    ''' if not MIO_BN_REM == 0 else ' ')




        if MIO_BN_REM == 0:
            save_output = f'''
    for(unsigned int k = lid; k < {MIO_BN_LESS}; k += {self.xlocalsize}) {{
        nidx  = k / {self.in_cstride};
        hwidx = k - (nidx * {self.in_cstride});
        index = nidx * {self.in_nstride} + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    }} // end for
        '''
        else:
            save_output = f'''

    Dtype xhat[{MIO_MAX_READ}];
    for(unsigned int k = ({MIO_MAX_READ} * lid); k < {MIO_BN_LESSOUT}; k += {MIO_BN_CHUNK}) {{
        for(unsigned int j = 0; j < {MIO_MAX_READ}; j++) {{
            unsigned int l = k + j;
            nidx           = l / {self.in_cstride};
            hwidx          = l - (nidx * {self.in_cstride});
            index          = nidx * {self.in_nstride} + chwid + hwidx;
            xhat[j]        = ((Dtype)(*(in + index)) - mean) * invVariance;
        }}
        barrier(CLK_GLOBAL_MEM_FENCE);
        for(unsigned int j = 0; j < {MIO_MAX_READ}; j++) {{
            unsigned int l = k + j;
            nidx           = l / {self.in_cstride};
            hwidx          = l - (nidx * {self.in_cstride});
            index          = nidx * {self.in_nstride} + chwid + hwidx;
            *(out + index) = (Dtype)mad(pvscale, xhat[j], pvbias);
        }}
    }} // end for
        ''' + (f'''
    unsigned int remkeyout = ({MIO_MAX_READ} * lid) + {MIO_BN_LESSOUT};
    for(unsigned int j = 0; j < {MIO_MAX_READ}; j++) {{
        unsigned int l  = remkeyout + j;
        nidx            = l / {self.in_cstride};
        hwidx           = l - (nidx * {self.in_cstride});
        index           = nidx * {self.in_nstride} + chwid + hwidx;
        Dtype xin = (index < {self.in_nchw}) ? (Dtype)(*(in + index)) : (Dtype)0.;
        xhat[j]         = (xin - mean) * invVariance;
    }}
    barrier(CLK_GLOBAL_MEM_FENCE);
    for(unsigned int j = 0; j < {MIO_MAX_READ}; j++) {{
        unsigned int l = remkeyout + j;
        nidx           = l / {self.in_cstride};
        hwidx          = l - (nidx * {self.in_cstride});
        index          = nidx * {self.in_nstride} + chwid + hwidx;
        if(index < {self.in_nchw}) {{
            *(out + index) = (Dtype)mad(pvscale, xhat[j], pvbias);
        }}
    }}
            ''' if not MIO_BN_REMOUT == 0 else ' ')


        return f'''

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {{

    // SPATIAL

    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype pvscale, pvbias;

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint grpid = get_group_id(0);
    uint chwid = grpid * {self.in_cstride};
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {{
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    {read_input}

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[{self.ldsnogcn}];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = ({self.xlocalsize} >> 1); red > 256; red >>= 1) {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&mean, lcl_data, lid, (Dtype){self.inhw});

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = ({self.xlocalsize} >> 1); red > 256; red >>= 1) {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&variance, lcl_data, lid, (Dtype){self.inhw});
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + {self.epsilon});

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    {save_output}


}} // end spatial norm
        '''


    def generate_variant2_(self):


        MIO_BN_LDS_SIZE = self.ldsnogcn
        MIO_BN_NGRPS = int(math.ceil(float(self.ygridsize) / self.ylocalsize))


        reduce_mean = f'''
    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&mean, lcl_data, lid, (Dtype){self.inhw});
        ''' if MIO_BN_NGRPS > 64 else f'''
    {self.op_name}regLDSreduce(&mean, lcl_data, lid, (Dtype){self.inhw});
    commitID = 0;

        '''


        if MIO_BN_NGRPS > 256:
            reduce_var = f'''
        for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
            if(lid < red)
                lcl_data[lid] += lcl_data[lid + red];
            barrier(CLK_LOCAL_MEM_FENCE);
        }}
        {self.op_name}regLDSreduce(&variance, lcl_data, lid, (Dtype){self.inhw});
        '''
        elif MIO_BN_NGRPS > 16:
            reduce_var = f'''
        {self.op_name}regLDSreduce(&variance, lcl_data, lid, (Dtype){self.inhw});
        '''
        else:
            reduce_var = f'''
        variance = (Dtype)0.;
        for(int i = 0; i < {MIO_BN_NGRPS}; i++) {{
            variance += lcl_data[i];
        }}
        '''




        return f'''
__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatialNorm(const __global Dtype* __restrict in,
                                   __global Dtype* __restrict out,
                                   const __global Dtype* __restrict scale,
                                   const __global Dtype* __restrict bias)
{{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype inhat       = (Dtype)0.;
    Dtype pvt_scale   = (Dtype)0.;
    Dtype pvt_bias    = (Dtype)0.;
    __local Dtype lcl_mean, lcl_ivar, lcl_scale, lcl_bias;

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx           = xgid * {self.in_cstride};
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + {self.epsilon})
    if(get_local_id(1) == 0) {{
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
        lcl_mean  = *(out + meanstashindex); // load stashed mean
        lcl_ivar  = *(out + varstashindex);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < {self.in_cstride}) {{
        mean        = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale   = lcl_scale;
        pvt_bias    = lcl_bias;
        for(unsigned int n = 0; n < {self.batch_size}; n++) {{ // apply normalization
            index        = n * {self.in_nstride} + cidx + ygid;
            Dtype inhat = (*(in + index) - mean) * invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        }} // end for(n)
    }}     // end if(inImgIndex)
}} // end spatial norm

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatialFinalMeanVariance(
    __global Dtype* __restrict meanvarbuff
    ) {{
    Dtype variance             = (Dtype)0.;
    Dtype invVariance          = (Dtype)0.;
    Dtype mean                 = (Dtype)0.;
    unsigned int lid            = get_local_id(1);
    unsigned int ygrp_id        = get_group_id(1);
    unsigned int xgid           = get_global_id(0);
    unsigned int ygrp_sz        = get_local_size(1);
    unsigned int yngrps         = get_num_groups(1);
    unsigned int cidx           = xgid * {self.in_cstride};
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
    unsigned int commitID       = 0;

    for(int gn = 0; gn < yngrps; gn++) {{
        unsigned int offset    = gn * ygrp_sz + lid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        unsigned int varindex  = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps) {{ 
            // modify to span larger number of groups
            mean += *(meanvarbuff + meanindex);
            variance += *(meanvarbuff + varindex); // load per group variance
        }}
    }}


    __local Dtype lcl_data[{MIO_BN_LDS_SIZE}];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    {reduce_mean}

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    {reduce_var}



    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + {self.epsilon});
    if(lid == commitID) {{
        meanvarbuff[meanstashindex] = mean;        // stash mean
        meanvarbuff[varstashindex]  = invVariance; // stash mean
    }}


}}

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatialMeanVariance(const __global Dtype* __restrict in,
                                           __global Dtype* __restrict mvbuff) {{

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * {self.in_cstride};
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id;
    unsigned int varindex  = meanindex + 2;
    Dtype mean            = (Dtype)0.;
    Dtype variance        = (Dtype)0.;
    Dtype value           = (Dtype)0.;

    if(ygid < {self.in_cstride}) {{
        for(unsigned int n = 0; n < {self.batch_size}; n++) {{
            index = n * {self.in_nstride} + cidx + ygid;
            value = *(in + index);
            mean += value;
            variance = mad(value, value, variance);
        }}
    }}



    __local Dtype lcl_data[{MIO_BN_LDS_SIZE}];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&mean, lcl_data, ylid, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&variance, lcl_data, ylid, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid == 0) {{
        mvbuff[meanindex] = mean;
        mvbuff[varindex]  = variance;
    }}
}} // end spatial mean kernel

    '''

    def generate_variant3_(self):

        MIO_BN_MAXN = 65
        MIO_BN_LDS_SIZE = self.ldsnogcn


        compute_mean_var = '''
            minibatch[n] = (Dtype)(*(in + index));
            mean += minibatch[n];
            variance = mad(minibatch[n], minibatch[n], variance);
        ''' if self.batch_size < MIO_BN_MAXN else '''
            Dtype xin = (Dtype)(*(in + index));
            mean += xin;
            variance = mad(xin, xin, variance);
        '''



        if self.batch_size < MIO_BN_MAXN:
            assign_inhat = 'inhat = (minibatch[n] - mean) * invVariance;'
        else:
            assign_inhat = 'inhat = ((Dtype)(*(in + index)) - mean) * invVariance;'


        return f'''


// This kernel implies the image is greater than a wavefront, but smaller than 257
__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
{self.op_name}MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype inhat       = (Dtype)0.;
    Dtype pvscale, pvbias;

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    unsigned int index;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int cidx  = grpid * {self.in_cstride};

    {'Dtype minibatch[{}];'.format(self.batch_size) if self.batch_size < MIO_BN_MAXN else ' '}

    if(lid == 0)
    {{
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    // MEAN
    if(lid < {self.in_cstride})
    {{
        for(unsigned int n = 0; n < {self.batch_size}; n++)
        {{
            index        = n * {self.in_nstride} + cidx + lid;
            {compute_mean_var}
        }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local Dtype lcl_data[{MIO_BN_LDS_SIZE}];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = ({self.xlocalsize} >> 1); red > 256; red >>= 1)
    {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {self.op_name}regLDSreduce(&mean, lcl_data, lid, (Dtype){self.inhw});
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
    {self.op_name}regLDSreduce(&variance, lcl_data, lid, (Dtype){self.inhw});
    barrier(CLK_LOCAL_MEM_FENCE);


    barrier(CLK_LOCAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (Dtype){self.epsilon});

    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(lid < {self.in_cstride})
    {{
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

        for(unsigned int n = 0; n < {self.batch_size}; n++)
        {{ // apply normalization
            index      = n * {self.in_nstride} + cidx + lid;
            {assign_inhat}
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
        }} // end for
    }}     // end if


}} // end spatial norm

'''
