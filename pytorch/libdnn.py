from functools import reduce
import itertools


class LibdnnGenerator(object):
  """docstring for LibdnnGenerator"""
  def __init__(self, precision, kernels):
    super(LibdnnGenerator, self).__init__()
    self.precision = precision

    self.libdnn_conv_code = ''.join(kernels)

    self.header = '''
#define Dtype {0}
#define Dtype1 {0}
#define Dtype2 {0}2
#define Dtype4 {0}4
#define Dtype8 {0}8
#define Dtype16 {0}16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
    '''.format(self.precision)



  def generate_conv_code(self):

    return f'''
std::string conv_opencl_funcs = R"(
{self.header}
{self.libdnn_conv_code}
)";
        '''


    

class Libdnn(object):
    """docstring for Libdnn"""
    def __init__(self, 
        name, group,
        op_type, bias_term, 
        in_shape, out_shape, 
        kernel_shape, pad, 
        stride, dilation):

        super(Libdnn, self).__init__()
            
        self.name = name;

        self.is_deconv = True if op_type == "DeconvolutionOp" else False

        self.bias_term = bias_term;
        self.bias_multiplier = 0.0 if bias_term is None else 1.0

        self.dims = len(in_shape)
        self.spatial_dims = len(kernel_shape)

        # print(in_shape)
        # print(kernel_shape)

        self.num_axes = self.spatial_dims
        self.fmaps_in = in_shape[self.dims - self.spatial_dims - 1]
        self.fmaps_out = out_shape[self.dims - self.spatial_dims - 1]
        self.group = group

        self.skip_range_check = all(list(map(lambda x: x <= 0, pad)))


        self.batch_size = in_shape[0]

        self.im_in_shape = in_shape[(self.dims - self.spatial_dims):]
        self.im_out_shape = out_shape[(self.dims - self.spatial_dims):]
        self.kernel_shape = kernel_shape
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
    

        self.vwm = 4
        self.vwn = 4
        self.tsk_unroll = 8
        self.wptm = 4
        self.wptn = 8
        self.rtsm = 4
        self.rtsn = 16
        self.tsk = 8
        self.tsm = self.wptm * self.rtsm
        self.tsn = self.wptn * self.rtsn
        self.lpta = (self.tsm * self.tsk) / (self.rtsm * self.rtsn)
        self.lptb = (self.tsn * self.tsk) / (self.rtsm * self.rtsn)
        self.unroll = True







    def generate_libdnn_code(self):
        if not self.is_deconv:
            return self.conv_fw_def_() + self.conv_fw_kernel_(self.name)
        else:
            return self.deconv_fw_def_() + self.deconv_fw_kernel_(self.name)

    def local_shape(self):
      return [self.rtsn, self.rtsm , 1]

    def global_shape(self):
      return [((self.prod_(self.im_out_shape) - 1) // self.tsn + 1) * self.rtsn,
              (((self.fmaps_out // self.group) - 1) // self.tsm + 1) * self.rtsm,
              self.batch_size * self.group
              ]


    
    # def range_declare_(self, line, space, iter_range):
        # return ('\n' + ' ' * space).join([line.format(i) for i in iter_range])

    def range_declare_(self, line, space, iter_range, map_func = lambda x:(x,)):
        return ('\n' + ' ' * space).join([line.format(*map_func(i)) for i in iter_range])



    def new_line_joiner_(self, space):
        return '\n' + ' ' * space

    def conv_fw_kernel_(self, name):
    
        iter_init = self.new_line_joiner_(16).join([
            'd_iter_{0} = (tiledIndex % v_k_{0}) * v_d_{0};',
            'tiledIndex = tiledIndex / v_k_{0};',
            'd_temp_{0} = (imageIndex % v_imso_{0}) * v_s_{0} - v_p_{0};',
            'imageIndex = imageIndex / v_imso_{0};'
            ])
        iter_init = self.range_declare_(iter_init, 16, range(self.num_axes-1, -1, -1))



        assign_Bsub = [
            'if (in_range) {',
            '   Bsub[row][col] = Bptr[tiledIndex];',
            '} else {',
            '   Bsub[row][col] = 0.0;',
            '}'
        ] if not self.skip_range_check else ['Bsub[row][col] = Bptr[tiledIndex];']



        assign_d_iter = self.new_line_joiner_(16).join([
            'd_iter_im = d_temp_{0} + d_iter_{0};',
            'tiledIndex = tiledIndex * v_imsi_{0} + d_iter_im;',
            'in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_{0};' if not self.skip_range_check else ' '
        ])
        assign_d_iter = self.range_declare_(assign_d_iter, 16, range(self.num_axes))


        group_decider = [
          'int group = get_global_id(2) % v_g;',
          ' / v_g',
          ' + group * (M * K)',
          ' + group * (v_B_off / v_g)',
          ' + group * (M * N)',
          ' + group * (v_fout / v_g)',
        ] if self.group > 1 else ['','','','', '', '']




        return f'''
__kernel
__attribute__((reqd_work_group_size({self.rtsn}, {self.rtsm}, 1)))
__attribute__((vec_type_hint(Dtype{min(self.vwm, self.vwn)})))
void {name}(
  __global const Dtype* __restrict im_in, 
  __global const Dtype* __restrict wg,
  __global Dtype* __restrict im_out 
  {', __global const Dtype* __restrict bias' if self.bias_term else ''}
  ) {{

    const int tidn = get_local_id(0);
    const int tidm = get_local_id(1);
    const int offN = TSN*get_group_id(0);
    const int offM = TSM*get_group_id(1);

    volatile __local Dtype Asub[{self.tsm}][{self.tsk} + v_pad_A];
    volatile __local Dtype Bsub[{self.tsk}][{self.tsn} + v_pad_B];

    {group_decider[0]}
    int batch = get_global_id(2){group_decider[1]};
    

    __global const Dtype* Aptr = wg{group_decider[2]};
    __global const Dtype* Bptr = im_in + v_B_off * batch{group_decider[3]};
    __global Dtype* Cptr = im_out + v_C_off * batch{group_decider[4]};
    {"__global const Dtype* Dptr = bias{};".format(group_decider[5]) if self.bias_term else ' '}
    {{
      {self.generate_accreg_init_(False, False, 10)}
      
      {{
        #pragma unroll 1
        for (int t = 0; t < v_num_tiles; ++t) {{
          {{

            #pragma unroll 4
            for (int la = 0; la < LPTA; ++la) {{
              int tid = tidm * RTSN + tidn;
              int id = la * RTSN * RTSM + tid;
              int row = id / TSK;
              int col = id % TSK;
              int tiledIndex = TSK * t + col;
              
              if ((offM + row) < M && tiledIndex < K) {{
                Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
              }} else {{  
                Asub[row][col] = 0.0;
              }}
            }}  
          }}  

    
          {{  
            #pragma unroll 4
            for (int lb = 0; lb < LPTB; ++lb) {{
              int tid = tidm * RTSN + tidn;
              int id = lb * RTSN * RTSM + tid;
              int col = id % TSN;
              int row = id / TSN;
              int tiledIndex = TSK * t + row;

              if ((offN + col) < N && tiledIndex < K) {{
                {self.range_declare_('int d_iter_{};', 16, range(self.num_axes))}
                {self.range_declare_('int d_temp_{};', 16, range(self.num_axes))}

                int imageIndex = offN + col;

                {iter_init}

                {"bool in_range = true;" if not self.skip_range_check else ' '}

                int d_iter_im;

                {assign_d_iter}
                
                {self.new_line_joiner_(16).join(assign_Bsub)}
              }} else {{
                Bsub[row][col] = 0.0;
              }}
            }}
          }}

          barrier(CLK_LOCAL_MEM_FENCE);

          {self.generate_gemm_core_(False, 10)}

          barrier(CLK_LOCAL_MEM_FENCE);
        }}
      }}


      #pragma unroll
      for (int wm=0; wm<WPTM; ++wm) {{
        int globalRow = offM + tidm + wm * RTSM;
        
        {'Dtype biasval = Dptr[globalRow];' if self.bias_term else ' '}

        #pragma unroll
        for (int wn=0; wn<WPTN; ++wn) {{
          int globalCol = offN + tidn + wn * RTSN;
        
          if (globalRow < M && globalCol < N) {{
            Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]{" + biasval;" if self.bias_term else ";"}
          }}
        }}
      }}
    }}
  }} 

        '''
  



    def deconv_fw_kernel_(self, name):

        iter_init = self.new_line_joiner_(16).join([
            'd_iter_{0} = (tiledIndex % v_k_{0}) * v_d_{0};',
            'tiledIndex = tiledIndex / v_k_{0};',
            'd_temp_{0} = (imageIndex % v_imso_{0}) - v_p_{0};',
            'imageIndex = imageIndex / v_imso_{0};'
            ])
        iter_init = self.range_declare_(iter_init, 16, range(self.num_axes - 1, -1, -1))



        assign_Bsub = [
            'if (in_range) {',
            '   Bsub[row][col] = Bptr[tiledIndex];',
            '} else {',
            '   Bsub[row][col] = 0.0;',
            '}'
        ] if not self.skip_range_check else ['Bsub[row][col] = Bptr[tiledIndex];']



        assign_d_iter = self.new_line_joiner_(16).join([
            'd_iter_im = d_temp_{0} + d_iter_{0};',
            'tiledIndex = tiledIndex * v_imsi_{0} + d_iter_im / v_s_{0};',
            'in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_{0} * v_s_{0} && d_iter_im % v_s_{0} == 0;' if not self.skip_range_check else ' '
        ])
        assign_d_iter = self.range_declare_(assign_d_iter, 16, range(self.num_axes))


        group_decider = [
          ' % v_g',
          'int group = get_global_id(2) / v_g;',
          ' + group * (v_A_off / (v_g * v_g))',
          ' + group * (v_B_off / v_g)',
          ' + group * (v_C_off / v_g)',
          ' + group * (v_fout / v_g)',
        ] if self.group > 1 else ['','','','', '', '']




        return f'''
    __kernel
    __attribute__((reqd_work_group_size({self.rtsn}, {self.rtsm}, 1)))
    __attribute__((vec_type_hint(Dtype{min(self.vwm, self.vwn)})))
    void {name}(
    __global const Dtype* __restrict im_out, 
    __global const Dtype* __restrict wg, 
    __global Dtype* __restrict im_in
    {', __global const Dtype* __restrict bias' if self.bias_term else ''}
    ) {{

    const int tidn = get_local_id(0);
    const int tidm = get_local_id(1);
    const int offN = TSN*get_group_id(0);
    const int offM = TSM*get_group_id(1);

    volatile __local Dtype Asub[{self.tsm}][{self.tsk} + v_pad_A];
    volatile __local Dtype Bsub[{self.tsk}][{self.tsn} + v_pad_B];

    int batch = get_global_id(2){group_decider[0]};
    {group_decider[1]}

    __global const Dtype* Aptr = wg{group_decider[2]};
    __global const Dtype* Bptr = im_out + v_B_off * batch{group_decider[3]};
    __global Dtype* Cptr = im_in + v_C_off * batch{group_decider[4]};
    {"__global const Dtype* Dptr = bias{};".format(group_decider[5]) if self.bias_term else ' '}
    {{
      {self.generate_accreg_init_(False, False, 10)}
      
      {{
        #pragma unroll 1
        for (int t = 0; t < v_num_tiles; ++t) {{
          {{

            #pragma unroll 4
            for (int la = 0; la < LPTA; ++la) {{
              int tid = tidm * RTSN + tidn;
              int id = la * RTSN * RTSM + tid;
              int row = id / TSK;
              int col = id % TSK;
              int tiledIndex = TSK * t + col;
              
              int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;
              int midx = tiledIndex / v_ks;

              if ((offM + row) < M && tiledIndex < K) {{
                Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];
              }} else {{  
                Asub[row][col] = 0.0;
              }}
            }}  
          }}  


          {{  
            #pragma unroll 4
            for (int lb = 0; lb < LPTB; ++lb) {{
              int tid = tidm * RTSN + tidn;
              int id = lb * RTSN * RTSM + tid;
              int col = id % TSN;
              int row = id / TSN;
              int tiledIndex = TSK * t + row;

              if ((offN + col) < N && tiledIndex < K) {{
                {self.range_declare_('int d_iter_{};', 16, range(self.num_axes))}
                {self.range_declare_('int d_temp_{};', 16, range(self.num_axes))}


                int imageIndex = offN + col;

                {iter_init}

                {"bool in_range = true;" if not self.skip_range_check else ' '}

                int d_iter_im;

                {assign_d_iter}
                
                {self.new_line_joiner_(16).join(assign_Bsub)}
              }} else {{
                Bsub[row][col] = 0.0;
              }}
            }}
          }}

          barrier(CLK_LOCAL_MEM_FENCE);

          {self.generate_gemm_core_(False, 10)}

          barrier(CLK_LOCAL_MEM_FENCE);
        }}
      }}


      #pragma unroll
      for (int wm=0; wm<WPTM; ++wm) {{
        int globalRow = offM + tidm + wm * RTSM;
        
        {'Dtype biasval = Dptr[globalRow];' if self.bias_term else ' '}

        #pragma unroll
        for (int wn=0; wn<WPTN; ++wn) {{
          int globalCol = offN + tidn + wn * RTSN;
        
          if (globalRow < M && globalCol < N) {{
            Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]{" + biasval;" if self.bias_term else ";"}
          }}
        }}
      }}
    }}
  }} 
        '''



    def conv_fw_def_(self):
    
        conv_def = ''

        conv_def += self.add_def_("v_g", self.group)

        # Input image batch offset
        conv_def += self.add_def_("v_B_off", self.fmaps_in * self.prod_(self.im_in_shape))
        # Output image batch offset
        conv_def += self.add_def_("v_C_off", self.fmaps_out * self.prod_(self.im_out_shape))


        for i in range(len(self.im_in_shape)):
          conv_def += self.add_def_("v_imsi_{}".format(i), self.im_in_shape[i])
          conv_def += self.add_def_("v_imso_{}".format(i), self.im_out_shape[i])

        conv_def += self.add_def_("v_imsi", self.prod_(self.im_in_shape))
        conv_def += self.add_def_("v_imso", self.prod_(self.im_out_shape))


        for i in range(len(self.kernel_shape)):
          conv_def += self.add_def_("v_k_{}".format(i), self.kernel_shape[i])

        for i in range(len(self.pad)):
          conv_def += self.add_def_("v_p_{}".format(i), self.pad[i])

        for i in range(len(self.stride)):
          conv_def += self.add_def_("v_s_{}".format(i), self.stride[i])

        for i in range(len(self.dilation)):
          conv_def += self.add_def_("v_d_{}".format(i), self.dilation[i])



        conv_def += self.add_def_("v_fin", self.fmaps_in);
        conv_def += self.add_def_("v_fout", self.fmaps_out);

        if self.bias_term:
          conv_def += self.add_def_("v_bmul", self.bias_multiplier)


        MG_FW_ = self.fmaps_out
        M_FW_ = self.fmaps_out // self.group
        N_FW_ = self.prod_(self.im_out_shape)
        KG_FW_ = self.fmaps_in * self.prod_(self.kernel_shape)
        K_FW_ = self.fmaps_in // self.group * self.prod_(self.kernel_shape)

        # GEMM definitions
        conv_def += self.add_def_("MG", MG_FW_)
        conv_def += self.add_def_("M", M_FW_)
        conv_def += self.add_def_("N", N_FW_)
        conv_def += self.add_def_("KG", KG_FW_)
        conv_def += self.add_def_("K", K_FW_)

        # Local memory padding
        conv_def += self.add_def_("v_pad_A", 1)
        conv_def += self.add_def_("v_pad_B", 1)

        # The tile-size in dimension M
        conv_def += self.add_def_("TSM", self.tsm)
        # The tile-size in dimension N
        conv_def += self.add_def_("TSN", self.tsn)
        # The tile-size in dimension K
        conv_def += self.add_def_("TSK", self.tsk)
        # TSK unrolling
        conv_def += self.add_def_("TSK_UNROLL", self.tsk_unroll)
        # The work-per-thread in dimension M
        conv_def += self.add_def_("WPTM", self.wptm)
        conv_def += self.add_def_("VWM", self.vwm)
        # The work-per-thread in dimension N
        conv_def += self.add_def_("WPTN", self.wptn)
        conv_def += self.add_def_("VWN", self.vwn)
        # The reduced tile-size in dimension M
        conv_def += self.add_def_("RTSM", self.rtsm)
        # The reduced tile-size in dimension N
        conv_def += self.add_def_("RTSN", self.rtsn)

        # Loads-per-thread for A
        conv_def += self.add_def_("LPTA", "((TSK*TSM)/(RTSM*RTSN))")
        # Loads-per-thread for B
        conv_def += self.add_def_("LPTB", "((TSK*TSN)/(RTSM*RTSN))")

        # Num tiles needs to be next higher even integer
        # (due to some quirky bug in AMD OpenCL 2.0 on Windows)
        conv_def += self.add_def_("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

        return conv_def


    def deconv_fw_def_(self):
    
        deconv_def = ''

        deconv_def += self.add_def_("v_g", self.group);


        deconv_def += self.add_def_("v_A_off", self.fmaps_in * self.fmaps_out * self.prod_(self.kernel_shape))
        # Input image batch offset
        deconv_def += self.add_def_("v_B_off", self.fmaps_in * self.prod_(self.im_in_shape))
        # Output image batch offset
        deconv_def += self.add_def_("v_C_off", self.fmaps_out * self.prod_(self.im_out_shape))


        for i in range(len(self.im_in_shape)):
          deconv_def += self.add_def_("v_imsi_{}".format(i), self.im_in_shape[i])
          deconv_def += self.add_def_("v_imso_{}".format(i), self.im_out_shape[i])

        deconv_def += self.add_def_("v_imsi", self.prod_(self.im_in_shape))
        deconv_def += self.add_def_("v_imso", self.prod_(self.im_out_shape))


        for i in range(len(self.kernel_shape)):
          deconv_def += self.add_def_("v_k_{}".format(i), self.kernel_shape[i])

        deconv_def += self.add_def_("v_ks", self.prod_(self.kernel_shape));

        for i in range(len(self.pad)):
          deconv_def += self.add_def_("v_p_{}".format(i), (self.kernel_shape[i] - 1) * self.dilation[i] - self.pad[i])

        for i in range(len(self.stride)):
          deconv_def += self.add_def_("v_s_{}".format(i), self.stride[i])

        for i in range(len(self.dilation)):
          deconv_def += self.add_def_("v_d_{}".format(i), self.dilation[i])



        deconv_def += self.add_def_("v_fin", self.fmaps_in);
        deconv_def += self.add_def_("v_fout", self.fmaps_out);

        if self.bias_term:
          deconv_def += self.add_def_("v_bmul", self.bias_multiplier)


        MG_FW_ = self.fmaps_out;
        M_FW_ = self.fmaps_out // self.group;
        N_FW_ = self.prod_(self.im_out_shape);
        KG_FW_ = self.fmaps_in * self.prod_(self.kernel_shape)
        K_FW_ = self.fmaps_in // self.group * self.prod_(self.kernel_shape)

        # GEMM definitions
        deconv_def += self.add_def_("MG", MG_FW_);
        deconv_def += self.add_def_("M", M_FW_);
        deconv_def += self.add_def_("N", N_FW_);
        deconv_def += self.add_def_("KG", KG_FW_);
        deconv_def += self.add_def_("K", K_FW_);

        # Local memory padding
        deconv_def += self.add_def_("v_pad_A", 1);
        deconv_def += self.add_def_("v_pad_B", 1);

        # The tile-size in dimension M
        deconv_def += self.add_def_("TSM", self.tsm)
        # The tile-size in dimension N
        deconv_def += self.add_def_("TSN", self.tsn)
        # The tile-size in dimension K
        deconv_def += self.add_def_("TSK", self.tsk)
        # TSK unrolling
        deconv_def += self.add_def_("TSK_UNROLL", self.tsk_unroll)
        # The work-per-thread in dimension M
        deconv_def += self.add_def_("WPTM", self.wptm)
        deconv_def += self.add_def_("VWM", self.vwm)
        # The work-per-thread in dimension N
        deconv_def += self.add_def_("WPTN", self.wptn)
        deconv_def += self.add_def_("VWN", self.vwn)
        # The reduced tile-size in dimension M
        deconv_def += self.add_def_("RTSM", self.rtsm)
        # The reduced tile-size in dimension N
        deconv_def += self.add_def_("RTSN", self.rtsn)

        # Loads-per-thread for A
        deconv_def += self.add_def_("LPTA", "((TSK*TSM)/(RTSM*RTSN))");
        # Loads-per-thread for B
        deconv_def += self.add_def_("LPTB", "((TSK*TSN)/(RTSM*RTSN))");

        # Num tiles needs to be next higher even integer
        # (due to some quirky bug in AMD OpenCL 2.0 on Windows)
        deconv_def += self.add_def_("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

        return deconv_def





    def generate_accreg_init_(self, dterm, load, space):

        line_space = ' ' * space

        load_dterm = self.new_line_joiner_(space).join([
            '#pragma unroll',
            'for (int wm=0; wm<WPTM; ++wm) {',
            '  int globalRow = offM + tidm + wm * RTSM;',
            '  ((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];',
            '}'
        ]) if dterm else ''


        init_dterm = f'''
            #pragma unroll
            for (int wm=0; wm<WPTM/VWM; ++wm) {{
              {self.range_declare_('VEC_{}_{}(Dreg[wm]) = 0.0;', 
                        space + 4, range(self.vwm), 
                        lambda x:(self.vwm, x)
               ) if self.unroll else 'Dreg[wm] = 0.0;'
              }
            }}
        ''' if dterm else ''



        if load:
            return f'''
{line_space}{'Dtype{self.vwm} Dreg[WPTM/VWM];' if dterm else ''}
{line_space}Dtype{self.vwn} Creg[WPTM][WPTN/VWN];

{line_space}{load_dterm}

{line_space}#pragma unroll
{line_space}for (int wm=0; wm<WPTM; ++wm) {{
{line_space}  int globalRow = offM + tidm + wm * RTSM;
              
{line_space}  #pragma unroll
{line_space}  for (int wn=0; wn<WPTN; ++wn) {{
{line_space}    int globalCol = offN + tidn + wn * RTSN; 
{line_space}    if (globalRow < M && globalCol < N) {{
{line_space}      ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] = Cptr[globalRow * N + globalCol];
{line_space}    }}
{line_space}  }}
{line_space}}}
            '''

        else:
            return f'''
{line_space}Dtype{self.vwn} Creg[WPTM][WPTN/VWN];
{line_space}{init_dterm}
{line_space}#pragma unroll
{line_space}for (int wm=0; wm<WPTM; ++wm) {{
{line_space}  #pragma unroll
{line_space}  for (int wn=0; wn<WPTN/VWN; ++wn) {{
{line_space}    {self.range_declare_('VEC_{}_{}(Creg[wm][wn]) = 0.0;', 
                        space + 4, range(self.vwn), 
                        lambda x:(self.vwn, x)
                 ) if self.unroll else 'Creg[wm][wn] = 0.0;'
                }
{line_space}  }}
{line_space}}}

            '''
        
        


    def generate_gemm_core_(self, dterm, space):
        
        line_space = ' '*space

        if self.unroll:
            assign_Creg = self.range_declare_('VEC_{0}_{1}(Creg[wm * VWM + {2}][wn]) += VEC_{3}_{2}(Areg) * VEC_{0}_{1}(Breg[wn]);', 
                space + 8, itertools.product(range(self.vwn), range(self.vwm)), 
                lambda x: (self.vwn, x[0], x[1], self.vwm))
        else:
            assign_Creg = self.range_declare_('Creg[wm * VWM + {0}][wn] += VEC_{1}_{0}(Areg) * (Breg[wn]);', 
                space + 8, range(self.vwm), 
                lambda x:(x, self.vwm))

        if dterm:
            if self.unroll:
                assign_Dreg = self.range_declare_('VEC_{0}_{1}(Dreg[wm]) += VEC_{0}_{1}(Areg);', 
                    space + 6, range(self.vwm), 
                    lambda x: (self.vwm, x))
            else:
                assign_Dreg = 'Dreg[wm] += Areg;'
        else:
            assign_Dreg = ''



        return f'''
{line_space}Dtype{self.vwm} Areg;
{line_space}Dtype{self.vwn} Breg[WPTN/VWN];

{line_space}#pragma unroll {1}
{line_space}for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {{
{line_space}  #pragma unroll {1}
{line_space}  for (int ku=0; ku<TSK_UNROLL; ++ku) {{
{line_space}    int k = kt + ku;
    
{line_space}    #pragma unroll
{line_space}    for (int wn=0; wn<WPTN/VWN; ++wn) {{
{line_space}      int col = tidn + wn*VWN*RTSN;
{line_space}      {self.range_declare_('VEC_{}_{}(Breg[wn]) = Bsub[k][col + {}];', 
                                      space + 6, range(self.vwn), 
                                      lambda x: (self.vwn, x, x*self.rtsn))}
{line_space}    }}
    
{line_space}    #pragma unroll
{line_space}    for (int wm=0; wm<WPTM/VWM; ++wm) {{
{line_space}      int row = tidm + wm*VWM*RTSM;
{line_space}      {self.range_declare_('VEC_{}_{}(Areg) = Asub[row + {}][k];', 
                                      space + 6, range(self.vwm), 
                                      lambda x: (self.vwm, x, x*self.rtsm))}

{line_space}      {assign_Dreg}

{line_space}      #pragma unroll
{line_space}      for (int wn=0; wn<WPTN/VWN; ++wn) {{
{line_space}        {assign_Creg}
{line_space}      }}
{line_space}    }}
{line_space}  }}
{line_space}}}
        '''




    def add_def_(self, name, value):
        return '''
#ifdef {}
#undef {}
#endif
#define {} {}'''.format(name, name, name, value)

    def prod_(self, l):
        return reduce(lambda x, y: x*y, l)


# temp = Libdnn("deconv1", 1,
#         "DeconvolutionOp", True, 
#         [1, 32, 512, 512], [1, 3, 512, 512], 
#         [9, 9], [4, 4], [1, 1], [1, 1])



# # # print(temp.generate_gemm_core_(True))

# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')
# # # print(temp.generate_accreg_init_(True, False))


# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')
# # print('----------')


# print(temp.deconv_fw_def_())
# # # print(temp.deconv_fw_def_())

# print(temp.deconv_fw_kernel_("deconv1"))







