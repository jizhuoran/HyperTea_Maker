def deconv_fw_kernel(self, name):

    iter_init = self.new_line_joiner_(16).join([
        'd_iter_{0} = (tiledIndex % v_k_{0}) * v_d_{0};',
        'tiledIndex = tiledIndex / v_k_{0};',
        'd_temp_{0} = (imageIndex % v_imso_{0}) * v_s_{0} - v_p_{0};',
        'imageIndex = imageIndex / v_imso_{0};'
        ])
    iter_init = self.range_declare(iter_init, 16, range(self.num_axes, -1, -1))



    assign_Bsub = [
        'if (in_range) {',
        '   Bsub[row][col] = Bptr[tiledIndex];',
        '} else {',
        '   Bsub[row][col] = 0.0;',
        '}'
    ] if not self.skip_range_check else ['Bsub[row][col] = Bptr[tiledIndex];']



    assign_d_iter = self.new_line_joiner_(16).join([
        'd_iter_im = d_temp_{0} + d_iter_{0}',
        'tiledIndex = tiledIndex * v_imsi_{0} + d_iter_im / v_s_{0};',
        'in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_{0} * v_s_{0} && d_iter_im % v_s_{} == 0;' if self.skip_range_check else ' '
    ])
    assign_d_iter = self.range_declare(assign_d_iter, 16, range(self.num_axes))


    group_decider = [
      ' % v_g',
      'int batch = get_global_id(2) / v_g;',
      ' + + group * (v_A_off / (v_g * v_g))',
      ' + group * (v_B_off / v_g)',
      ' + group * (v_C_off / v_g)',
      ' + group * (v_fout / v_g)',
    ] if self.group > 1 else ['','','','', '', '']




    return f'''
__kernel
__attribute__((reqd_work_group_size({self.rtsn}, {self.rtsm}, 1)))
__attribute__((vec_type_hint(Dtype{min(self.vwm, self.vwn)})))
void {name}(
__global const Dtype* __restrict im_in, 
__global const Dtype* __restrict wg, 
{'__global const Dtype* __restrict bias,' if self.bias_term else ''}
__global Dtype* __restrict im_out) {{

const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);

volatile __local Dtype Asub[{self.tsm}][{self.tsk} + v_pad_A];
volatile __local Dtype Bsub[{self.tsk}][{self.tsn} + v_pad_B];

int group = get_global_id(2){group_decider[0]};
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
            {self.range_declare('int d_iter_{};', 16, range(self.num_axes))}
            {self.range_declare('int d_temp_{};', 16, range(self.num_axes))}


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
        "Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + {"v_bmul * biasval;" if self.bias_term else ";"}
      }}
    }}
  }}
}}
}} 
    '''