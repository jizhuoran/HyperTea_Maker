
#include "caffe/util/libdnn_util.hpp"

namespace caffe {


template<class T>
void LibdnnInfo::add_def(std::stringstream& ss,  // NOLINT
  const char* name, T value) {
ss << "#ifdef " << name << std::endl;
ss << "#undef " << name << std::endl;
ss << "#endif" << std::endl;
if (std::is_same<T, float>::value) {
  ss << "#define " << name << " (float) " << std::setprecision(32) << value
      << std::endl;
} else if (std::is_same<T, double>::value) {
  ss << "#define " << name << " (double) " << std::setprecision(32) << value
      << std::endl;
} else {
  ss << "#define " << name << " " << value << std::endl;
}
}

template<class T>
void LibdnnInfo::add_def(std::stringstream& ss,  // NOLINT
  const std::string name, T value) {
	add_def(ss, name.c_str(), value);
}

std::string LibdnnInfo::generate_gemm_core(bool dterm) {
  std::stringstream ss;

  // Temporary registers for A and B
  ss << "Dtype" << this->vwm_ << " Areg;" << std::endl;
  ss << "Dtype" << this->vwn_ << " Breg[WPTN/VWN];" << std::endl;

  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {" << std::endl;
  ss << "#pragma unroll " << 1 << std::endl;
  ss << "for (int ku=0; ku<TSK_UNROLL; ++ku) {" << std::endl;
  ss << "int k = kt + ku;" << std::endl;

  // Cache the values of Bsub in registers
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  ss << "int col = tidn + wn*VWN*RTSN;" << std::endl;
  for (int i = 0; i < this->vwn_; ++i) {
    ss << "VEC_" << this->vwn_ << "_" << i << "(Breg[wn])"
       << " = Bsub[k][col + " << (i*this->rtsn_)
       << "];" << std::endl;
  }
  ss << "}" << std::endl;

  // Perform the computation
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  ss << "int row = tidm + wm*VWM*RTSM;" << std::endl;
  for (int i = 0; i < this->vwm_; ++i) {
    ss << "VEC_" << this->vwm_ << "_" << i << "(Areg)" << " = Asub[row + " << (i*this->rtsm_)
       << "][k];" << std::endl;
  }
  if (dterm) {
    if (unroll_) {
      for (int i = 0; i < this->vwm_; ++i) {
        ss << "VEC_" << this->vwm_ << "_" << i << "(Dreg[wm]) " << "+= VEC_" << this->vwm_
           << "_" << i << "(Areg) * v_bmul;" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
    }
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  if (unroll_) {
    for (int n = 0; n < this->vwn_; ++n) {
      for (int m = 0; m < this->vwm_; ++m) {
        ss << "VEC_" << this->vwn_ << "_" << n << "(Creg[wm * VWM + " << m << "][wn])"
           << " += VEC_" << this->vwm_ << "_" << m << "(Areg)" << " * VEC_" << this->vwn_
           << "_" << n << "(Breg[wn]);" << std::endl;
      }
    }
  } else {
    for (int m = 0; m < this->vwm_; ++m) {
      ss << "Creg[wm * VWM + " << m << "][wn]"
         << " += VEC_"<< this->vwm_ << "_" << m << "(Areg)" << " * (Breg[wn]);"
         << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string LibdnnInfo::generate_accreg_init(bool dterm, bool load) {
  std::stringstream ss;

  if (dterm) {
    ss << "Dtype" << this->vwm_ << " Dreg[WPTM/VWM];" << std::endl;
  }
    ss << "Dtype" << this->vwn_ << " Creg[WPTM][WPTN/VWN];" << std::endl;

  // Initialize the accumulation registers
  if (load) {
    // Load
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
      ss << "int globalRow = offM + tidm + wm * RTSM;"
         << std::endl;
      ss << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];"
         << std::endl;
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "int globalRow = offM + tidm + wm * RTSM;"
       << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
    ss << "int globalCol = offN + tidn + wn * RTSN;"
       << std::endl;
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] = "
       << "Cptr[globalRow * N + globalCol];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero init
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
      if (unroll_) {
        for (int i = 0; i < this->vwm_; ++i) {
          ss << "VEC_" << this->vwm_ << "_" << i << "(Dreg[wm]) = 0.0;" << std::endl;
        }
      } else {
        ss << "Dreg[wm] = 0.0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
    if (unroll_) {
      for (int i = 0; i < this->vwn_; ++i) {
        ss << "VEC_" << this->vwn_ << "_" << i << "(Creg[wm][wn]) = 0.0;" << std::endl;
      }
    } else {
      ss << "Creg[wm][wn] = 0.0;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

  std::string LibdnnInfo::conv_fw_def() {
	  
	  std::stringstream ss;

	  add_def(ss, "v_g", group_);

	  int B_off = fmaps_in_;
	  int C_off = fmaps_out_;
	  for (int i = 0; i < im_in_shape_.size(); ++i) {
	    B_off *= im_in_shape_[i];
	    C_off *= im_out_shape_[i];
	  }
	  // Input image batch offset
	  add_def(ss, "v_B_off", B_off);
	  // Output image batch offset
	  add_def(ss, "v_C_off", C_off);

	  int imsi = 1;
	  int imso = 1;
	  for (int i = 0; i < im_in_shape_.size(); ++i) {
	    add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
	    imsi *= im_in_shape_[i];
	    add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
	    imso *= im_out_shape_[i];
	  }
	  add_def(ss, "v_imsi", imsi);
	  add_def(ss, "v_imso", imso);

	  for (int i = 0; i < kernel_shape_.size(); ++i) {
	    add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
	  }

	  for (int i = 0; i < pad_.size(); ++i) {
	    add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
	  }

	  for (int i = 0; i < stride_.size(); ++i) {
	    add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
	  }

	  for (int i = 0; i < dilation_.size(); ++i) {
	    add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
	  }

	  add_def(ss, "v_fin", fmaps_in_);
	  add_def(ss, "v_fout", fmaps_out_);

	  if (bias_term_) {
	    add_def(ss, "v_bmul", bias_multiplier_);
	  }

	  MG_FW_ = fmaps_out_;
	  M_FW_ = fmaps_out_ / group_;
	  N_FW_ = 1;
	  KG_FW_ = fmaps_in_;
	  K_FW_ = fmaps_in_ / group_;

	  for (int i = 0; i < im_in_shape_.size(); ++i) {
	    K_FW_ *= kernel_shape_[i];
	    KG_FW_ *= kernel_shape_[i];
	    N_FW_ *= im_out_shape_[i];
	  }

	  // GEMM definitions
	  add_def(ss, "MG", MG_FW_);
	  add_def(ss, "M", M_FW_);
	  add_def(ss, "N", N_FW_);
	  add_def(ss, "KG", KG_FW_);
	  add_def(ss, "K", K_FW_);

	  // Local memory padding
	  add_def(ss, "v_pad_A", 1);
	  add_def(ss, "v_pad_B", 1);

	  // The tile-size in dimension M
	  this->add_def(ss, "TSM", this->tsm_);
	  // The tile-size in dimension N
	  this->add_def(ss, "TSN", this->tsn_);
	  // The tile-size in dimension K
	  this->add_def(ss, "TSK", this->tsk_);
	  // TSK unrolling
	  this->add_def(ss, "TSK_UNROLL", this->tsk_unroll_);
	  // The work-per-thread in dimension M
	  this->add_def(ss, "WPTM", this->wptm_);
	  this->add_def(ss, "VWM", this->vwm_);
	  // The work-per-thread in dimension N
	  this->add_def(ss, "WPTN", this->wptn_);
	  this->add_def(ss, "VWN", this->vwn_);
	  // The reduced tile-size in dimension M
	  this->add_def(ss, "RTSM", this->rtsm_);
	  // The reduced tile-size in dimension N
	  this->add_def(ss, "RTSN", this->rtsn_);

	  // Loads-per-thread for A
	  add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
	  // Loads-per-thread for B
	  add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

	  // Num tiles needs to be next higher even integer
	  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
	  add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

	  return ss.str();

  }

  std::string LibdnnInfo::conv_fw_kernel(std::string name) {

	  std::stringstream ss;

	  // Forward kernel
	  ss << "__kernel" << std::endl;
	  ss << "__attribute__((reqd_work_group_size("
	     << rtsn_ << ", " << rtsm_ << ", 1)))" << std::endl;
	  ss << "__attribute__((vec_type_hint(Dtype"
	     << std::min(vwm_, vwn_) << ")))" << std::endl;
	  ss << "void " + name + "(";
	  ss << "__global const Dtype* __restrict im_in, ";
	  ss << "__global const Dtype* __restrict wg, ";
	  if (bias_term_) {
	    ss << "__global const Dtype* __restrict bias, ";
	  }
	  ss << "__global Dtype* __restrict im_out";
	  ss << ") {" << std::endl;

	  // Thread identifiers
	  // Local row ID (max: RTSM=TSM/WPTM)
	  ss << "const int tidn = get_local_id(0);" << std::endl;
	  // Local col ID (max: RTSN=TSN/WPTN)
	  ss << "const int tidm = get_local_id(1);" << std::endl;
	  // Work-group offset
	  ss << "const int offN = TSN*get_group_id(0);" << std::endl;
	  // Work-group offset
	  ss << "const int offM = TSM*get_group_id(1);" << std::endl;

	  // Local tile memory
	  // Asub for loading weights & shuffling the output
	  ss << "volatile __local Dtype Asub[" << tsm_ << "][" << tsk_ << " + v_pad_A];"
	     << std::endl;
	  // Bsub for loading the input image and shuffling the output image
	  ss << "volatile __local Dtype Bsub[" << tsk_ << "][" << tsn_ << " + v_pad_B];"
	     << std::endl;

	  // Batch and group
	  if (group_ > 1) {
	    ss << "int group = get_global_id(2) % v_g;" << std::endl;
	    ss << "int batch = get_global_id(2) / v_g;" << std::endl;
	  } else {
	    ss << "int batch = get_global_id(2);" << std::endl;
	  }

	  if (group_ > 1) {
	    ss << "__global const Dtype* Aptr = wg + group * (M * K);" << std::endl;
	    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch "
	       << "+ group * (v_B_off / v_g);" << std::endl;
	    ss << "__global Dtype* Cptr = im_out + v_C_off * batch + group * (M * N);"
	       << std::endl;
	    if (bias_term_) {
	      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
	         << std::endl;
	    }
	  } else {
	    ss << "__global const Dtype* Aptr = wg;" << std::endl;
	    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
	    ss << "__global Dtype* Cptr = im_out + v_C_off * batch;" << std::endl;
	    if (bias_term_) {
	      ss << "__global const Dtype* Dptr = bias;" << std::endl;
	    }
	  }

	  // Initialize the accumulation registers
	  ss << "{" << std::endl;  // Scoping for C registers
	  ss << generate_accreg_init(false, false);

	  ss << "{" << std::endl;  // Scoping for load & compute block
	  // Loop over all tiles
	  ss << "#pragma unroll 1" << std::endl;
	  ss << "for (int t = 0; t < v_num_tiles; ++t) {" << std::endl;

	  // Load one tile of A into local memory
	  ss << "{" << std::endl;  // Scoping for loading A

	    ss << "#pragma unroll 4" << std::endl;
	    ss << "for (int la = 0; la < LPTA; ++la) {" << std::endl;
	    ss << "int tid = tidm * RTSN + tidn;" << std::endl;
	    ss << "int id = la * RTSN * RTSM + tid;" << std::endl;
	    ss << "int row = id / TSK;" << std::endl;
	    ss << "int col = id % TSK;" << std::endl;
	    ss << "int tiledIndex = TSK * t + col;" << std::endl;
	    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
	    ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
	    ss << "} else {" << std::endl;  // M-K-Guard
	    ss << "Asub[row][col] = 0.0;" << std::endl;
	    ss << "}" << std::endl;
	    ss << "}" << std::endl;  // LPTA

	  ss << "}" << std::endl;  // Scoping for loading A

	  // Load one tile of B into local memory
	  ss << "{" << std::endl;  // Scoping for loading B
	  ss << "#pragma unroll 4" << std::endl;
	  ss << "for (int lb = 0; lb < LPTB; ++lb) {" << std::endl;
	  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
	  ss << "int id = lb * RTSN * RTSM + tid;" << std::endl;
	  ss << "int col = id % TSN;" << std::endl;
	  ss << "int row = id / TSN;" << std::endl;
	  ss << "int tiledIndex = TSK * t + row;" << std::endl;

	  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;
	  // Define temporary registers
	  for (int i = 0; i < num_axes_; ++i) {
	    ss << "int d_iter_" << i << ";" << std::endl;
	    ss << "int d_temp_" << i << ";" << std::endl;
	  }

	  ss << "int imageIndex = offN + col;" << std::endl;
	  for (int i = num_axes_ - 1; i >= 0; --i) {
	    // Compute d_iter, final tiledIndex becomes input feature map ID
	    // Scale d_iter by the dilation factor
	    ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
	       << ";" << std::endl;
	    ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

	    // Compute d_temp
	    // Scale d_temp by the stride and subtract the padding
	    ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ") * v_s_" << i
	       << " - v_p_" << i << ";" << std::endl;
	    ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
	  }

	  // Recombine final index, compute in-range
	  if (!skip_range_check_) {
	    ss << "bool in_range = true;" << std::endl;
	  }
	  ss << "int d_iter_im;" << std::endl;
	  for (int i = 0; i < num_axes_; ++i) {
	    // Here, d_temp_ represents the column shift,
	    // while d_iter_ is the kernel shift
	    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
	    ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im;"
	       << std::endl;
	    if (!skip_range_check_) {
	      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i << ";"
	         << std::endl;
	    }
	  }

	  if (!skip_range_check_) {
	    ss << "if (in_range) {" << std::endl;
	  }
	  // tiledIndex now holds the memory offset for the input image
	  ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
	  if (!skip_range_check_) {
	    ss << "} else {" << std::endl;
	    ss << "Bsub[row][col] = 0.0;" << std::endl;
	    ss << "}" << std::endl;
	  }
	  ss << "} else {" << std::endl;
	  ss << "Bsub[row][col] = 0.0;" << std::endl;
	  ss << "}" << std::endl;
	  ss << "}" << std::endl;
	  ss << "}" << std::endl;  // Scoping for loading B

	  // Synchronize to make sure the tile is loaded
	  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	  ss << generate_gemm_core(false) << std::endl;

	  // Synchronize before loading the next tile
	  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

	  // Loop over all tiles
	  ss << "}" << std::endl;
	  ss << "}" << std::endl;  // Scoping for load & compute block

	  // Store the final results in C
	  ss << "#pragma unroll" << std::endl;
	  ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
	  ss << "int globalRow = offM + tidm + wm * RTSM;"
	     << std::endl;
	  if (bias_term_) {
	    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
	  }
	  ss << "#pragma unroll" << std::endl;
	  ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
	  ss << "int globalCol = offN + tidn + wn * RTSN;"
	     << std::endl;
	  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
	  if (bias_term_) {
	    ss << "Cptr[globalRow * N + globalCol] = "
	       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + v_bmul * biasval;"
	       << std::endl;
	  } else {
	    ss << "Cptr[globalRow * N + globalCol] = "
	       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
	  }
	  ss << "}" << std::endl;   // M-N-Guard
	  ss << "}" << std::endl;   // For (N)
	  ss << "}" << std::endl;   // For (M)
	  ss << "}" << std::endl;   // Scoping for C registers

	  // Kernel
	  ss << "}" << std::endl;

	  return ss.str();
  }


  std::string LibdnnInfo::deconv_fw_def() {

	  std::stringstream ss;

  // Groups
  add_def(ss, "v_g", this->group_);

  int A_off = this->fmaps_in_ * this->fmaps_out_;
  int B_off = this->fmaps_in_;
  int C_off = this->fmaps_out_;
  for (int i = 0; i < this->im_in_shape_.size(); ++i) {
    A_off *= this->kernel_shape_[i];
    B_off *= this->im_in_shape_[i];
    C_off *= this->im_out_shape_[i];
  }

  // Weight offset (only used for groups)
  add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  add_def(ss, "v_C_off", C_off);

  int imsi = 1;
  int imso = 1;
  for (int i = 0; i < this->im_in_shape_.size(); ++i) {
    add_def(ss, "v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    add_def(ss, "v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  add_def(ss, "v_imsi", imsi);
  add_def(ss, "v_imso", imso);

  int v_ks = 1;
  for (int i = 0; i < this->kernel_shape_.size(); ++i) {
    add_def(ss, "v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
    v_ks *= this->kernel_shape_[i];
  }
  add_def(ss, "v_ks", v_ks);


	for (int i = 0; i < this->pad_.size(); ++i) {
	  add_def(ss, "v_p_" + std::to_string(i), this->pad_[i]);
	}

  for (int i = 0; i < this->stride_.size(); ++i) {
    add_def(ss, "v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int i = 0; i < this->dilation_.size(); ++i) {
    add_def(ss, "v_d_" + std::to_string(i), this->dilation_[i]);
  }

  add_def(ss, "v_fin", this->fmaps_in_);
  add_def(ss, "v_fout", this->fmaps_out_);

  if (this->bias_term_) {
    add_def(ss, "v_bmul", this->bias_multiplier_);
  }

	this->MG_FW_ = this->fmaps_out_;
	this->M_FW_ = this->fmaps_out_ / this->group_;
	this->N_FW_ = 1;
	this->KG_FW_ = this->fmaps_in_;
	this->K_FW_ = this->fmaps_in_ / this->group_;

	for (int i = 0; i < this->im_in_shape_.size(); ++i) {
	  this->K_FW_ *= this->kernel_shape_[i];
	  this->KG_FW_ *= this->kernel_shape_[i];
	  this->N_FW_ *= this->im_out_shape_[i];
	}



  // GEMM definitions
  add_def(ss, "MG", this->MG_FW_);
  add_def(ss, "M", this->M_FW_);
  add_def(ss, "N", this->N_FW_);
  add_def(ss, "KG", this->KG_FW_);
  add_def(ss, "K", this->K_FW_);

  // Local memory padding
  add_def(ss, "v_pad_A", 1);
  add_def(ss, "v_pad_B", 1);

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  // The tile-size in dimension M
  this->add_def(ss, "TSM", this->tsm_);
  // The tile-size in dimension N
  this->add_def(ss, "TSN", this->tsn_);
  // The tile-size in dimension K
  this->add_def(ss, "TSK", this->tsk_);
  // TSK unrolling
  this->add_def(ss, "TSK_UNROLL", this->tsk_unroll_);
  // The work-per-thread in dimension M
  this->add_def(ss, "WPTM", this->wptm_);
  this->add_def(ss, "VWM", this->vwm_);
  // The work-per-thread in dimension N
  this->add_def(ss, "WPTN", this->wptn_);
  this->add_def(ss, "VWN", this->vwn_);
  // The reduced tile-size in dimension M
  this->add_def(ss, "RTSM", this->rtsm_);
  // The reduced tile-size in dimension N
  this->add_def(ss, "RTSN", this->rtsn_);
  // Loads-per-thread for A
  this->add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  this->add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
  }

  std::string LibdnnInfo::deconv_fw_kernel(std::string name) {
	  std::stringstream ss;
  	
  // Backward kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn_ << ", " << rtsm_ << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm_, vwn_) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_out, ";
  ss << "__global const Dtype* __restrict wg, ";
  if (this->bias_term_) {
    ss << "__global const Dtype* __restrict bias, ";
  }
  ss << "__global Dtype* __restrict im_in";
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile __local Dtype Asub[" << tsm_ << "][" << tsk_ << " + v_pad_A];"
     << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile __local Dtype Bsub[" << tsk_ << "][" << tsn_ << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (this->group_ > 1) {
    ss << "int group = get_global_id(2) % v_g;" << std::endl;
    ss << "int batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (v_A_off / (v_g * v_g));"
       << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch "
       << "+ group * (v_C_off / v_g);" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
          << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch;" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias;" << std::endl;
    }
  }


  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  ss << "for (int la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int row = id / TSK;" << std::endl;
  ss << "int col = id % TSK;" << std::endl;
  ss << "int tiledIndex = TSK * t + col;" << std::endl;

    // Load weights (wg) into Asub, flip fin/fout and inverse spatially
    // Compute kidx and midx, the column and row index of the
    // weights in the original A (weights) matrix
    ss << "int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;"
       << std::endl;
    ss << "int midx = tiledIndex / v_ks;" << std::endl;
    // Check range of the spatially flipped, fin/fout inverted weights
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    // Access weights with the original (translated) weight indices
    ss << "Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];"
       << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;



  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int col = id % TSN;" << std::endl;
  ss << "int row = id / TSN;" << std::endl;
  ss << "int tiledIndex = TSK * t + row;" << std::endl;

  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;

    // Load from B with im2col transformation

    // Define temporary registers
    for (int i = 0; i < this->num_axes_; ++i) {
      ss << "int d_iter_" << i << ";" << std::endl;
      ss << "int d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;

    ss << "int imageIndex = offN + col;" << std::endl;
    for (int i = this->num_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Subtract the padding from d_temp, note v_p_i can be negative
      ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ")"
         << " - v_p_" << i << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
    }

    ss << "int d_iter_im;" << std::endl;
    for (int i = 0; i < this->num_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im / v_s_"
         << i << ";" << std::endl;
      // In range: Not before or after actual image data
      // and not between image strides
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i
         << " * v_s_" << i << " && d_iter_im % v_s_" << i << " == 0;"
         << std::endl;
    }

    ss << "if (in_range) {" << std::endl;
    // tiledIndex now holds the memory offset for the input image
    ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
    ss << "} else {" << std::endl;
    // Out of B's image dimensions
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;


  ss << "} else {" << std::endl;
  // Out of B's matrix dimensions
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << this->generate_gemm_core(false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int globalRow = offM + tidm + wm * RTSM;" <<std::endl;
  if (this->bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int globalCol = offN + tidn + wn * RTSN;" << std::endl;

    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Cptr[globalRow * N + globalCol] = ";
    if (this->bias_term_) {
      ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]"
         << " + v_bmul * biasval;" << std::endl;
    } else {
      ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
    }
    ss << "}" << std::endl;

  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
  }

} //namespace caffe