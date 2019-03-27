//
//  main.m
//  metal_mac
//
//  Created by Tec GSQ on 27/11/2017.
//  Copyright © 2017 Tec GSQ. All rights reserved.
//

#include "caffe/caffe.hpp"



int main(int argc, char** argv) {

  // if (argc != 4) {
  //   LOG(INFO) << "./caffemodel_convertor.bin prototxt_file fp32.caffemodel(input) fp16.caffemodel(output)";
  //   exit(0);
  // }
  caffe::Net<float> *_net;

  caffe::Caffe::Get().set_mode(caffe::Caffe::CPU);

  std::string argv_fake[4] = {"", "./examples/style_transfer/style.prototxt", "./examples/style_transfer/a1.caffemodel", "new_net"};


  _net = new caffe::Net<float>(argv_fake[1], caffe::TEST);
  _net->CopyTrainedLayersFrom(argv_fake[2]);
  

  // caffe::hypertea_func *hypertea = &caffe::Caffe::Get().hypertea_code_generator;


  auto input_blobs = _net->input_blob_indices();
  auto output_blobs = _net->output_blob_indices();

  for (auto const & id : input_blobs) {
    caffe::hypertea_func::Get().input_names.push_back(_net->blob_names()[id]);
  }

  for (auto const & id : output_blobs) {
    caffe::hypertea_func::Get().output_names.push_back(_net->blob_names()[id]);
  }

  _net->Forward();

  std::cout << caffe::hypertea_func::Get().hypertea_gpu(argv_fake[3]) << std::endl;


  std::ofstream myfile ("/home/zrji/hypertea/tools/reference.hpp");
  if (myfile.is_open()) {
    myfile << caffe::hypertea_func::Get().hypertea_gpu(argv_fake[3]) << std::endl;

  } else {
    std::cout << "wanner???" << std::endl;
  }





}
