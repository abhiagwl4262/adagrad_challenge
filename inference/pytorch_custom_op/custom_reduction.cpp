// "Copyright 2021 <abhishek agrawal>"
// #include <bits/stdc++.h>
#include <ctime>
#include <torch/script.h>
// #include "Eigen/Dense"

torch::Device device(torch::kCUDA);

torch::Tensor repeatInterleave(torch::Tensor input) {
  /// Implements the repeatInterleave function which
  /// resizes the input to the desired size op [512].
  /// We will do this by repeating the values by "repeat"
  /// number of times

  int dim = input.sizes()[0];
  int repeat = 512/dim; //dim=64
  /* First Implementation  
  torch::Tensor output = torch::zeros(512);
  for (int i=0; i < dim; i++) {
    for (int j=0; j < repeat; j++) {  
      output[i*repeat+j] = input[i];
    }
  } */ 

  /*  Optimized Implementation, Little bit extra memory required */
  float out[512];
  float* input_arr = input.data<float>();
  for (int i=0; i < dim; i++) {
    for (int j=0; j < repeat; j++) {  
      out[i*repeat+j] = *(input_arr+i);
    }
  }
  torch::Tensor output = torch::from_blob(out, {512}).clone();
  /*  Optimized Implementation, Little bit extra memory required */
  return output;

}

torch::Tensor reduction(
                      torch::Tensor layerOne,
                      torch::Tensor layerTwo,
                      torch::Tensor layerThree,
                      torch::Tensor layerFour) {
  /// Implements the Reduction ONNX operator by interleaving the
  /// param layerOne: layer 1 output after AdaptiveAvgPooling [1x64] or [64]
  /// param layerTwo: layer 2 output after AdaptiveAvgPooling [1x128] or [128]
  /// param layerThree: layer 3 output after AdaptiveAvgPooling [1x256] or [256]
  /// param layerFour: layer 4 output after AdaptiveAvgPooling [1x512] or [512]
  /// return torch::Tensor: reduction output of the tensors [1x512] or [512]
    torch::Tensor X1_out, X2_out, X3_out;
    // torch::Tensor X1_out = torch::zeros(512, device);
    // torch::Tensor X2_out = torch::zeros(512, device);
    // torch::Tensor X3_out = torch::zeros(512, device);
    //clock_t begin = clock();
    X1_out = repeatInterleave(layerOne);
    X2_out = repeatInterleave(layerTwo);
    X3_out = repeatInterleave(layerThree);
    //clock_t end = clock();
    //double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //std::cout << "Time taken by C++ interleaves is : " << elapsed_secs << std::endl;;
    // torch::Tensor output = torch::zeros(512, device);
    torch::Tensor output = torch::zeros(512);
    // output = (X1_out.to(device) + X2_out.to(device) + X3_out.to(device) + layerFour)/4.0;
    //begin = clock();
    output = (X1_out + X2_out + X3_out + layerFour)/4.0;
    //end = clock();
    //elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    //std::cout << "Time taken by C++ sum reduction is : " << elapsed_secs << std::endl;;
    //std::cout << output << std::endl;;
    return output;
}

static auto registry = torch::RegisterOperators("adagradChallenge::reduction", &reduction);
