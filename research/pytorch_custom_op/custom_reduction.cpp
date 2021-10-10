// "Copyright 2021 <abhishek agrawal>"

#include <torch/script.h>
#include "Eigen/Dense"

torch::Tensor repeatInterleave(torch::Tensor input) {
  /// Implements the repeatInterleave function which
  /// resizes the input to the desired size op [512].
  /// We will do this by repeating the values by "repeat"
  /// number of times

  int repeat = 512/input.sizes()[0];
  torch::Tensor output = torch::zeros(512);
  for (int i=0; i < input.sizes()[0]; i++) {
    // output[i*repeat:(i+1)*repeat] = input[i];
    for (int j=0; j < repeat; j++) {
      output[i*repeat+j] = input[i];
    }
  }
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

    X1_out = repeatInterleave(layerOne);
    X2_out = repeatInterleave(layerTwo);
    X3_out = repeatInterleave(layerThree);
    torch::Tensor output = (X1_out + X2_out + X3_out + layerFour)/4.0;
    return output;
}

static auto registry = torch::RegisterOperators("adagradChallenge::reduction", &reduction);
