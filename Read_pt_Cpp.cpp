#include <iostream>
#include <vector>

#include <torch/script.h>

int main() {

  // Load pretrained flow model
  auto track3D = torch::jit::load("/home/patrick/MUonE/ML_for_MUonE/Read_pt_Cpp/best_wide.pt");
  track3D.eval();

  int num_iter = 100;

  for (int i = 0; i < num_iter; ++i) {
    auto x0 = track3D({torch::zeros({1, 36})}).toTensor().exp();
    auto y0 = track3D({torch::ones({1, 36})}).toTensor().exp();
        
    // Plot samples
    std::cout << "Iteration " << i+1 << "\n";
    std::cout << x0 << "\n";
    std::cout << y0 << "\n";
  }

  return 0;
}
