//#pragma once

#include <opt_processor.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace OPT {
class OptInferEngine {
public:
    OptInferEngine(const std::string& model_path, const std::string &vocab_file, const std::string &merges_file);
    std::string getReturnString(const std::string& text);
private:
    void loadModel(const std::string& opt_model_path);
    torch::jit::script::Module opt_model_;
    OptProcessor processor_;
};
}