#include <opt_infer_engine.h>

namespace OPT {
    OptInferEngine::OptInferEngine(const std::string& model_path,
                                   const std::string &vocab_file,
                                   const std::string &merges_file)
    : preprocessor_(vocab_file, merges_file)
    {
        loadModel(model_path);
    }

    void OptInferEngine::loadModel(const std::string& opt_model_path) {
        auto start = std::chrono::high_resolution_clock::now();
        try {
            opt_model_ = torch::jit::load(opt_model_path);
            opt_model_.eval();
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            std::cerr << e.what();
            exit(1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "OPT-125M Model is loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    std::string OptInferEngine::getReturnString(const std::string& text) {
        auto text_tensor = preprocessor_.preprocessText(text);

        std::vector<torch::jit::IValue> inputs = {text_tensor};
        auto output = opt_model_.forward(inputs).toTensor();
        auto output_a = output.accessor<float, 2>();
        std::string return_string = "";
        for (int i = 0; i < output_a.size(1); i++) {
            return_string += std::to_string(output_a[0][i]) + " ";
        }
        return return_string;
    }
}