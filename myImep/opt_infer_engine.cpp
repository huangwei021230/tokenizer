#include <opt_infer_engine.h>

namespace OPT {
    OptInferEngine::OptInferEngine(const std::string& model_path,
                                   const std::string &vocab_file,
                                   const std::string &merges_file)
    : processor_(vocab_file, merges_file)
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
        auto text_tensor = processor_.preprocessText(text);
        std::vector<torch::jit::IValue> inputs = {text_tensor};

        auto output = opt_model_.forward(inputs).toTuple();
        auto all_encoder_layers = output->elements()[0].toTensor();

        // 计算 softmax

        torch::Tensor probs_high = torch::softmax(all_encoder_layers, -1);
        torch::Tensor input_tensor = torch::abs(probs_high);

        torch::Tensor input_tensor_reshaped = input_tensor.view({-1, 50272});

        int num_samples = 10;
        torch::Tensor samples = torch::multinomial(input_tensor_reshaped, num_samples, true);

        torch::Tensor random_indices = torch::randint(0, 10, {4});
        torch::Tensor random_elements = samples.index({torch::arange(4), random_indices});



        auto return_vec = processor_.getReturnString(random_elements);
        std::string return_string;
        for (auto &str : return_vec) {
            return_string += str + " ";
        }
        return return_string;
    }
}