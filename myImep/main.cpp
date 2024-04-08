#include <iostream>
#include <fstream>
#include <chrono>
#include "opt_infer_engine.h"

int main() {
    OPT::OptInferEngine opt_infer_engine("../traced_opt-125m.pt", "../vocab.json", "../merges.txt");
    std::string text = "Hello, my dog's 2 feet tall.";

    std::ofstream outfile("execution_times.txt");

    for (int i = 0; i < 50; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string return_string = opt_infer_engine.getReturnString(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << return_string << std::endl;
        outfile << "Execution " << i + 1 << ": " << duration.count() << " microseconds" << std::endl;
    }

    outfile.close();

    return 0;
}
