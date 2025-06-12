// include/LlamaPyInferenceEngine.h

#pragma once
#include "InferenceEngine.hpp"
#include <pybind11/embed.h>

namespace py = pybind11;

class LlamaPyInferenceEngine : public InferenceEngine {
public:
    LlamaPyInferenceEngine();
    ~LlamaPyInferenceEngine() override = default;

    bool init(const std::string& modelPath, const std::string& promptHeader) override;
    std::string generate(const std::string& prompt) override;

private:
    py::scoped_interpreter mGuard;  // boots & tears down Python
    py::object             mClient; // the Python-side Llama client
    std::string            mHeader;
};
