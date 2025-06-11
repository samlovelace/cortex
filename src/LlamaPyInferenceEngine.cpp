// src/LlamaPyInferenceEngine.cpp

#include "LlamaPyInferenceEngine.h"
#include <stdexcept>

#include "plog/Log.h"

namespace py = pybind11;

LlamaPyInferenceEngine::LlamaPyInferenceEngine()
    : mGuard() // initialize the embedded Python interpreter
{}

bool LlamaPyInferenceEngine::init(const std::string& modelPath,
                                  const std::string& promptHeader)
{
    // 1) Import the Python module
    try {
        py::module_::import("llama_cpp");
    } catch (py::error_already_set &e) {
        throw std::runtime_error(
            std::string("Failed to import llama_cpp Python module:\n") + e.what());
    }

    // 2) Construct a Python-side Llama client
    auto Llama = py::module_::import("llama_cpp").attr("Llama");
    mClient = Llama(
        py::arg("model_path") = modelPath,
        py::arg("n_ctx")      = 2048,
        py::arg("n_threads")  = 4,
        py::arg("verbose")    = false
    );
    LOGD << "Successfully loaded model from " << modelPath; 

    mHeader = promptHeader;
    return true;
}

std::string LlamaPyInferenceEngine::generate(const std::string& prompt) {
    // 1) Build the full prompt
    std::string full = mHeader + prompt;

    LOGD << "Generating Plan for Task: " << prompt; 

    // 2) Prepare stop sequences
    py::list stop_list;
    stop_list.append("\n\n");
    stop_list.append("\n\nTask:");

    // 3) Acquire the GIL before any Python calls
    LOGD << "Getting GIL"; 
    py::gil_scoped_acquire gil;
    LOGD << "DOne getting gil"; 

    // 4) Call the Llama instance as a function
    //    Equivalent to: response = llm(full, temperature=0.95, max_tokens=100, stop=[...])
    LOGD << "Calling pybind to generate response"; 
    py::object result_obj = mClient(
        full,
        py::arg("temperature") = 0.95,
        py::arg("max_tokens")  = 100,
        py::arg("stop")        = stop_list
    );

    LOGD << "Done generating response"; 

    // 5) Extract the dict and pull out the first choice’s text
    py::dict result = result_obj.cast<py::dict>();
    auto choices = result["choices"].cast<py::list>();
    if (choices.empty()) {
        return "";
    }
    LOGD << "Choices not empty, getting response"; 
    py::dict first = choices[0].cast<py::dict>();
    return first["text"].cast<std::string>();
}

