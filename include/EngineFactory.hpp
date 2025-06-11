#ifndef ENGINEFACTORY
#define ENGINEFACTORY
 
#include "InferenceEngine.hpp"
#include "LlamaInferenceEngine.h"
#include "LlamaPyInferenceEngine.h"
#include "plog/Log.h"
 
class EngineFactory 
{ 
public:
    EngineFactory();
    ~EngineFactory();

    static std::shared_ptr<InferenceEngine> create(const std::string& anEngineType)
    {
        if("llama" == anEngineType || "llama-cpp"  == anEngineType)
        {
            LOGD << "Using LLaMa.cpp Inference Engine"; 
            return std::make_shared<LlamaInferenceEngine>(); 
        }
        else if("llama-py" == anEngineType)
        {
            return std::make_shared<LlamaPyInferenceEngine>(); 
        }
        else{
            LOGE << "Unsupported Inference Engine type: " << anEngineType;
            return nullptr; 
        }
    }

private:
   
};
#endif //ENGINEFACTORY