#ifndef LLAMAENGINE_H
#define LLAMAENGINE_H

#include <memory> 
 
#include "InferenceEngine.hpp"
#include "llama.h"
 
class LlamaInferenceEngine : public InferenceEngine
{ 
public:
    LlamaInferenceEngine();
    ~LlamaInferenceEngine();

    bool init() override; 
    std::string generate(const std::string& aPrompt) override; 

private:

    std::shared_ptr<llama_model> mModel; 

   
};
#endif //LLAMAENGINE_H