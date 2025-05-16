
#include "LlamaInferenceEngine.h"
#include "plog/Log.h"

LlamaInferenceEngine::LlamaInferenceEngine()
{

}

LlamaInferenceEngine::~LlamaInferenceEngine()
{

}

bool LlamaInferenceEngine::init()
{
    return true; 
}

std::string LlamaInferenceEngine::generate(const std::string& aPrompt)
{
    LOGD << "Receieved a prompt to input to model!"; 
    LOGD << "Prompt: " << aPrompt; 

    return ""; 
}