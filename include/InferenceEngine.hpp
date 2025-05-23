#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include <string> 
#include <mutex> 

class InferenceEngine 
{
    
public:
    virtual bool init(const std::string& aModelPath) = 0; 
    virtual std::string generate(const std::string& aPrompt) = 0;
    virtual ~InferenceEngine() = default;

    void setGeneratingResponse(bool aFlag) {std::lock_guard<std::mutex> lock(mGeneratingMutex); mGeneratingResponse = aFlag;}
    bool isGenerating() {std::lock_guard<std::mutex> lock(mGeneratingMutex); return mGeneratingResponse;}

private: 
    std::mutex mGeneratingMutex; 
    bool mGeneratingResponse; 

};

#endif