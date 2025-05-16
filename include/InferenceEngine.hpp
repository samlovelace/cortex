#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include <string> 


class InferenceEngine 
{
    
public:
    virtual bool init() = 0; 
    virtual std::string generate(const std::string& aPrompt) = 0;
    virtual ~InferenceEngine() = default;
};

#endif