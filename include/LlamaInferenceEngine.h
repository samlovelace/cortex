#ifndef LLAMAENGINE_H
#define LLAMAENGINE_H

#include <memory> 
 
#include "InferenceEngine.hpp"
#include "llama.h"
#include "common.h"
 
class LlamaInferenceEngine : public InferenceEngine
{ 
public:
    LlamaInferenceEngine();
    ~LlamaInferenceEngine();

    bool init(const std::string& aModelPath, const std::string& aPromptHeader) override; 
    std::string generate(const std::string& aPrompt) override; 

    void addPrompt(const std::string& message, const std::string& role);

private:

    llama_model* mModel;
    llama_context* mContext; 
    llama_sampler* mSampler;
    llama_batch mBatch;
    llama_token mCurrToken;
    std::vector<llama_token> mStablePromptTokens;

    std::vector<int> mPos;
    std::vector<int*> mSeqId;
    std::vector<int8_t> mLogits;

    std::mutex mMutex; 

    int mMaxTokens;
    std::vector<llama_seq_id> mSeqIdBacking;
    std::vector<llama_seq_id*> mSeqIdPtrs;
    std::vector<int8_t> mLogitsArray;
    int* mNSlots = nullptr;


    // stores the complete response for the given query
    std::string mResponse = "";
    
    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> mMessages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> mFormattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> mPromptTokens;
    int mPrevLen = 0;

    void startCompletion(const std::string& query);
    std::string completionLoop();

   
};
#endif //LLAMAENGINE_H