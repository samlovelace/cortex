
#include "LlamaInferenceEngine.h"
#include "plog/Log.h"
#include <cstring>
#include <iostream> 

// A no-op logger
void quietLogger(enum ggml_log_level level, const char* text, void* user_data) 
{
    // do nothing
}

LlamaInferenceEngine::LlamaInferenceEngine()
{

}

LlamaInferenceEngine::~LlamaInferenceEngine()
{
    // Clean up batch allocations
    for (int i = 0; i < mBatch.n_tokens; ++i) {
        delete[] mBatch.seq_id[i];
    }
    delete[] mBatch.pos;
    delete[] mBatch.seq_id;
    delete[] mBatch.logits;

    llama_batch_free(mBatch); // Just in case
    llama_backend_free();
}

bool LlamaInferenceEngine::init(const std::string& aModelPath, const std::string& aPromptHeader)
{
    mPromptHeader = aPromptHeader; 
    llama_log_set(quietLogger, nullptr); 

    // TODO: get from config maybe? 
    float temperature = 1.0f;
    float minP = 0.05f;

    // create an instance of llama_model
    llama_model_params modelParams = llama_model_default_params();
    mModel = llama_model_load_from_file(aModelPath.data(), modelParams);

    if (!mModel) {
        throw std::runtime_error("load_model() failed");
    }

    LOGD << "Successfully loaded model from " << aModelPath; 

    // create an instance of llama_context
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx                = 0;    // take context size from the model GGUF file
    ctxParams.no_perf              = true; // disable performance metrics
    mContext                           = llama_init_from_model(mModel, ctxParams);

    if (!mContext) {
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    // initialize sampler
    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    samplerParams.no_perf                    = true; // disable performance metrics
    mSampler                                 = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(mSampler, llama_sampler_init_min_p(minP, 1));
    llama_sampler_chain_add(mSampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    mFormattedMessages = std::vector<char>(llama_n_ctx(mContext));
    mMessages.clear();

    // You want a token buffer of size max_tokens, no embedding buffer:
    int32_t n_tokens    = 512;
    int32_t embd        = 0;      // ← must be zero for token inference
    int32_t n_seq_max   = 1;      // e.g. single sequence context

    mBatch = llama_batch_init(n_tokens, embd, n_seq_max);

    LOGD << "LlaMa Inference Engine initialized successfully!"; 
    
    return true; 
}

std::string LlamaInferenceEngine::generate(const std::string& aPrompt)
{
    std::lock_guard<std::mutex> lock(mMutex); 
    setGeneratingResponse(true); 
    LOGD << "Receieved a prompt to input to model!"; 
    std::string fullPrompt = mPromptHeader + aPrompt; 
    LOGD << "Task: " << aPrompt;
    
    startCompletion(fullPrompt);

    LOGD << "Generating Task Plan..."; 
    std::string predictedToken;
    std::stringstream fullResponse; 
    while ((predictedToken = completionLoop()) != "[EOG]") 
    {
        fullResponse << predictedToken;
        //fflush(stdout);
    }

    setGeneratingResponse(false);  
    return fullResponse.str(); 
}

void LlamaInferenceEngine::startCompletion(const std::string& query) {
    addPrompt(query, "user");

    const char* tmpl = llama_model_chat_template(mModel, nullptr);
    int newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());

    if (newLen > static_cast<int>(mFormattedMessages.size())) {
        mFormattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());
    }

    std::string prompt(mFormattedMessages.begin() + mPrevLen, mFormattedMessages.begin() + newLen);
    mPromptTokens = common_tokenize(llama_model_get_vocab(mModel), prompt, true, true);
    int N = mPromptTokens.size();

    // Resize internal vectors
    mPos.resize(N);
    mSeqId.resize(N);
    mLogits.resize(N);

    for (int i = 0; i < N; ++i) {
        mPos[i] = 0;
        mSeqId[i] = new int[1]{0};  // sequence ID 0
        mLogits[i] = false;
    }

    mBatch.token = mPromptTokens.data();
    mBatch.pos = mPos.data();
    mBatch.seq_id = mSeqId.data();
    mBatch.logits = mLogits.data();
    mBatch.n_tokens = N;
    mBatch.embd = 0;
}



std::string LlamaInferenceEngine::completionLoop() 
{
    LOGD << "completion loop"; 
    if (llama_get_kv_cache_used_cells(mContext) + mBatch.n_tokens > llama_n_ctx(mContext)) {
        throw std::runtime_error("Context exceeded");
    }

    if (llama_decode(mContext, mBatch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    mCurrToken = llama_sampler_sample(mSampler, mContext, -1);
    if (llama_vocab_is_eog(llama_model_get_vocab(mModel), mCurrToken)) {
        addPrompt(strdup(mResponse.data()), "assistant");
        mResponse.clear();
        return "[EOG]";
    }

    std::string piece = common_token_to_piece(mContext, mCurrToken, true);
    mResponse += piece;

    // Reuse the vectors safely for next token
    mPromptTokens = { mCurrToken };
    mPos = { llama_get_kv_cache_used_cells(mContext) };  // or 0
    mLogits = { false };

    int* id = new int[1]{0};
    if (!mSeqId.empty()) delete[] mSeqId[0];
    mSeqId = { id };

    mBatch.token = mPromptTokens.data();
    mBatch.pos = mPos.data();
    mBatch.seq_id = mSeqId.data();
    mBatch.logits = mLogits.data();
    mBatch.n_tokens = 1;
    mBatch.embd = 0;

    return piece;
}


void LlamaInferenceEngine::addPrompt(const std::string& message, const std::string& role) 
{
    mMessages.push_back({ strdup(role.data()), strdup(message.data()) });
}
