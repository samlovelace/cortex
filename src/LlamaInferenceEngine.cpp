
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
    delete[] mBatch.token;
    delete[] mBatch.pos;
    delete[] mBatch.seq_id;
    delete[] mBatch.logits;
    delete[] mBatch.n_seq_id;



    llama_backend_free();
}

bool LlamaInferenceEngine::init(const std::string& aModelPath, const std::string& aPromptHeader)
{
    mPromptHeader = aPromptHeader; 
    llama_log_set(quietLogger, nullptr); 

    float temperature = 1.0f;
    float minP = 0.05f;

    llama_model_params modelParams = llama_model_default_params();
    mModel = llama_model_load_from_file(aModelPath.c_str(), modelParams);
    if (!mModel) {
        throw std::runtime_error("load_model() failed");
    }

    LOGD << "Successfully loaded model from " << aModelPath;

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx   = 2048;
    ctxParams.no_perf = true;
    mContext = llama_init_from_model(mModel, ctxParams);
    if (!mContext) {
        throw std::runtime_error("llama_init_from_model() returned null");
    }

    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    samplerParams.no_perf = true;
    mSampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(mSampler, llama_sampler_init_min_p(minP, 1));
    llama_sampler_chain_add(mSampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    mFormattedMessages = std::vector<char>(llama_n_ctx(mContext));
    mMessages.clear();

    mMaxTokens = llama_n_ctx(mContext);
    mBatch = llama_batch_init(mMaxTokens, 0, 1);

    // Allocate token and pos arrays
    mBatch.token = new llama_token[mMaxTokens];
    mBatch.pos   = new int[mMaxTokens];

    // Allocate logits (int8_t)
    mLogitsArray.resize(mMaxTokens, 0);
    mBatch.logits = mLogitsArray.data();

    // Allocate seq_id (llama_seq_id**)
    mSeqIdBacking.resize(mMaxTokens, 0);
    mSeqIdPtrs.resize(mMaxTokens);
    for (int i = 0; i < mMaxTokens; ++i) {
        mSeqIdPtrs[i] = &mSeqIdBacking[i];
    }

    mBatch.n_seq_id = new int[mMaxTokens];
    mNSlots = mBatch.n_seq_id;

    mBatch.embd = nullptr;
    mBatch.n_tokens = 0;

    LOGD << "LlaMa Inference Engine initialized successfully!";
    return true;
}


std::string LlamaInferenceEngine::generate(const std::string& aPrompt)
{
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

void LlamaInferenceEngine::startCompletion(const std::string& query) 
{
    addPrompt(query, "user");

    const char* tmpl = llama_model_chat_template(mModel, nullptr);
    int newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());

    if (newLen > static_cast<int>(mFormattedMessages.size())) {
        mFormattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());
    }

    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() failed");
    }

    std::string prompt(mFormattedMessages.begin() + mPrevLen, mFormattedMessages.begin() + newLen);
    mPromptTokens = common_tokenize(llama_model_get_vocab(mModel), prompt, true, true);

    if ((int)mPromptTokens.size() > mMaxTokens) {
        throw std::runtime_error("Too many tokens for batch size!");
    }

        for (int i = 0; i < mPromptTokens.size(); ++i) {
        mBatch.token[i]     = mPromptTokens[i];
        mBatch.pos[i]       = i;
        mSeqIdBacking[i]    = 0;
        mBatch.n_seq_id[i]  = 1;               // <--- this is the missing piece!
        mLogitsArray[i]     = (i == mPromptTokens.size() - 1) ? 1 : 0;
    }
    mBatch.n_tokens = mPromptTokens.size();
}

std::string LlamaInferenceEngine::completionLoop() {
    int contextSize = llama_n_ctx(mContext);
    int nCtxUsed    = llama_get_kv_cache_used_cells(mContext);
    if (nCtxUsed + mBatch.n_tokens > contextSize) {
        LOGE << "Context size exceeded!";
        exit(1);
    }

    if (llama_decode(mContext, mBatch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    mCurrToken = llama_sampler_sample(mSampler, mContext, -1);

    if (llama_vocab_is_eog(llama_model_get_vocab(mModel), mCurrToken)) 
    {
        addPrompt(strdup(mResponse.data()), "assistant");
        mResponse.clear();
        return "[EOG]";
    }

    std::string piece = common_token_to_piece(mContext, mCurrToken, true);
    mResponse += piece;

    nCtxUsed = llama_get_kv_cache_used_cells(mContext);
    mBatch.token[0]    = mCurrToken;
    mBatch.pos[0]      = nCtxUsed;
    mSeqIdBacking[0]   = 0;
    mBatch.n_seq_id[0] = 1;
    mLogitsArray[0]    = 1;
    mBatch.n_tokens    = 1; 

    return piece;
}


void LlamaInferenceEngine::addPrompt(const std::string& message, const std::string& role) 
{
    mMessages.push_back({ strdup(role.data()), strdup(message.data()) });
}
