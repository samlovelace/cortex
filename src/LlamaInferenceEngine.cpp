
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

    llama_batch batch = llama_batch_init(n_tokens, embd, n_seq_max);

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

    // apply the chat-template
    const char* tmpl = llama_model_chat_template(mModel, nullptr);
    int newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());

    if (newLen > static_cast<int>(mFormattedMessages.size())) 
    {
        // resize the output buffer `_formattedMessages`
        // and re-apply the chat template
        mFormattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(tmpl, mMessages.data(), mMessages.size(), true,
                                           mFormattedMessages.data(), mFormattedMessages.size());
    }

    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in "
                                 "LLMInference::start_completion() failed");
    }

    std::string prompt(mFormattedMessages.begin() + mPrevLen, mFormattedMessages.begin() + newLen);
    mPromptTokens = common_tokenize(llama_model_get_vocab(mModel), prompt, true, true);

    // create a llama_batch containing a single sequence
    // see llama_batch_init for more details
    mBatch.token    = mPromptTokens.data();
    mBatch.embd = 0; 
    mBatch.n_tokens = mPromptTokens.size();
}

std::string LlamaInferenceEngine::completionLoop() {
    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    int contextSize = llama_n_ctx(mContext);
    int nCtxUsed    = llama_get_kv_cache_used_cells(mContext);
    if (nCtxUsed + mBatch.n_tokens > contextSize) {
        LOGE << "context size exceeded"; 
        exit(0);
    }
    // run the model
    if (llama_decode(mContext, mBatch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    mCurrToken = llama_sampler_sample(mSampler, mContext, -1);

    if (llama_vocab_is_eog(llama_model_get_vocab(mModel), mCurrToken)) 
    {
        addPrompt(strdup(mResponse.data()), "assistant");
        mResponse.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(mContext, mCurrToken, true);
    mResponse += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached
    // in the KV cache
    mBatch.token    = &mCurrToken;
    mBatch.n_tokens = 1;

    return piece;
}

void LlamaInferenceEngine::addPrompt(const std::string& message, const std::string& role) 
{
    mMessages.push_back({ strdup(role.data()), strdup(message.data()) });
}
