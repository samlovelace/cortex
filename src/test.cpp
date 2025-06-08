#include <iostream>
#include <llama.h>

int main() {
    // Initialize the llama context
    llama_context_params ctx_params;
    llama_init_params(&ctx_params);

    // Load the model from a file
    const char* model_path = "path_to_your_model_file";
    llama_model* model = llama_load_model_from_file(model_path, ctx_params);

    if (model == nullptr) {
        std::cerr << "Failed to load model from file: " << model_path << std::endl;
        return 1;
    }

    // Create a new context with the loaded model
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        std::cerr << "Failed to create context with model." << std::endl;
        return 1;
    }

    // Tokenize input text
    const char* input_text = "Hello, llama!";
    std::vector<llama_token> tokens = llama_tokenize(ctx, input_text);

    // Generate output tokens
    std::vector<llama_token> output_tokens = llama_generate(ctx, tokens);

    // Convert output tokens back to text
    std::string output_text = llama_detokenize(ctx, output_tokens);

    std::cout << "Generated text: " << output_text << std::endl;

    // Clean up
    llama_free_context(ctx);
    llama_free_model(model);

    return 0;
}
