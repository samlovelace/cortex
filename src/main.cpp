#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Logger.hpp"
#include "ConfigManager.hpp"
#include "CommandHandler.h"
#include "llama.h"
#include "EngineFactory.hpp"


int main()
{
    createLogger(); 
    std::string packagePath = ament_index_cpp::get_package_share_directory("cortex");
    std::string configPath = packagePath + "/configuration/config.yaml";

    ConfigManager::get().load(configPath); 

    std::string engineType; 
    if(!ConfigManager::get().getConfig("engine", engineType))
    {
        LOGE << "Invalid Inference Engine Configuration"; 
        return 0; 
    }
    
    std::string modelPath; 
    if(!ConfigManager::get().getConfig("path", modelPath))
    {
        LOGE << "Missing or wrongly configured path to folder containing models"; 
        return 0; 
    }

    std::string modelName; 
    if(!ConfigManager::get().getConfig("model", modelName))
    {
        LOGE << "Invalid model name configuration"; 
        return 0; 
    }

    rclcpp::init(0, nullptr); 
    auto engine = EngineFactory::create(engineType);
    auto ch = CommandHandler(engine); 

    engine->init(modelPath + "/" +  modelName); 
    ch.init(); 

    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); 
    }

    rclcpp::shutdown(); 
}