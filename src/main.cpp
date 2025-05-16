#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Logger.hpp"
#include "ConfigManager.hpp"
#include "CommandHandler.h"
#include "llama.h"
#include "EngineFactory.hpp"


int main()
{
    rclcpp::init(0, nullptr); 
    createLogger(); 
    std::string package_path = ament_index_cpp::get_package_share_directory("cortex");
    std::string config_path = package_path + "/configuration/config.yaml";

    ConfigManager::get().load(config_path); 

    std::string engineType; 
    if(!ConfigManager::get().getConfig("engine", engineType))
    {
        LOGE << "Invalid Inference Engine Configuration"; 
        return 0; 
    }

    auto engine = EngineFactory::create(engineType);
    auto ch = CommandHandler(engine); 

    engine->init(); 
    ch.init(); 

    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); 
    }

    rclcpp::shutdown(); 
}