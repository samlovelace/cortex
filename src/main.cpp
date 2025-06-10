#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Logger.hpp"
#include "ConfigManager.hpp"
#include "CommandHandler.h"
#include "TaskPlanner.h"
#include "llama.h"

#include "EngineFactory.hpp"

int main()
{
    createLogger(); 
    rclcpp::init(0, nullptr); 
    std::string packagePath = ament_index_cpp::get_package_share_directory("cortex");
    std::string configPath = packagePath + "/configuration/config.yaml";

    ConfigManager::get().load(configPath); 

    auto planner = std::make_shared<TaskPlanner>(); 
    auto ch = CommandHandler(planner); 
    ch.init(); 

    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); 
    }

    rclcpp::shutdown(); 
}