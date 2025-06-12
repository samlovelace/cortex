#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Logger.hpp"
#include "ConfigManager.hpp"
#include "CommandHandler.h"
#include "TaskPlanner.h"
#include "llama.h"

#include "ConcurrentQueue.hpp"


int main()
{
    createLogger(); 
    rclcpp::init(0, nullptr); 
    std::string packagePath = ament_index_cpp::get_package_share_directory("cortex");
    std::string configPath = packagePath + "/configuration/config.yaml";

    ConfigManager::get().load(configPath); 

    std::shared_ptr<ConcurrentQueue<std::string>> promptQueue = std::make_shared<ConcurrentQueue<std::string>>();  
    
    auto planner = std::make_shared<TaskPlanner>(promptQueue); 
    auto ch = CommandHandler(promptQueue); 
    
    ch.init(); 
    planner->run(); 

    rclcpp::shutdown(); 
}