#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "Logger.hpp"
#include "ConfigManager.hpp"

int main()
{
    createLogger(); 
    std::string package_path = ament_index_cpp::get_package_share_directory("cortex");
    std::string config_path = package_path + "/configuration/config.yaml";

    ConfigManager::get().load(config_path); 
    
    std::string modelPath; 
    if(!ConfigManager::get().getConfig<std::string>("model", modelPath)) 
    {
        LOGE << "Invalid Configuration";
        return 0; 
    }

    LOGD << "Model Path: " << modelPath;
}