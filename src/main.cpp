#include <cstdio> 
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "ConfigManager.hpp"

int main()
{
    std::string package_path = ament_index_cpp::get_package_share_directory("cortex");
    std::string config_path = package_path + "/configuration/config.yaml";

    ConfigManager::get().load(config_path); 
    
    std::string modelPath; 
    if(!ConfigManager::get().getConfig<std::string>("model", modelPath)) 
    {
        std::cerr << "Invalid Configuration" << std::endl; 
        return 0; 
    }

    std::cout << "Model Path: " << modelPath << std::endl;
}