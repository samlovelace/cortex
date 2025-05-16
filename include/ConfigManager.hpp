#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H
 
#include <yaml-cpp/yaml.h>
#include <optional>
#include <iostream> 

class ConfigManager 
{ 
public:
    static ConfigManager& get()
    {
        static ConfigManager mInstance; 
        return mInstance; 
    }

    bool load(const std::string& filename) 
    {
        try {
            mRoot = YAML::LoadFile(filename);
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }

    template<typename T>
    bool getConfig(const std::string& key, T& value) const 
    {
        auto node = findKeyRecursive(mRoot, key);
        if (node && node->IsDefined()) {
            try {
                value = node->as<T>();
                return true;
            } catch (const YAML::TypedBadConversion<T>& e) {
                std::cerr << "Type conversion error for key '" << key << "': " << e.what() << std::endl;
                return false;
            }
        }
        return false;
    }

private:

    ConfigManager() {}
    ~ConfigManager() {}

    std::optional<YAML::Node> findKeyRecursive(const YAML::Node& node, const std::string& key) const 
    {
        if (!node.IsMap()) return std::nullopt;

        for (const auto& kv : node) {
            const std::string currentKey = kv.first.as<std::string>();
            const YAML::Node& value = kv.second;

            if (currentKey == key) return value;

            // Recursive descent
            if (value.IsMap()) {
                auto result = findKeyRecursive(value, key);
                if (result) return result;
            }

            if (value.IsSequence()) {
                for (const auto& item : value) {
                    auto result = findKeyRecursive(item, key);
                    if (result) return result;
                }
            }
        }

        return std::nullopt;
    }

    YAML::Node mRoot; 
   
};
#endif //CONFIGMANAGER_H