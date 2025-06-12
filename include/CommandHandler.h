#ifndef COMMANDHANDLER_H
#define COMMANDHANDLER_H
 
#include <memory>

#include "RosTopicManager.hpp"
#include "InferenceEngine.hpp"
#include "std_msgs/msg/string.hpp"
#include "ConcurrentQueue.hpp"
#include <string> 
 
class CommandHandler 
{ 
public:
    CommandHandler(std::shared_ptr<ConcurrentQueue<std::string>> aPromptQueue);
    ~CommandHandler();

    bool init(); 
    void promptCallback(const std_msgs::msg::String::SharedPtr aMsg); 

private:

    std::shared_ptr<ConcurrentQueue<std::string>> mPromptQueue; 
   
};
#endif //COMMANDHANDLER_H