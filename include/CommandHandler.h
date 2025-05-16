#ifndef COMMANDHANDLER_H
#define COMMANDHANDLER_H
 
#include <memory>

#include "RosTopicManager.hpp"
#include "InferenceEngine.hpp"
#include "std_msgs/msg/string.hpp"

 
class CommandHandler 
{ 
public:
    CommandHandler(std::shared_ptr<InferenceEngine> anEngine);
    ~CommandHandler();

    bool init(); 

    void promptCallback(const std_msgs::msg::String::SharedPtr aMsg); 

private:

    std::shared_ptr<InferenceEngine> mEngine; 
    bool mPromptRcvd; 
   
};
#endif //COMMANDHANDLER_H