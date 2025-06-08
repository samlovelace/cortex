#ifndef COMMANDHANDLER_H
#define COMMANDHANDLER_H
 
#include <memory>

#include "RosTopicManager.hpp"
#include "InferenceEngine.hpp"
#include "std_msgs/msg/string.hpp"
#include "TaskPlanner.h"
 
class CommandHandler 
{ 
public:
    CommandHandler(std::shared_ptr<TaskPlanner> aPlanner);
    ~CommandHandler();

    bool init(); 
    void promptCallback(const std_msgs::msg::String::SharedPtr aMsg); 

private:

    std::shared_ptr<TaskPlanner> mPlanner; 
   
};
#endif //COMMANDHANDLER_H