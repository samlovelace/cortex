
#include "CommandHandler.h"
#include "plog/Log.h"

CommandHandler::CommandHandler(std::shared_ptr<TaskPlanner> aPlanner) : mPlanner(aPlanner)
{

}

CommandHandler::~CommandHandler()
{

}

bool CommandHandler::init()
{
    RosTopicManager::getInstance()->createSubscriber<std_msgs::msg::String>("/cortex/prompt", 
                                                                            std::bind(&CommandHandler::promptCallback, 
                                                                            this, 
                                                                            std::placeholders::_1)); 

    RosTopicManager::getInstance()->createPublisher<std_msgs::msg::String>("/cortex/response");

    RosTopicManager::getInstance()->spinNode(); 
    return true; 
}

void CommandHandler::promptCallback(const std_msgs::msg::String::SharedPtr aMsg)
{
    // TODO: maybe i need something like this in the future ? 
    // if(mPlanner->isPlanning())
    // {
    //     return ; 
    // }

    mPlanner->plan(aMsg->data); 
}
