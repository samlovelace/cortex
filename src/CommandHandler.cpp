
#include "CommandHandler.h"
#include "plog/Log.h"

CommandHandler::CommandHandler(std::shared_ptr<ConcurrentQueue<std::string>> aPromptQueue) : mPromptQueue(aPromptQueue)
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
    // push the prompt to the task planning queue
    mPromptQueue->push(aMsg->data); 
    LOGD << "Pushed prompt to the queue for processing!"; 
}
