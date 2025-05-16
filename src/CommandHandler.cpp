
#include "CommandHandler.h"
#include "plog/Log.h"

CommandHandler::CommandHandler(std::shared_ptr<InferenceEngine> anEngine) : mEngine(anEngine), mPromptRcvd(false)
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


    RosTopicManager::getInstance()->spinNode(); 
    return true; 
}

void CommandHandler::promptCallback(const std_msgs::msg::String::SharedPtr aMsg)
{
    if(!mPromptRcvd)
    {
        mPromptRcvd = true; 
        mEngine->generate(aMsg->data); 
    }
}
