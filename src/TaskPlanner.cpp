
#include "TaskPlanner.h"
#include "ConfigManager.hpp"
#include "plog/Log.h" 
#include "EngineFactory.hpp"

TaskPlanner::TaskPlanner(std::shared_ptr<ConcurrentQueue<std::string>> aPromptQueue) : 
    mPromptQueue(aPromptQueue), mInferenceEngine(nullptr), mValidator(std::make_shared<PlanValidator>()), mMaxPlanningAttempts(3)
{
    std::string engineType; 
    if(!ConfigManager::get().getConfig("engine", engineType))
    {
        throw std::invalid_argument("Invalid Inference Engine Configuration");  
    }
    
    std::string modelPath; 
    if(!ConfigManager::get().getConfig("path", modelPath))
    {
        throw std::invalid_argument("Missing or wrongly configured path to folder containing models"); 
    }

    std::string modelName; 
    if(!ConfigManager::get().getConfig("model", modelName))
    {
        throw std::invalid_argument("Invalid model name configuration"); 
    }

    std::string promptHeader; 
    if(!ConfigManager::get().getConfig("promptHeader", promptHeader))
    {
        throw std::invalid_argument("Invalid prompt header"); 
    }

    if(!ConfigManager::get().getConfig<int>("maxPlanningAttempts", mMaxPlanningAttempts))
    {
        throw std::invalid_argument("Invalid number of max planning attempts"); 
    }

    mInferenceEngine = EngineFactory::create(engineType);
    std::string fullModelPath = modelPath + "/" + modelName; 
    mInferenceEngine->init(fullModelPath, promptHeader); 
}

TaskPlanner::~TaskPlanner()
{

}

void TaskPlanner::run()
{
    while(true)
    {
        std::string prompt; 
        if(mPromptQueue->pop(prompt))
        {
            LOGD << "Prompt popped from queue!"; 
            plan(prompt); 
        }
    }
}

bool TaskPlanner::plan(const std::string& aCommand)
{
    std::string taskPlan = ""; 
    int numPlanningAttempts = 0; 

    while(!mValidator->validPlanGenerated() && numPlanningAttempts++ < mMaxPlanningAttempts)
    {
        auto start = std::chrono::steady_clock::now(); 
        taskPlan = mInferenceEngine->generate(aCommand);
        auto end = std::chrono::steady_clock::now(); 
        LOGD << "Task Plan: " << taskPlan; 
        LOGD << "Generated task plan in " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " secs"; 
        LOGD << "Validating task plan...";  
        mValidator->validate(taskPlan); 
    }

    LOGD << "Valid task plan generated!"; 
    mValidator->reset(); 
    return true; 
}