
#include "TaskPlanner.h"
#include "ConfigManager.hpp"
#include "plog/Log.h" 
#include "EngineFactory.hpp"

TaskPlanner::TaskPlanner() : mInferenceEngine(nullptr), mValidator(std::make_shared<PlanValidator>()), mMaxPlanningAttempts(3)
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

bool TaskPlanner::plan(const std::string& aCommand)
{
    std::string taskPlan = ""; 
    int numPlanningAttempts = 0; 

    while(!mValidator->validPlanGenerated() && numPlanningAttempts++ < mMaxPlanningAttempts)
    {
        taskPlan = mInferenceEngine->generate(aCommand); 
        mValidator->validate(taskPlan); 
    }

    return mValidator->validPlanGenerated(); 
}