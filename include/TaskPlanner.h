#ifndef TASKPLANNER_H
#define TASKPLANNER_H

#include <memory> 
#include <yaml-cpp/yaml.h>

#include "InferenceEngine.hpp" 
#include "PlanValidator.h"
#include "ConcurrentQueue.hpp"

class TaskPlanner 
{ 
public:
    TaskPlanner(std::shared_ptr<ConcurrentQueue<std::string>> aPromptQueue);
    ~TaskPlanner();

    void run(); 

private:

    bool plan(const std::string& aCommand); 

    std::shared_ptr<InferenceEngine> mInferenceEngine; 
    std::shared_ptr<PlanValidator> mValidator; 
    std::shared_ptr<ConcurrentQueue<std::string>> mPromptQueue; 
    
    int mMaxPlanningAttempts; 



   
};
#endif //TASKPLANNER_H