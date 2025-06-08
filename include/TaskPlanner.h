#ifndef TASKPLANNER_H
#define TASKPLANNER_H

#include <memory> 
#include <yaml-cpp/yaml.h>

#include "InferenceEngine.hpp" 
#include "PlanValidator.h"

class TaskPlanner 
{ 
public:
    TaskPlanner();
    ~TaskPlanner();

    bool plan(const std::string& aCommand); 

private:

    std::shared_ptr<InferenceEngine> mInferenceEngine; 
    std::shared_ptr<PlanValidator> mValidator; 

    int mMaxPlanningAttempts; 



   
};
#endif //TASKPLANNER_H