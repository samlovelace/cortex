
#include "PlanValidator.h"

PlanValidator::PlanValidator() : mValidPlanGenerated(false)
{

}

PlanValidator::~PlanValidator()
{

}

void PlanValidator::validate(const std::string& aPlan)
{
    // TODO: implement actual plan validation 
    mValidPlanGenerated = true;

    // check that format is correct  
    // check that only primitive actions are used 

}