#ifndef PLANVALIDATOR_H
#define PLANVALIDATOR_H
 
#include <string>  

class PlanValidator 
{ 
public:
    PlanValidator();
    ~PlanValidator();

    void validate(const std::string& aPlan); 
    void reset(); 
    bool validPlanGenerated() {return mValidPlanGenerated;}

private:
    bool mValidPlanGenerated; 
   
};
#endif //PLANVALIDATOR_H