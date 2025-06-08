cmake_minimum_required(VERSION 3.8)
project(cortex)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(LLAMA_BUILD_COMMON On)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(dependencies/llama-cpp)

set(SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/CommandHandler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/LlamaInferenceEngine.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/PlanValidator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/TaskPlanner.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
)
target_link_libraries(${PROJECT_NAME} 
  yaml-cpp 
  common 
  llama 
  ggml
)

include_directories(${PROJECT_NAME} 
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}/dependencies/plog/plog-master/include
  ${CMAKE_CURRENT_LIST_DIR}/dependencies/llama-cpp/common
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY configuration/ DESTINATION share/${PROJECT_NAME}/configuration)

ament_package()
