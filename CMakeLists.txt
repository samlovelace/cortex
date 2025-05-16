cmake_minimum_required(VERSION 3.8)
project(cortex)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

set(SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

include_directories(${PROJECT_NAME} 
  ${CMAKE_CURRENT_LIST_DIR}/include
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY configuration/ DESTINATION share/${PROJECT_NAME}/configuration)

ament_package()
