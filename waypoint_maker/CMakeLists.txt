cmake_minimum_required(VERSION 3.0.2)
project(waypoint_maker)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  std_msgs
  geometry_msgs
  message_generation
  roscpp
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Lane.msg
   State.msg
   Waypoint.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs geometry_msgs 
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES waypoint_loader waypoint_saver
  CATKIN_DEPENDS nav_msgs std_msgs geometry_msgs message_runtime roscpp
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(waypoint_loader
   src/new_waypoint_loader.cpp
   src/cubic_spline_planner.cpp
)
add_library(waypoint_saver
   src/new_waypoint_saver.cpp
)

add_library(spline_csv_maker
    src/spline_csv_maker.cpp
    src/cubic_spline_planner.cpp
)
add_executable(waypoint_loader_node
   src/new_waypoint_loader.cpp
   src/cubic_spline_planner.cpp
)

add_executable(waypoint_saver_node
   src/new_waypoint_saver.cpp
)
add_executable(spline_csv_maker_node
   src/spline_csv_maker.cpp
   src/cubic_spline_planner.cpp
)
target_link_libraries(waypoint_loader_node
   ${catkin_LIBRARIES}
)

target_link_libraries(waypoint_saver_node
   ${catkin_LIBRARIES}
)
target_link_libraries(spline_csv_maker_node
   ${catkin_LIBRARIES}
)
