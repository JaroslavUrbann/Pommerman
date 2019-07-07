cmake_minimum_required(VERSION 2.8)

project(munchen)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

#add_definitions(-DVERBOSE_STATE)
add_definitions(-DTIME_LIMIT_ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(munchen SHARED src/agents/basic_agents.cpp src/agents/berlin_agent.cpp src/agents/cologne_agent.cpp src/agents/dortmund_agent.cpp src/agents/eisenach_agent.cpp src/agents/simple_agent.cpp src/agents/uint128_t.cpp src/bboard/bboard.cpp src/bboard/environment.cpp src/bboard/step.cpp src/bboard/step_utility.cpp src/bboard/strategy.cpp)

message( STATUS ${CMAKE_SOURCE_DIR} )
include_directories(${CMAKE_SOURCE_DIR}/src/agents ${CMAKE_SOURCE_DIR}/src/bboard)



#install(TARGETS test DESTINATION lib)
#install(FILES test.h DESTINATION include)
