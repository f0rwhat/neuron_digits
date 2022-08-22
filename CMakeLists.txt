cmake_minimum_required(VERSION 3.0.0)
project(neuron_digits VERSION 0.1.0)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(HEADERS
    headers/activators/IActivatorFunc.hpp
    headers/activators/ModReluFunc.hpp
    
    headers/IRenderable.hpp
    headers/RenderWindow.hpp
    headers/BitMap.hpp

    headers/Matrix.hpp
    headers/NeuroNet.hpp
)

set(SOURCES
    src/main.cpp
)

add_executable(${PROJECT_NAME} 
    ${HEADERS}
    ${SOURCES}
)

include_directories(${PROJECT_NAME}
    headers/
    headers/activators

    ${OPENGL_INCLUDE_DIRS}  
    ${GLUT_INCLUDE_DIRS} 
)

target_link_libraries(${PROJECT_NAME} 
    pthread
    
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
