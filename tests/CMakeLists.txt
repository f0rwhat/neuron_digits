cmake_minimum_required(VERSION 3.6)

project(${PROJECT_NAME}-unit-tests LANGUAGES CXX)

set (SOURCES
    src/main.cpp
    src/MatrixTest.cpp
)

set (HEADERS
)

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    "src"
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Catch2::Catch2
)


add_test (NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR}/tests)
