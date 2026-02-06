cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Signalsmith DSP (header-only)
add_subdirectory(thirdparty/signalsmith)

nanobind_add_module(_core
    src/cydsp/_core.cpp
    src/cydsp/_core_signalsmith.cpp
)
target_link_libraries(_core PRIVATE signalsmith-dsp)

if(MSVC)
    target_compile_options(_core PRIVATE /W4)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

install(TARGETS _core DESTINATION cydsp)
