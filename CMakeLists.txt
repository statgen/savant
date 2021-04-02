cmake_minimum_required(VERSION 3.2)
project(savant VERSION 0.1.0)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
include(CMakeFindDependencyMacro) # missing in xtensor-blasCMake.config
find_package(xtensor-blas REQUIRED)
find_package(Boost REQUIRED)
find_package(savvy REQUIRED)

add_executable(savant src/main.cpp)
target_include_directories(savant PRIVATE include ${Boost_INCLUDE_DIRS})
target_link_libraries(savant savvy xtensor-blas)


install(TARGETS savant RUNTIME DESTINATION bin)
