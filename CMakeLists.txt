cmake_minimum_required(VERSION 3.2)
project(savant VERSION 0.2.0)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
include(CMakeFindDependencyMacro) # missing in xtensor-blasCMake.config
find_package(xtensor-blas REQUIRED)
find_package(Boost REQUIRED)
find_package(savvy REQUIRED)
find_package(Threads REQUIRED)
find_library(OMP_LIB omp)
#-DBOOST_UBLAS_NDEBUG

add_executable(savant src/main.cpp src/assoc.cpp src/single.cpp src/burden.cpp src/qtl.cpp src/plot.cpp src/pca.cpp src/fit.cpp src/grm.cpp src/inv_norm.cpp src/bed_file.cpp)
target_compile_definitions(savant PRIVATE -DSAVANT_VERSION="${PROJECT_VERSION}")
target_include_directories(savant PRIVATE include ${Boost_INCLUDE_DIRS})
target_link_libraries(savant savvy xtensor-blas ${OMP_LIB} ${CMAKE_THREAD_LIBS_INIT})


install(TARGETS savant RUNTIME DESTINATION bin)
