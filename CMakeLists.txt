cmake_minimum_required(VERSION 4.2)

project(stock_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------
# Kokkos
# -----------------------

set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(Kokkos_ARCH_ADA89 ON CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)

FetchContent_MakeAvailable(Kokkos)

# -----------------------
# Output
# -----------------------

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# -----------------------
# LinearAlgebra
# -----------------------

add_library(LinearAlgebra INTERFACE)

target_include_directories(LinearAlgebra
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(LinearAlgebra
        INTERFACE
        Kokkos::kokkos
)

# -----------------------
# Main
# -----------------------

add_executable(stock_analysis
        main.cpp
)

target_link_libraries(stock_analysis
        PRIVATE
        LinearAlgebra
)