cmake_minimum_required(VERSION 3.20)
project(siliconfer_neon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARM NEON is always available on Apple Silicon (arm64). No extra flags needed.
# For SME2 (Phase 8 stretch): add -march=armv9.2-a+sme2 here.

find_package(pybind11 REQUIRED)

pybind11_add_module(siliconfer_neon
    bindings.cpp
    q4_gemv.cpp
    q4_gemm.cpp
)

# -O3: enable full optimisation including auto-vectorisation
# -ffast-math: allow reassociation for FP (safe for our GEMV)
target_compile_options(siliconfer_neon PRIVATE -O3 -ffast-math)

# Install the .so next to the Python __init__.py so Python can import it
install(TARGETS siliconfer_neon
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}")
