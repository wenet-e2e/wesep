if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if(CXX11_ABI)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip")
    set(URL_HASH "SHA256=d52f63577a07adb0bfd6d77c90f7da21896e94f71eb7dcd55ed7835ccb3b2b59")
  else()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip")
    set(URL_HASH "SHA256=bee1b7be308792aa60fc95a4f5274d9658cb7248002d0e333d49eb81ec88430c")
  endif()
else()
  message(FATAL_ERROR "Unsported System '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
endif()

FetchContent_Declare(libtorch
  URL         ${LIBTORCH_URL}
  URL_HASH    ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
