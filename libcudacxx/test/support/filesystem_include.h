#ifndef TEST_SUPPORT_FILESYSTEM_INCLUDE_H
#define TEST_SUPPORT_FILESYSTEM_INCLUDE_H

#include <filesystem>

#include "test_macros.h"

#if defined(_LIBCUDACXX_VERSION) && TEST_STD_VER < 2017
namespace fs = std::__fs::filesystem;
#else
namespace fs = std::filesystem;
#endif

#endif
