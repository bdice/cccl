//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <testing.cuh>

#ifndef __CUDA_ARCH__

struct resource_a
{
  void* allocate(cuda::stream_ref, size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, size_t, size_t) noexcept {}
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  bool operator==(const resource_a&) const noexcept
  {
    return true;
  }
  bool operator!=(const resource_a&) const noexcept
  {
    return false;
  }
  friend void get_property(const resource_a&, cuda::mr::host_accessible) noexcept {}
};

struct resource_b
{
  void* allocate(cuda::stream_ref, size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, size_t, size_t) noexcept {}
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  bool operator==(const resource_b&) const noexcept
  {
    return true;
  }
  bool operator!=(const resource_b&) const noexcept
  {
    return false;
  }
  friend void get_property(const resource_b&, cuda::mr::host_accessible) noexcept {}
};

struct sync_resource_a
{
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  bool operator==(const sync_resource_a&) const noexcept
  {
    return true;
  }
  bool operator!=(const sync_resource_a&) const noexcept
  {
    return false;
  }
  friend void get_property(const sync_resource_a&, cuda::mr::host_accessible) noexcept {}
};

struct sync_resource_base
{
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  bool operator==(const sync_resource_base&) const noexcept
  {
    return true;
  }
  bool operator!=(const sync_resource_base&) const noexcept
  {
    return false;
  }
  friend void get_property(const sync_resource_base&, cuda::mr::host_accessible) noexcept {}
};

struct sync_resource_derived : sync_resource_base
{};

struct sync_resource_b
{
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  bool operator==(const sync_resource_b&) const noexcept
  {
    return true;
  }
  bool operator!=(const sync_resource_b&) const noexcept
  {
    return false;
  }
  friend void get_property(const sync_resource_b&, cuda::mr::host_accessible) noexcept {}
};

TEST_CASE("any_cast on any_resource", "[any_resource]")
{
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{resource_a{}};

  CHECK(cuda::mr::any_cast<resource_a>(&mr) != nullptr);
  CHECK(cuda::mr::any_cast<resource_b>(&mr) == nullptr);

  const cuda::mr::any_resource<cuda::mr::host_accessible> const_mr{resource_a{}};
  CHECK(cuda::mr::any_cast<resource_a>(&const_mr) != nullptr);
  CHECK(cuda::mr::any_cast<resource_b>(&const_mr) == nullptr);

  // nullptr input returns nullptr
  cuda::mr::any_resource<cuda::mr::host_accessible>* null_mr = nullptr;
  CHECK(cuda::mr::any_cast<resource_a>(null_mr) == nullptr);

  // empty (default-constructed) any_resource returns nullptr
  cuda::mr::any_resource<cuda::mr::host_accessible> empty_mr;
  CHECK(cuda::mr::any_cast<resource_a>(&empty_mr) == nullptr);

  // empty (moved-from) any_resource returns nullptr
  cuda::mr::any_resource<cuda::mr::host_accessible> moved_from{resource_a{}};
  auto moved_to = std::move(moved_from);
  CHECK(cuda::mr::any_cast<resource_a>(&moved_from) == nullptr);
  CHECK(cuda::mr::any_cast<resource_a>(&moved_to) != nullptr);
}

TEST_CASE("any_cast on any_synchronous_resource", "[any_resource]")
{
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> mr{sync_resource_a{}};

  CHECK(cuda::mr::any_cast<sync_resource_a>(&mr) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_b>(&mr) == nullptr);

  const cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> const_mr{sync_resource_a{}};
  CHECK(cuda::mr::any_cast<sync_resource_a>(&const_mr) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_b>(&const_mr) == nullptr);

  // nullptr input returns nullptr
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>* null_mr = nullptr;
  CHECK(cuda::mr::any_cast<sync_resource_a>(null_mr) == nullptr);
}

TEST_CASE("any_cast on resource_ref", "[any_resource]")
{
  resource_a ra{};
  cuda::mr::resource_ref<cuda::mr::host_accessible> ref{ra};

  CHECK(cuda::mr::any_cast<resource_a>(&ref) != nullptr);
  CHECK(cuda::mr::any_cast<resource_a>(&ref) == &ra);
  CHECK(cuda::mr::any_cast<resource_b>(&ref) == nullptr);

  const cuda::mr::resource_ref<cuda::mr::host_accessible> const_ref{ra};
  CHECK(cuda::mr::any_cast<resource_a>(&const_ref) != nullptr);
  CHECK(cuda::mr::any_cast<resource_b>(&const_ref) == nullptr);

  // nullptr input returns nullptr
  cuda::mr::resource_ref<cuda::mr::host_accessible>* null_ref = nullptr;
  CHECK(cuda::mr::any_cast<resource_a>(null_ref) == nullptr);
}

TEST_CASE("any_cast on synchronous_resource_ref", "[any_resource]")
{
  sync_resource_a ra{};
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{ra};

  CHECK(cuda::mr::any_cast<sync_resource_a>(&ref) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_a>(&ref) == &ra);
  CHECK(cuda::mr::any_cast<sync_resource_b>(&ref) == nullptr);

  const cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> const_ref{ra};
  CHECK(cuda::mr::any_cast<sync_resource_a>(&const_ref) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_b>(&const_ref) == nullptr);

  // nullptr input returns nullptr
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>* null_ref = nullptr;
  CHECK(cuda::mr::any_cast<sync_resource_a>(null_ref) == nullptr);
}

TEST_CASE("any_cast requires exact type match (no derived-to-base)", "[any_resource]")
{
  // any_cast performs an exact type match, like std::any_cast.
  // Casting to a base class when a derived class is stored returns nullptr.
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> mr{sync_resource_derived{}};
  CHECK(cuda::mr::any_cast<sync_resource_derived>(&mr) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_base>(&mr) == nullptr);

  sync_resource_derived derived{};
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{derived};
  CHECK(cuda::mr::any_cast<sync_resource_derived>(&ref) != nullptr);
  CHECK(cuda::mr::any_cast<sync_resource_base>(&ref) == nullptr);
}

#endif // __CUDA_ARCH__
