// fatbin_registry.h — Global registry for embedded fatbin fragments.
//
// Each generated registration .cpp auto-registers its fatbin at static
// init time. The host program looks up fragments by name string at runtime.
// No #include of individual bin2c headers is needed in the host code.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

struct fatbin_fragment
{
  const unsigned char* data;
  size_t size;
};

class fatbin_registry
{
public:
  static fatbin_registry& instance()
  {
    static fatbin_registry reg;
    return reg;
  }

  void register_fragment(const char* name, const unsigned char* data, size_t size)
  {
    entries_[name] = {data, size};
  }

  const fatbin_fragment* lookup(const char* name) const
  {
    auto it = entries_.find(name);
    return it != entries_.end() ? &it->second : nullptr;
  }

private:
  fatbin_registry() = default;
  std::map<std::string, fatbin_fragment> entries_;
};

// Helper for static registration in generated .cpp files.
struct fatbin_registrar
{
  fatbin_registrar(const char* name, const unsigned char* data, size_t size)
  {
    fatbin_registry::instance().register_fragment(name, data, size);
  }
};
