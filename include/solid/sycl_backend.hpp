#pragma once

#include "ggml-backend.h"

// Returns null for a device without shared DRAM. An allocation failure on a
// supported device is reported as an exception.
extern "C" ggml_backend_buffer_t solid_sycl_alloc_shared_kv(ggml_backend_t backend, size_t bytes);
