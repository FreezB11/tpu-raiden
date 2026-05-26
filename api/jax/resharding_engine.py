# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-performance worker-to-worker resharding engine for TPU Raiden."""

import jax
import numpy as np
from api.jax import resharding_planner
from api.jax import weight_synchronizer

WeightSynchronizer = weight_synchronizer.WeightSynchronizer


def reshard_matrix(
    src_sharded_array: jax.Array,
    dst_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Performs optimal worker-to-worker resharding collective using global WeightSynchronizers.

  Directly copies memory blocks between worker CPU Host buffers over Loopback
  sockets, completely avoiding intermediate JAX array allocations and copies.

  Args:
    src_sharded_array: The JAX sharded array on the source mesh.
    dst_sharding: The target NamedSharding on the destination mesh.

  Returns:
    A new JAX sharded array on the target destination mesh.
  """
  global_shape = src_sharded_array.shape
  rows, cols = global_shape

  # Retrieve the complete logical global slice maps for both layouts
  src_map = src_sharded_array.sharding.devices_indices_map(global_shape)
  dst_map = dst_sharding.devices_indices_map(global_shape)

  # Canonical Sorting: Sort physical devices strictly by global coordinate
  # starts
  sorted_src_devices = sorted(
      src_sharded_array.sharding.addressable_devices,
      key=lambda d: (src_map[d][0].start or 0, src_map[d][1].start or 0),
  )
  sorted_dst_devices = sorted(
      dst_sharding.addressable_devices,
      key=lambda d: (dst_map[d][0].start or 0, dst_map[d][1].start or 0),
  )

  # Step 1: Initialize Global WeightSynchronizers E2E!
  # A. Allocate JAX single-device zeros for Destination
  # Note: we allocate zeros in canonical logical order so that they are cleanly
  # mapped.
  dst_device_arrays = []
  for dev in sorted_dst_devices:
    row_slice, col_slice = dst_map[dev]
    row_start = row_slice.start if row_slice.start is not None else 0
    row_end = row_slice.stop if row_slice.stop is not None else rows
    col_start = col_slice.start if col_slice.start is not None else 0
    col_end = col_slice.stop if col_slice.stop is not None else cols

    # Create clean device-placed zeros
    zero_arr = jax.device_put(
        np.zeros((row_end - row_start, col_end - col_start), dtype=np.float32),
        dev,
    )
    dst_device_arrays.append(zero_arr)

  # B. Build JAX global sharded arrays E2E
  dst_sharded_array_initial = jax.make_array_from_single_device_arrays(
      global_shape, dst_sharding, dst_device_arrays
  )

  # PJRT Physical Layout Ordering: Extract physical addressable devices lists
  # from addressable_shards to strictly align with PJRT's C++ buffers order.
  src_pjrt_devices = [
      shard.device for shard in src_sharded_array.addressable_shards
  ]
  dst_pjrt_devices = [
      shard.device for shard in dst_sharded_array_initial.addressable_shards
  ]

  # C. Instantiate arrays of WeightSynchronizers (one for each shard/device
  # E2E!)
  # This emulates independent multi-host setups where each process handles
  # its own local HBM shards on independent loopback ports.
  src_device_arrays_pjrt = [
      shard.data for shard in src_sharded_array.addressable_shards
  ]
  dst_device_arrays_pjrt = [
      shard.data for shard in dst_sharded_array_initial.addressable_shards
  ]

  ws_source = [
      WeightSynchronizer([arr], local_port=0, unsafe_skip_buffer_lock=True)
      for arr in src_device_arrays_pjrt
  ]
  ws_destination = [
      WeightSynchronizer([arr], local_port=0, unsafe_skip_buffer_lock=True)
      for arr in dst_device_arrays_pjrt
  ]

  # Step 2: Source D2H copies (Symmetrically offload weights to independent Host
  # CPU staging buffers)
  for ws in ws_source:
    ws.d2h()

  # Step 3: Sockets Symmetrical resharding pull collective (Direct workers
  # pulls!)
  plan = resharding_planner.make_resharding_plan(
      global_shape=global_shape,
      src_sharding=src_sharded_array.sharding,
      dst_sharding=dst_sharding,
  )

  for chunk in plan:
    logical_src_idx = chunk.src_device_id
    logical_dst_idx = chunk.dst_device_id

    # Map logical coordinate index to physical JAX devices
    src_dev = sorted_src_devices[logical_src_idx]
    dst_dev = sorted_dst_devices[logical_dst_idx]

    # Map physical JAX devices to arrays of synchronizers indices!
    src_shard_idx = src_pjrt_devices.index(src_dev)
    dst_shard_idx = dst_pjrt_devices.index(dst_dev)

    r_start, r_end, c_start, c_end = chunk.src_slice
    dr_start, _, dc_start, _ = chunk.dst_slice
    chunk_width = c_end - c_start

    # Determine physical dimension widths
    _, src_col_slice = src_map[src_dev]
    n_src = (src_col_slice.stop or cols) - (src_col_slice.start or 0)

    _, dst_col_slice = dst_map[dst_dev]
    n_dst = (dst_col_slice.stop or cols) - (dst_col_slice.start or 0)

    # Execute socket pulls row-by-row contiguously over network loopback!
    for row in range(r_end - r_start):
      src_row = r_start + row
      src_offset_bytes = (src_row * n_src + c_start) * 4

      # Contiguously read from socket into destination shard j's scratch pad
      # offset
      scratch_pad_offset = ws_destination[dst_shard_idx].slice_byte_size
      ws_destination[dst_shard_idx].pull_weights_chunk(
          source=f"127.0.0.1:{ws_source[src_shard_idx].local_port}",
          src_shard_idx=0,  # Each instance handles 1 shard, so FFI index is 0
          src_offset_bytes=src_offset_bytes,
          dst_shard_idx=0,  # Each instance handles 1 shard, so FFI index is 0
          dst_offset_bytes=scratch_pad_offset,
          size_bytes=chunk_width * 4,
      )

      # Trigger high-speed direct DMA transfer from Host staging scratchpad
      # offset directly into strided offset inside accelerator Device HBM!
      # This completely eliminates CopyLocalBuffer (Host-to-Host std::memmove)
      # and ws.h2d(), executing the entire resharding pull E2E with only one raw
      # sockets network transfer + one DMA thunk!
      dst_row = dr_start + row
      dst_offset_bytes = (dst_row * n_dst + dc_start) * 4
      ws_destination[dst_shard_idx].h2d_chunk(
          shard_idx=0,  # Each instance handles 1 shard, so FFI index is 0
          host_offset_bytes=scratch_pad_offset,
          device_offset_bytes=dst_offset_bytes,
          size_bytes=chunk_width * 4,
      )

  # Step 4: Reconstruct the updated global sharded array cleanly!
  # Note: Since h2d_chunk wrote directly to physical device memory, the original
  # dst_device_arrays are already populated with the resharded float weights.
  # Rebuilding a sharded array wrapper from them is E2E 100% complete and
  # optimal.
  dst_sharded_array = jax.make_array_from_single_device_arrays(
      global_shape, dst_sharding, dst_device_arrays
  )

  return dst_sharded_array
