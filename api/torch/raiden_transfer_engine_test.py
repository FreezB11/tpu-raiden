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

"""E2E physical unit tests for RaidenTransferEngine on XLA TPUs."""

import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from api.torch.raiden_transfer_engine import RaidenTransferEngine


class RaidenTransferEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 1
    self.block_size = 1

  def test_e2e_producer_consumer_lifecycle(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8)  # 2 blocks capacity

    src_caches = []
    for _ in range(self.num_layers):
      src_caches.append(
          torch.full(
              shape, fill_value=1.0, dtype=torch.float32, device=self.device
          )
      )

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          torch.zeros(shape, dtype=torch.float32, device=self.device)
      )

    # Producer engine
    producer = RaidenTransferEngine(
        kv_caches=src_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    # Consumer engine
    consumer = RaidenTransferEngine(
        kv_caches=dst_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    self.assertIsNotNone(producer.local_control_port)
    self.assertIsNotNone(consumer.local_control_port)

    # 1. Producer registers blocks
    req_id = "test_req_1"
    uuid = 12345
    producer.register_send(req_id, uuid, [0, 1])

    # 2. Consumer submits load
    remote_endpoint = f"127.0.0.1:{producer.local_control_port}"
    op_id = consumer.submit_load(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=remote_endpoint,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
    )

    # Wait for consumer transfer to complete
    consumer.wait_transfer(op_id)

    # Check that consumer correctly loaded the values
    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 1.0, atol=1e-5)

    # Wait for producer to process the ack and finish
    time.sleep(0.5)

    # Poll finished
    done_sending, done_recving, failed_recving = producer.poll_finished()
    self.assertIn(req_id, done_sending)

    done_sending_c, done_recving_c, failed_recving_c = consumer.poll_finished()
    self.assertIn(req_id, done_recving_c)
    self.assertNotIn(req_id, failed_recving_c)

  def test_copy_planning(self):
    device = torch.device("cpu")
    shape = (4, 128, 8)
    kv_caches = [torch.zeros(shape, device=device)]

    engine = RaidenTransferEngine(
        kv_caches=kv_caches,
        tp_rank=0,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
    )

    plan = engine._send_copy_plan_for_testing([2, 1, 3])
    self.assertEqual(plan["num_blocks"], 3)
    self.assertEqual(plan["producer_remote_block_ids"], [1, 2, 3])

    load_plan = engine._load_copy_plan_for_testing([2, 1, 3], [5, 4, 6])
    self.assertEqual(load_plan["num_blocks"], 3)
    self.assertEqual(load_plan["requested_remote_block_ids"], [2, 1, 3])
    self.assertEqual(load_plan["requested_local_block_ids"], [5, 4, 6])


if __name__ == "__main__":
  absltest.main()
