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

load("//third_party/bazel_rules/rules_cc/cc/common:cc_info.bzl", "CcInfo", "merge_cc_infos")
load("//third_party/bazel_rules/rules_cc/cc/private:cc_info.bzl", "create_linking_context")

def _header_only_cc_info_impl(ctx):
    merged = merge_cc_infos(
        direct_cc_infos = [dep[CcInfo] for dep in ctx.attr.deps],
    )
    return [
        CcInfo(
            compilation_context = merged.compilation_context,
            linking_context = create_linking_context(
                linker_inputs = depset(),
            ),
        ),
    ]

header_only_cc_info = rule(
    attrs = {
        "deps": attr.label_list(),
    },
    implementation = _header_only_cc_info_impl,
)
