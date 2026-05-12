load("@jax//jaxlib:pywrap.bzl", "nanobind_pywrap_extension", "pywrap_library", "pywrap_binaries")

nanobind_pywrap_extension(
    name = "raw_transfer",
    srcs = [
        "raw_transfer.cc",
        "raw_transfer_impl.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    deps = [
        "@jax//jaxlib:py_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@nanobind",
        "@xla//xla:future",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla:shape_util",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:statusor",
    ],
)

pywrap_library(
    name = "raw_transfer_pywrap",
    deps = [":raw_transfer"],
)

pywrap_binaries(
    name = "raw_transfer_binaries",
    dep = ":raw_transfer_pywrap",
)
