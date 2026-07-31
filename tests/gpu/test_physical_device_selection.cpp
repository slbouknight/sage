#include <sage/gpu/physical_device.hpp>

#include <catch2/catch_test_macros.hpp>

namespace {

VkQueueFamilyProperties family(VkQueueFlags flags, std::uint32_t count = 1) {
    VkQueueFamilyProperties properties{};
    properties.queueFlags = flags;
    properties.queueCount = count;
    return properties;
}

VkPhysicalDeviceProperties device_of_type(VkPhysicalDeviceType type) {
    VkPhysicalDeviceProperties properties{};
    properties.deviceType = type;
    return properties;
}

VkPhysicalDeviceMemoryProperties memory_with_device_local(VkDeviceSize bytes) {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = bytes;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    return memory;
}

constexpr VkDeviceSize k_gib = 1024ULL * 1024ULL * 1024ULL;

}  // namespace

TEST_CASE("Prefers a single family supporting both graphics and present", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)};
    const std::vector<bool> present{true};

    const auto indices = sage::gpu::select_queue_families(families, present);

    REQUIRE(indices.complete());
    CHECK(*indices.graphics == 0);
    CHECK(*indices.present == 0);
}

TEST_CASE("Falls back to separate graphics and present families", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{family(VK_QUEUE_GRAPHICS_BIT),
                                                        family(VK_QUEUE_TRANSFER_BIT)};
    const std::vector<bool> present{false, true};

    const auto indices = sage::gpu::select_queue_families(families, present);

    REQUIRE(indices.complete());
    CHECK(*indices.graphics == 0);
    CHECK(*indices.present == 1);
}

TEST_CASE("Picks a transfer-only family when one exists", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT), family(VK_QUEUE_TRANSFER_BIT)};
    const std::vector<bool> present{true, false};

    const auto indices = sage::gpu::select_queue_families(families, present);

    REQUIRE(indices.transfer.has_value());
    CHECK(*indices.transfer == 1);
}

TEST_CASE("Transfer falls back to the graphics family", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{
        family(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)};
    const std::vector<bool> present{true};

    const auto indices = sage::gpu::select_queue_families(families, present);

    REQUIRE(indices.transfer.has_value());
    CHECK(*indices.transfer == *indices.graphics);
}

TEST_CASE("Families with no queues are skipped", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{family(VK_QUEUE_GRAPHICS_BIT, 0),
                                                        family(VK_QUEUE_GRAPHICS_BIT, 1)};
    const std::vector<bool> present{true, true};

    const auto indices = sage::gpu::select_queue_families(families, present);

    REQUIRE(indices.complete());
    CHECK(*indices.graphics == 1);
}

TEST_CASE("Incomplete when no family can present", "[physical_device]") {
    const std::vector<VkQueueFamilyProperties> families{family(VK_QUEUE_GRAPHICS_BIT)};
    const std::vector<bool> present{false};

    const auto indices = sage::gpu::select_queue_families(families, present);

    CHECK_FALSE(indices.complete());
    CHECK(indices.graphics.has_value());
    CHECK_FALSE(indices.present.has_value());
}

TEST_CASE("Device-local memory sums only device-local heaps", "[physical_device]") {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 3;
    memory.memoryHeaps[0].size = 8 * k_gib;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].size = 16 * k_gib;
    memory.memoryHeaps[1].flags = 0;
    memory.memoryHeaps[2].size = 2 * k_gib;
    memory.memoryHeaps[2].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

    CHECK(sage::gpu::device_local_memory_bytes(memory) == 10 * k_gib);
}

TEST_CASE("Discrete GPUs outrank integrated regardless of VRAM", "[physical_device]") {
    const auto discrete = sage::gpu::score_device(
        device_of_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU), memory_with_device_local(2 * k_gib));
    const auto integrated =
        sage::gpu::score_device(device_of_type(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
                                memory_with_device_local(64 * k_gib));

    CHECK(discrete > integrated);
}

TEST_CASE("VRAM breaks ties between same-type devices", "[physical_device]") {
    const auto small = sage::gpu::score_device(device_of_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
                                               memory_with_device_local(8 * k_gib));
    const auto large = sage::gpu::score_device(device_of_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
                                               memory_with_device_local(24 * k_gib));

    CHECK(large > small);
}

TEST_CASE("CPU and other device types score lowest", "[physical_device]") {
    const auto cpu = sage::gpu::score_device(device_of_type(VK_PHYSICAL_DEVICE_TYPE_CPU),
                                             memory_with_device_local(0));
    const auto integrated = sage::gpu::score_device(
        device_of_type(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU), memory_with_device_local(0));

    CHECK(cpu < integrated);
}
