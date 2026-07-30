#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

namespace sage::core {

// A generation-counted index into some owner-defined pool. Tag is a distinct
// empty type per pool (e.g. `struct MeshTag;`) so handles into different
// pools can't be swapped by accident at compile time.
template <typename Tag>
class Handle {
public:
    using Index = std::uint32_t;
    using Generation = std::uint32_t;

    constexpr Handle() noexcept = default;
    constexpr Handle(Index index, Generation generation) noexcept
        : index_(index), generation_(generation) {}

    [[nodiscard]] constexpr Index index() const noexcept { return index_; }
    [[nodiscard]] constexpr Generation generation() const noexcept { return generation_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return generation_ != 0; }

    friend constexpr bool operator==(const Handle&, const Handle&) noexcept = default;

private:
    Index index_ = 0;
    Generation generation_ = 0;
};

// A 128-bit random identifier (UUIDv4 layout) for objects that need identity
// independent of any pool or index, e.g. cross-session asset references.
class Guid {
public:
    constexpr Guid() noexcept = default;

    static Guid generate();

    [[nodiscard]] constexpr std::uint64_t high() const noexcept { return high_; }
    [[nodiscard]] constexpr std::uint64_t low() const noexcept { return low_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return high_ != 0 || low_ != 0; }

    friend constexpr bool operator==(const Guid&, const Guid&) noexcept = default;

private:
    constexpr Guid(std::uint64_t high, std::uint64_t low) noexcept : high_(high), low_(low) {}

    std::uint64_t high_ = 0;
    std::uint64_t low_ = 0;
};

inline Guid Guid::generate() {
    static thread_local std::mt19937_64 engine{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;

    std::uint64_t high = dist(engine);
    std::uint64_t low = dist(engine);

    // Stamp UUIDv4 version/variant bits so the pattern is recognizable in
    // debuggers and external tools that expect standard UUIDs.
    high = (high & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
    low = (low & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

    return Guid(high, low);
}

}  // namespace sage::core

namespace std {

template <typename Tag>
struct hash<sage::core::Handle<Tag>> {
    size_t operator()(const sage::core::Handle<Tag>& h) const noexcept {
        const auto packed = (static_cast<std::uint64_t>(h.index()) << 32) |
                            static_cast<std::uint64_t>(h.generation());
        return hash<std::uint64_t>{}(packed);
    }
};

template <>
struct hash<sage::core::Guid> {
    size_t operator()(const sage::core::Guid& g) const noexcept {
        const size_t h1 = hash<std::uint64_t>{}(g.high());
        const size_t h2 = hash<std::uint64_t>{}(g.low());
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

}  // namespace std
