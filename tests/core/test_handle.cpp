#include <sage/core/handle.hpp>

#include <catch2/catch_test_macros.hpp>

#include <unordered_set>

namespace {
struct TestTag {};
}  // namespace

TEST_CASE("Handle default-constructs invalid", "[handle]") {
    const sage::core::Handle<TestTag> h;
    CHECK_FALSE(h.valid());
    CHECK(h.index() == 0);
    CHECK(h.generation() == 0);
}

TEST_CASE("Handle stores index and generation", "[handle]") {
    const sage::core::Handle<TestTag> h(42, 7);
    CHECK(h.index() == 42);
    CHECK(h.generation() == 7);
    CHECK(h.valid());
}

TEST_CASE("Handle equality compares index and generation", "[handle]") {
    const sage::core::Handle<TestTag> a(1, 2);
    const sage::core::Handle<TestTag> b(1, 2);
    const sage::core::Handle<TestTag> c(1, 3);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("Handle is hashable and usable as a set key", "[handle]") {
    std::unordered_set<sage::core::Handle<TestTag>> handles;
    handles.insert(sage::core::Handle<TestTag>(1, 1));
    handles.insert(sage::core::Handle<TestTag>(1, 1));
    handles.insert(sage::core::Handle<TestTag>(2, 1));
    CHECK(handles.size() == 2);
}

TEST_CASE("Guid default-constructs invalid", "[guid]") {
    const sage::core::Guid g;
    CHECK_FALSE(g.valid());
}

TEST_CASE("Guid::generate produces valid, distinct ids", "[guid]") {
    const sage::core::Guid a = sage::core::Guid::generate();
    const sage::core::Guid b = sage::core::Guid::generate();
    CHECK(a.valid());
    CHECK(b.valid());
    CHECK(a != b);
}

TEST_CASE("Guid is hashable and usable as a set key", "[guid]") {
    std::unordered_set<sage::core::Guid> guids;
    guids.insert(sage::core::Guid::generate());
    guids.insert(sage::core::Guid::generate());
    CHECK(guids.size() == 2);
}
