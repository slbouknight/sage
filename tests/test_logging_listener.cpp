#include <sage/core/log.hpp>

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace {

// main() initialises the logger in the application; the test binary uses
// Catch2's own main and so never did. Anything under test that logs therefore
// tripped the assert in core::log::logger() -- which is how this got noticed,
// when EditHistory's "Undo: <name>" line aborted twelve tests at once.
//
// A listener rather than a fixture: the requirement is per-binary, not
// per-test, and putting it in each test that happens to log would leave the
// next one to discover this the same way.
class LoggingListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo& /*info*/) override { sage::core::log::init(); }
};

}  // namespace

CATCH_REGISTER_LISTENER(LoggingListener)
