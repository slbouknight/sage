#include <sage/core/job.hpp>

namespace sage::core {

void JobSystem::schedule(const JobFn& fn) {
    if (fn) {
        fn();
    }
}

}  // namespace sage::core
