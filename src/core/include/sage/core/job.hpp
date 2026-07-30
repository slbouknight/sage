#pragma once

#include <functional>

namespace sage::core {

using JobFn = std::function<void()>;

// M0 placeholder: runs jobs inline on the calling thread. The real scheduler
// (worker pool, work-stealing, dependency graph) lands when a milestone
// actually needs concurrent execution.
class JobSystem {
public:
    void schedule(const JobFn& fn);
};

}  // namespace sage::core
