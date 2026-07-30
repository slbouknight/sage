#include <sage/core/log.hpp>

int main() {
    sage::core::log::init();
    SAGE_LOG_INFO("sage starting (M0 skeleton, no graphics yet)");
    return 0;
}
