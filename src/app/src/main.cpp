#include <sage/core/log.hpp>

#include "application.hpp"

int main() {
    sage::core::log::init();
    sage::app::Application app;
    app.run();
    return 0;
}
