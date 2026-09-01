#include <sage/core/log.hpp>

#include <filesystem>

#include "application.hpp"

int main(int argc, char** argv) {
    sage::core::log::init();

    // Empty by default: the viewer starts with nothing and waits for a file to
    // be picked. A path on the command line still loads, which is what keeps
    // `sage some.gltf` useful for going straight at one model.
    std::filesystem::path model_path;

    if (argc > 1) {
        model_path = argv[1];
    }

    sage::app::Application app(model_path);
    app.run();
    return 0;
}
