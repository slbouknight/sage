#include <sage/core/log.hpp>

#include <filesystem>

#include "application.hpp"

int main(int argc, char** argv) {
    sage::core::log::init();

    std::filesystem::path model_path =
        std::filesystem::path(SAGE_ASSET_DIR) / "lantern" / "Lantern.gltf";

    if (argc > 1) {
        model_path = argv[1];
    }

    sage::app::Application app(model_path);
    app.run();
    return 0;
}
