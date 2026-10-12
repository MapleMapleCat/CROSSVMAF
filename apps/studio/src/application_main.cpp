#include "app/studio_controller.hpp"
#include "platform/window_integration.hpp"

#include "app.h"

int main() {
    auto application = StudioApp::create();
    const VM& view_model = application->global<VM>();
    crossvmaf::studio::StudioController controller(view_model);
    controller.initialize();

    application->show();
    crossvmaf::studio::installWindowIntegration(application);
    application->run();
    return 0;
}
