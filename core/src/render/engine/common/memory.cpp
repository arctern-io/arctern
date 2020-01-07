#include "zcommon/config/megawise_config.h"

#include "render/engine/common/memory.h"
#include "render/engine/common/log.h"

namespace zilliz {
namespace render {
namespace engine {

MemManager::MemManager() {
    auto main_memory_size = common::megawise::DevCfg::render_engine::main_memory_size();
    if (main_memory_size <= 0) {
        RENDER_ENGINE_LOG_ERROR << "Invalid main memory size.";
    }
    main_memory_pool_ = std::make_shared<MainMemoryPool>(main_memory_size * 1024 * 1024 * 1024);
}


} // namespace engine
} // namespace render
} // namespace zilliz
