#pragma once
#include <memory>
namespace zilliz {
namespace gis {
namespace cuda {

// create a class deleter from normal function
template<auto& Fn>
struct DeleterWrapper {
    template<class Ptr>
    void operator()(Ptr ptr) const {
        Fn(ptr);
    }
};

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
