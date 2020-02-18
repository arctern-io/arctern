#pragma once
#include <memory>
namespace zilliz {
namespace gis {
namespace cuda {

// create a class deleter from normal function
template<class T, void (*fn)(T*)>
struct DeleterWrapper {
    template<class Ptr>
    void operator()(Ptr ptr) const {
        fn(ptr);
    }
};

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
