#pragma once

template<typename WkbVisitorImpl>
struct WkbVisitor: public WkbVisitorImpl {
  using WkbVisitorImpl::WkbVisitorImpl;

};
