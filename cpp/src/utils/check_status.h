
#define CHECK_GDAL(action)                                                          \
{                                                                                   \
    auto check = action;                                                            \
    if(!!check){                                                                    \
        std::string err_msg = "gdal error code = " + std::to_string((int)check);    \
        throw std::runtime_error(err_msg);                                          \
    }                                                                               \
}

#define CHECK_ARROW(action)                                                 \
{                                                                           \
    arrow::Status status = action;                                          \
    if(!status.ok()){                                                       \
        std::string err_msg = "arrow error: " + status.ToString();          \
        throw std::runtime_error(err_msg);                                  \
    }                                                                       \
}


