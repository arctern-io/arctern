#include <iostream>
#include "vega.h"


namespace zilliz {
namespace render {


bool
Vega::JsonLabelCheck(rapidjson::Value &value, const std::string &label) {

    if (!value.HasMember(label.c_str())) {
        std::cout << "Cannot find label [" << label << "] !";
        return false;
    }
    return true;
}


bool
Vega::JsonSizeCheck(rapidjson::Value &value, const std::string &label, size_t size) {

    if (value.Size() != size) {
        std::cout << "Member [" << label << "].size should be " << size << ", but get " << value.Size() << std::endl;
        return false;
    }
    return true;
}


bool
Vega::JsonTypeCheck(rapidjson::Value &value, rapidjson::Type type) {

    switch (type) {
        case rapidjson::Type::kNumberType:
            if (!value.IsNumber()) {
                std::cout << "not number type" << std::endl;
                return false;
            }
            return true;
        case rapidjson::Type::kArrayType:
            if (!value.IsArray()) {
                std::cout << "not array type" << std::endl;
                return false;
            }
            return true;
        case rapidjson::Type::kStringType :
            if (!value.IsString()) {
                std::cout << "not string type" << std::endl;
                return false;
            }
            return true;
        default: {
            std::cout << "unknown type" << std::endl;
            return false;
        }
    }
}


bool
Vega::JsonNullCheck(rapidjson::Value &value) {

    if (value.IsNull()) {
        std::cout << "null!!!" << std::endl;
        return false;
    }
    return true;
}



} //namespace render
} //namespace zilliz