#pragma once
#include <string>
#include <cstdlib>
#include <vector>

std::vector<char> hexstring_to_binary(std::string str) {
    std::vector<char> vec;
    assert(str.size() % 2 == 0);
    for(auto index = 0; index < str.size(); index += 2) {
        auto byte_str = str.substr(index, 2);
        char* tmp;
        auto data = strtoul(byte_str.c_str(), &tmp, 16);
        assert(*tmp == 0);
        vec.push_back((char)data);
    }
    return vec;
}
