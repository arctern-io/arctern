#pragma once
#include <string>
#include <vector>

// csv format, for better readability in linestring.csv file
constexpr auto relation_csv = R"(left_linestring,right_linestring,matrix
0_0_0_3,0_0_0_1_1_1_0_2_0_3,FFFFFFFF*
0_0_0_3,0_-100_0_-99_3_3_0_-1_0_1_0_2_0_4,FFFFFFFF*
0_0_0_1,0_1_0_2,FFFFFFFF*
0_0_0_1,0_0_2_1_-2_0,FFFFFFFF*
0_0_0_1,0_0_2_3,FFFFFFFF*
0_0_0_1,-2_0_2_0,FFFFFFFF*
0_0_0_2,0_1_2_3,FFFFFFFF*
0_0_0_1,-2_0_2_1,FFFFFFFF*
0_0_0_1,0_1_2_2,FFFFFFFF*
0_0_0_1,0_3_2_2,FFFFFFFF*
0_0_0_1,0_0_0_1,FFFFFFFF*
0_0_0_3,0_0_0_1_0_2_0_3,FFFFFFFF*
0_0_0_3,0_0_0_2_0_1_0_3,FFFFFFFF*
0_0_0_3,0_0_0_1_1_1_0_2_0_3_4_4_0_2_0_1,FFFFFFFF*)";

using std::string;
using std::vector;

inline vector<string> SplitString(const string& raw, char delimitor) {
  int index = 0;
  vector<string> result;
  while (index < raw.size()) {
    auto pos = raw.find_first_of(delimitor, index);
    if (pos == raw.npos) {
      pos = raw.size();
    }
    result.push_back(raw.substr(index, raw.size() - index));
    index = pos + 1;
  }
}

inline vector<double> StringToDoubleArray(const string& underscore_splitted_str) {
  auto tmp_vec = SplitString(underscore_splitted_str, '_');
  vector<double> result;
  for (auto str : tmp_vec) {
    result.push_back(strtod(str.data(), nullptr));
  }
  return result;
}

vector<vector<string>> GetTableFromCsv(const string& csv_raw) {
  auto lines = SplitString(csv_raw, '\n');
  vector<vector<string>> table;
  for (auto& line : lines) {
    table.emplace_back(SplitString(line, ','));
  }
  return table;
}
