#pragma once
#include <cuda.h>
#include <vector_types.h>

#include <map>
#include <string>
#include <vector>

namespace datasource {
// csv format, for better readability in linestring.csv file
constexpr auto relation_csv = R"(left_linestring,right_linestring,matrix
0_0_0_3,0_0_0_1_1_1_0_2_0_3,1F1F0F1F*
0_0_0_1,0_1_0_2,FF1F0010*
0_0_0_1,0_0_2_1_-2_0,0F1F0010*
0_0_0_1,0_0_2_3,FF1F0010*
0_0_0_1,-2_0_2_0,FF10F010*
0_0_0_2,0_1_2_3,F01FF010*
0_0_0_1,-2_0_2_1,0F1FF010*
0_0_0_1,0_1_2_2,FF1F0010*
0_0_0_1,0_3_2_2,FF1FF010*
0_0_0_1,0_0_0_1,1FFF0FFF*
0_0_0_3,0_0_0_1_0_2_0_3,1FFF0FFF*
0_0_0_3,0_0_0_2_0_1_0_3,1FFF0FFF*
1_0_2_0,0_0_1_0_2_0_3_0,1FF0FF10*
0_0_0_3,0_0_0_1_1_1_0_2_0_3_4_4_0_2_0_1,10F00F1F*
0_0_0_3,0_0_0_1_1_1_0_2_0_3_4_4_0_2_0_1.5,10100F1F*
0_0_0_3,0_-100_0_-99_3_3_0_-1_0_1_0_2_0_4,1FF0FF10*
0_1_1_0_0_0_0_-1_-1_0_0_0_0_1,0_1_0_-1_-1_0_1_0_0_1,1FFF0FFF*
)";
}  // namespace datasource

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
    result.push_back(raw.substr(index, pos - index));
    index = pos + 1;
  }
  return result;
}

// should be underscore splitted
inline vector<double> ToDoubleArray(const string& str_raw) {
  auto tmp_vec = SplitString(str_raw, '_');
  vector<double> result;
  for (auto str : tmp_vec) {
    result.push_back(strtod(str.data(), nullptr));
  }
  return result;
}

inline vector<double2> ToDouble2Array(const vector<double>& vec) {
  assert(vec.size() % 2 == 0);
  vector<double2> result;
  for (int i = 0; i < vec.size() / 2; ++i) {
    result.push_back(double2{vec[i * 2], vec[i * 2 + 1]});
  }
  return result;
}

// should be underscore splitted
inline vector<double2> ToDouble2Array(const string& str_raw) {
  auto vec = ToDoubleArray(str_raw);
  return ToDouble2Array(vec);
}

inline vector<vector<string>> GetTableFromCsv(const string& csv_raw) {
  auto lines = SplitString(csv_raw, '\n');
  vector<vector<string>> table;
  for (auto& line : lines) {
    auto vec = SplitString(line, ',');
    if(vec.size() == 0) {
      continue;
    }
    table.emplace_back(vec);
  }
  return table;
}

// csv[keys..]
inline std::vector<vector<string>> ProjectedTableFromCsv(const string& csv_raw,
                                                         const vector<string>& keys) {
  auto raw_table = GetTableFromCsv(csv_raw);
  auto raw_keys = raw_table[0];
  raw_table.erase(raw_table.begin());

  std::map<string, int> mapping;
  std::vector<int> indexes;
  for (int i = 0; i < raw_keys.size(); ++i) {
    mapping[raw_keys[i]] = i;
  }
  for (auto& key : keys) {
    indexes.push_back(mapping[key]);
  }
  vector<vector<string>> result;
  for (const auto& raw_line : raw_table) {
    vector<string> line;
    for (auto index : indexes) {
      line.push_back(raw_line[index]);
    }
    result.push_back(std::move(line));
  }
  return result;
}

