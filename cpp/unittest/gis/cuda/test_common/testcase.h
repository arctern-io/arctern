// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once
#include <cuda.h>
#include <vector_types.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace datasource {
// csv format, for better readability in linestring.csv file
constexpr auto relation_csv = R"(left_linestring,right_linestring,matrix
,,FFFFFFFF*
,0 0 0 1,FFFFFF10*
0 0 0 3,0 0 0 1 1 1 0 2 0 3,1F1F0F1F*
0 0 0 1,0 1 0 2,FF1F0010*
0 0 0 1,0 0 2 1 -2 0,0F1F0010*
0 0 0 1,0 0 2 3,FF1F0010*
0 0 0 1,-2 0 2 0,FF10F010*
0 0 0 2,0 1 2 3,F01FF010*
0 0 0 1,-2 0 2 1,0F1FF010*
0 0 0 1,0 1 2 2,FF1F0010*
0 0 0 1,0 3 2 2,FF1FF010*
0 0 0 1,0 0 0 1,1FFF0FFF*
0 0 0 3,0 0 0 1 0 2 0 3,1FFF0FFF*
0 0 0 3,0 0 0 2 0 1 0 3,1FFF0FFF*
1 0 2 0,0 0 1 0 2 0 3 0,1FF0FF10*
0 0 0 3,0 0 0 1 1 1 0 2 0 3 4 4 0 2 0 1,10F00F1F*
0 0 0 3,0 0 0 1 1 1 0 2 0 3 4 4 0 2 0 1.5,10100F1F*
0 0 0 3,0 -100 0 -99 3 3 0 -1 0 1 0 2 0 4,1FF0FF10*
0 1 1 0 0 0 0 -1 -1 0 0 0 0 1,0 1 0 -1 -1 0 1 0 0 1,1FFF0FFF*

)";
}  // namespace datasource

using std::string;
using std::vector;

inline vector<string> SplitString(const string& raw, char delimitor,
                                  bool skip_dup = false) {
  int index = 0;
  vector<string> result;
  while (index < raw.size()) {
    int pos = raw.find_first_of(delimitor, index);
    if (pos == string::npos) {
      pos = raw.size();
    }
    if (!skip_dup || pos != index) {
      result.push_back(raw.substr(index, pos - index));
    }
    index = pos + 1;
  }
  return result;
}

// should be underscore splitted
inline vector<double> ToDoubleArray(const string& str_raw) {
  auto tmp_vec = SplitString(str_raw, ' ', true);
  vector<double> result;
  for (auto& str : tmp_vec) {
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
    if (vec.size() == 0) {
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
