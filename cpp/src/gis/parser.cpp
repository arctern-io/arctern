/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gis/parser.h"

namespace arctern {
namespace gis {
namespace parser {

bool IsWhiteSpace(const char c) {
  if (c == ' ' || c == '\t') return true;
  return false;
}

bool IsAlphabet(const char c) {
  if (c >= 'a' && c <= 'z') return true;
  if (c >= 'A' && c <= 'Z') return true;
  return false;
}

bool IsNumber(const char c) {
  if (c >= '0' && c <= '9') return true;
  if (c == '.') return true;
  if (c == '-') return true;
  if (c == '+') return true;
  return false;
}

void SetIfEmpty(TokenInfo* token) {
  static auto empty = (const char*)"empty";
  static auto Empty = (const char*)"EMPTY";
  if (token->type != TokenType::WktKey) return;
  if (token->len != 5) return;
  for (int i = 0; i < token->len; ++i) {
    if ((token->start[i] != empty[i]) && (token->start[i] != Empty[i])) return;
  }
  token->type = TokenType::Empty;
}

bool NextToken(const char* src, TokenInfo* token) {
  if (*src == '\0') return false;
  token->len = 0;
  token->type = TokenType::Unknown;

  while (*src != '\0') {
    char input = *(src);
    switch (token->type) {
      case TokenType::WktKey: {
        if (IsWhiteSpace(input) || input == '(') {
          SetIfEmpty(token);
          return true;
        } else if (IsAlphabet(input)) {
          token->len++;
          break;
        } else {
          token->type = TokenType::Unknown;
          return true;
        }
      }
      case TokenType::Number: {
        if (IsWhiteSpace(input) || input == ',' || input == ')') {
          return true;
        } else if (IsNumber(input)) {
          token->len++;
          break;
        } else {
          token->type = TokenType::Unknown;
          return true;
        }
      }
      case TokenType::Unknown: {
        if (IsWhiteSpace(input)) {
          break;
        } else if (IsAlphabet(input)) {
          token->start = src;
          token->type = TokenType::WktKey;
          token->len++;
          break;
        } else if (IsNumber(input)) {
          token->start = src;
          token->type = TokenType::Number;
          token->len++;
          break;
        } else if (input == '(') {
          token->start = src;
          token->type = TokenType::LeftBracket;
          token->len++;
          return true;
        } else if (input == ')') {
          token->start = src;
          token->type = TokenType::RightBracket;
          token->len++;
          return true;
        } else if (input == ',') {
          token->start = src;
          token->type = TokenType::Comma;
          token->len++;
          return true;
        } else {  // return unkonwn tokentype
          return true;
        }
      }
      default: { return true; }
    }
    src++;
  }
  if (token->type == TokenType::WktKey) {
    SetIfEmpty(token);
    return true;
  }
  return false;
}

bool IsValidWkt(const char* src) {
  if (src == nullptr) return false;
  TokenInfo token;
  int bracket_nest = 0;
  int num_cnt = 0;
  auto pre_type = TokenType::Unknown;

  while (NextToken(src, &token)) {
    if (token.type == TokenType::Unknown) return false;
    switch (pre_type) {
      case TokenType::Unknown: {
        if (token.type != TokenType::WktKey) return false;
        break;
      }
      case TokenType::WktKey: {
        switch (token.type) {
          case TokenType::Empty: {
            break;
          }
          case TokenType::LeftBracket: {
            ++bracket_nest;
            break;
          }
          default: { return false; }
        }
        break;
      }
      case TokenType::LeftBracket: {
        switch (token.type) {
          case TokenType::Number: {
            num_cnt = 1;
            break;
          }
          case TokenType::LeftBracket: {
            ++bracket_nest;
            break;
          }
          case TokenType::WktKey: {
            break;
          }
          default: { return false; }
        }
        break;
      }
      case TokenType::Number: {
        switch (token.type) {
          case TokenType::Number: {
            ++num_cnt;
            break;
          }
          case TokenType::Comma: {
            if ((num_cnt != 2) && (num_cnt != 3)) return false;
            break;
          }
          case TokenType::RightBracket: {
            --bracket_nest;
            if (bracket_nest < 0) return false;
            if ((num_cnt != 2) && (num_cnt != 3)) return false;
            num_cnt = 0;
            break;
          }
          default: { return false; }
        }
        break;
      }
      case TokenType::Comma: {
        switch (token.type) {
          case TokenType::WktKey: {
            break;
          }
          case TokenType::Number: {
            num_cnt = 1;
            break;
          }
          case TokenType::LeftBracket: {
            ++bracket_nest;
            if (num_cnt != 0) return false;
            break;
          }
          case TokenType::RightBracket: {
            --bracket_nest;
            if (bracket_nest < 0) return false;
            if (num_cnt != 0) return false;
            break;
          }
          default: { return false; }
        }
        break;
      }
      case TokenType::RightBracket: {
        switch (token.type) {
          case TokenType::Comma:
          case TokenType::WktKey: {
            break;
          }
          case TokenType::RightBracket: {
            --bracket_nest;
            if (bracket_nest < 0) return false;
            break;
          }
          default: { return false; }
        }
        break;
      }
      case TokenType::Empty: {
        return false;
      }
      default: { return false; }
    }
    pre_type = token.type;
    src = token.start + token.len;
  }
  if (bracket_nest != 0) return false;
  if ((pre_type == TokenType::RightBracket) || (pre_type == TokenType::Empty)) {
    return true;
  }
  return false;
}

}  // namespace parser
}  // namespace gis
}  // namespace arctern
