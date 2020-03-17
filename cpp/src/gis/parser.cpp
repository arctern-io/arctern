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
  return false;
}

bool NextToken(const char* src, TokenInfo* token) {
  if (src == '\0') return false;
  token->start = src;
  token->len = 0;

  while (src != '\0') {
    char input = *(src);
    src++;
    switch (token->type) {
      case TokenType::WktKey: {
          if(IsWhiteSpace(input)){
              return true;
          }else if(input=='('){
              --src;
              return true;
          }else if(IsAlphabet(input)){
              token->len++;
              break;
          }else{
              token->type=TokenType::Unknown;
              return true;
          }
      }
      case TokenType::Number: {
          if(IsWhiteSpace(input)){
              return true;
          }else if(input==',' || input==')'){
              --src;
              return true;
          }else if(IsNumber(input)){
              token->len++;
              break;
          }else{
              token->type==TokenType::Unknown;
              return true;
          }
      }
      case TokenType::Unknown: {
        if (IsWhiteSpace(input)) {
          break;
        } else if (IsAlphabet(input)) {
          token->type = TokenType::WktKey;
          token->len++;
          break;
        } else if (IsNumber(input)) {
          token->type = TokenType::Number;
          token->len++;
          break;
        } else if (input == '(') {
          token->type = TokenType::LeftBracket;
          token->len++;
          return true;
        } else if (input == ')') {
          token->type == TokenType::RightBracket;
          token->len++;
          return true;
        } else if (input == ',') {
          token->type == TokenType::Comma;
          token->len++;
          return true;
        } else {  // return unkonwn tokentype
          return true;
        }
      }
      default:{
          return true;
      }
    }
  }
  return false;
}

bool IsValidWkt(const char *src){
    if(src==nullptr) return false;
    TokenInfo token;
    int bracet_cnt = 0;
    while(NextToken(src,&token)){
        src+=token.len;
    }
    return false;
}

}  // namespace parser
}  // namespace gis
}  // namespace arctern
