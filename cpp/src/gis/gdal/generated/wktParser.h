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

// Generated from wkt.g4 by ANTLR 4.7

#pragma once

#include "antlr4-runtime/antlr4-runtime.h"

#include <string>
#include <vector>

class wktParser : public antlr4::Parser {
 public:
  enum {
    DECIMAL = 1,
    INTEGERPART = 2,
    DECIMALPART = 3,
    COMMA = 4,
    LPAR = 5,
    RPAR = 6,
    POINT = 7,
    LINESTRING = 8,
    POLYGON = 9,
    MULTIPOINT = 10,
    MULTILINESTRING = 11,
    MULTIPOLYGON = 12,
    GEOMETRYCOLLECTION = 13,
    EMPTY = 14,
    CIRCULARSTRING = 15,
    COMPOUNDCURVE = 16,
    CURVEPOLYGON = 17,
    MULTICURVE = 18,
    TRIANGLE = 19,
    TIN = 20,
    POLYHEDRALSURFACE = 21,
    STRING = 22,
    WS = 23
  };

  enum {
    RuleGeometryCollection = 0,
    RuleGeometry = 1,
    RulePointGeometry = 2,
    RuleLineStringGeometry = 3,
    RulePolygonGeometry = 4,
    RuleMultiPointGeometry = 5,
    RuleMultiLineStringGeometry = 6,
    RuleMultiPolygonGeometry = 7,
    RuleCircularStringGeometry = 8,
    RulePointOrClosedPoint = 9,
    RulePolygon = 10,
    RuleLineString = 11,
    RulePoint = 12,
    RuleName = 13
  };

  explicit wktParser(antlr4::TokenStream* input);
  ~wktParser();

  std::string getGrammarFileName() const override;
  const antlr4::atn::ATN& getATN() const override { return _atn; };
  const std::vector<std::string>& getTokenNames() const override {
    return _tokenNames;
  };  // deprecated: use vocabulary instead.
  const std::vector<std::string>& getRuleNames() const override;
  antlr4::dfa::Vocabulary& getVocabulary() const override;

  class GeometryCollectionContext;
  class GeometryContext;
  class PointGeometryContext;
  class LineStringGeometryContext;
  class PolygonGeometryContext;
  class MultiPointGeometryContext;
  class MultiLineStringGeometryContext;
  class MultiPolygonGeometryContext;
  class CircularStringGeometryContext;
  class PointOrClosedPointContext;
  class PolygonContext;
  class LineStringContext;
  class PointContext;
  class NameContext;

  class GeometryCollectionContext : public antlr4::ParserRuleContext {
   public:
    GeometryCollectionContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* GEOMETRYCOLLECTION();
    antlr4::tree::TerminalNode* LPAR();
    std::vector<GeometryContext*> geometry();
    GeometryContext* geometry(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    antlr4::tree::TerminalNode* EMPTY();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  GeometryCollectionContext* geometryCollection();

  class GeometryContext : public antlr4::ParserRuleContext {
   public:
    GeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    PolygonGeometryContext* polygonGeometry();
    LineStringGeometryContext* lineStringGeometry();
    PointGeometryContext* pointGeometry();
    MultiPointGeometryContext* multiPointGeometry();
    MultiLineStringGeometryContext* multiLineStringGeometry();
    MultiPolygonGeometryContext* multiPolygonGeometry();
    CircularStringGeometryContext* circularStringGeometry();
    GeometryCollectionContext* geometryCollection();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  GeometryContext* geometry();

  class PointGeometryContext : public antlr4::ParserRuleContext {
   public:
    PointGeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* POINT();
    antlr4::tree::TerminalNode* EMPTY();
    antlr4::tree::TerminalNode* LPAR();
    PointContext* point();
    antlr4::tree::TerminalNode* RPAR();
    NameContext* name();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  PointGeometryContext* pointGeometry();

  class LineStringGeometryContext : public antlr4::ParserRuleContext {
   public:
    LineStringGeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* LINESTRING();
    LineStringContext* lineString();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  LineStringGeometryContext* lineStringGeometry();

  class PolygonGeometryContext : public antlr4::ParserRuleContext {
   public:
    PolygonGeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* POLYGON();
    PolygonContext* polygon();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  PolygonGeometryContext* polygonGeometry();

  class MultiPointGeometryContext : public antlr4::ParserRuleContext {
   public:
    MultiPointGeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* MULTIPOINT();
    antlr4::tree::TerminalNode* EMPTY();
    antlr4::tree::TerminalNode* LPAR();
    std::vector<PointOrClosedPointContext*> pointOrClosedPoint();
    PointOrClosedPointContext* pointOrClosedPoint(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  MultiPointGeometryContext* multiPointGeometry();

  class MultiLineStringGeometryContext : public antlr4::ParserRuleContext {
   public:
    MultiLineStringGeometryContext(antlr4::ParserRuleContext* parent,
                                   size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* MULTILINESTRING();
    antlr4::tree::TerminalNode* EMPTY();
    antlr4::tree::TerminalNode* LPAR();
    std::vector<LineStringContext*> lineString();
    LineStringContext* lineString(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  MultiLineStringGeometryContext* multiLineStringGeometry();

  class MultiPolygonGeometryContext : public antlr4::ParserRuleContext {
   public:
    MultiPolygonGeometryContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* MULTIPOLYGON();
    antlr4::tree::TerminalNode* EMPTY();
    antlr4::tree::TerminalNode* LPAR();
    std::vector<PolygonContext*> polygon();
    PolygonContext* polygon(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  MultiPolygonGeometryContext* multiPolygonGeometry();

  class CircularStringGeometryContext : public antlr4::ParserRuleContext {
   public:
    CircularStringGeometryContext(antlr4::ParserRuleContext* parent,
                                  size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* CIRCULARSTRING();
    antlr4::tree::TerminalNode* LPAR();
    std::vector<PointContext*> point();
    PointContext* point(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  CircularStringGeometryContext* circularStringGeometry();

  class PointOrClosedPointContext : public antlr4::ParserRuleContext {
   public:
    PointOrClosedPointContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    PointContext* point();
    antlr4::tree::TerminalNode* LPAR();
    antlr4::tree::TerminalNode* RPAR();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  PointOrClosedPointContext* pointOrClosedPoint();

  class PolygonContext : public antlr4::ParserRuleContext {
   public:
    PolygonContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* LPAR();
    std::vector<LineStringContext*> lineString();
    LineStringContext* lineString(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode* EMPTY();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  PolygonContext* polygon();

  class LineStringContext : public antlr4::ParserRuleContext {
   public:
    LineStringContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* LPAR();
    std::vector<PointContext*> point();
    PointContext* point(size_t i);
    antlr4::tree::TerminalNode* RPAR();
    std::vector<antlr4::tree::TerminalNode*> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode* EMPTY();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  LineStringContext* lineString();

  class PointContext : public antlr4::ParserRuleContext {
   public:
    PointContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode*> DECIMAL();
    antlr4::tree::TerminalNode* DECIMAL(size_t i);

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  PointContext* point();

  class NameContext : public antlr4::ParserRuleContext {
   public:
    NameContext(antlr4::ParserRuleContext* parent, size_t invokingState);
    size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode* STRING();

    void enterRule(antlr4::tree::ParseTreeListener* listener) override;
    void exitRule(antlr4::tree::ParseTreeListener* listener) override;
  };

  NameContext* name();

 private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};
