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

#include "gis/gdal/generated/wktParser.h"
#include "gis/gdal/generated/wktListener.h"

#include <string>
#include <vector>

wktParser::wktParser(antlr4::TokenStream* input) : Parser(input) {
  _interpreter = new antlr4::atn::ParserATNSimulator(this, _atn, _decisionToDFA,
                                                     _sharedContextCache);
}

wktParser::~wktParser() { delete _interpreter; }

std::string wktParser::getGrammarFileName() const { return "wkt.g4"; }

const std::vector<std::string>& wktParser::getRuleNames() const { return _ruleNames; }

antlr4::dfa::Vocabulary& wktParser::getVocabulary() const { return _vocabulary; }

//----------------- GeometryCollectionContext
//------------------------------------------------------------------

wktParser::GeometryCollectionContext::GeometryCollectionContext(ParserRuleContext* parent,
                                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::GeometryCollectionContext::GEOMETRYCOLLECTION() {
  return getToken(wktParser::GEOMETRYCOLLECTION, 0);
}

antlr4::tree::TerminalNode* wktParser::GeometryCollectionContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::GeometryContext*>
wktParser::GeometryCollectionContext::geometry() {
  return getRuleContexts<wktParser::GeometryContext>();
}

wktParser::GeometryContext* wktParser::GeometryCollectionContext::geometry(size_t i) {
  return getRuleContext<wktParser::GeometryContext>(i);
}

antlr4::tree::TerminalNode* wktParser::GeometryCollectionContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

antlr4::tree::TerminalNode* wktParser::GeometryCollectionContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

std::vector<antlr4::tree::TerminalNode*> wktParser::GeometryCollectionContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::GeometryCollectionContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

size_t wktParser::GeometryCollectionContext::getRuleIndex() const {
  return wktParser::RuleGeometryCollection;
}

void wktParser::GeometryCollectionContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterGeometryCollection(this);
}

void wktParser::GeometryCollectionContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitGeometryCollection(this);
}

wktParser::GeometryCollectionContext* wktParser::geometryCollection() {
  GeometryCollectionContext* _localctx =
      _tracker.createInstance<GeometryCollectionContext>(_ctx, getState());
  enterRule(_localctx, 0, wktParser::RuleGeometryCollection);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(28);
    match(wktParser::GEOMETRYCOLLECTION);
    setState(41);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(29);
        match(wktParser::LPAR);
        setState(30);
        geometry();
        setState(35);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(31);
          match(wktParser::COMMA);
          setState(32);
          geometry();
          setState(37);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(38);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(40);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GeometryContext
//------------------------------------------------------------------

wktParser::GeometryContext::GeometryContext(ParserRuleContext* parent,
                                            size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

wktParser::PolygonGeometryContext* wktParser::GeometryContext::polygonGeometry() {
  return getRuleContext<wktParser::PolygonGeometryContext>(0);
}

wktParser::LineStringGeometryContext* wktParser::GeometryContext::lineStringGeometry() {
  return getRuleContext<wktParser::LineStringGeometryContext>(0);
}

wktParser::PointGeometryContext* wktParser::GeometryContext::pointGeometry() {
  return getRuleContext<wktParser::PointGeometryContext>(0);
}

wktParser::MultiPointGeometryContext* wktParser::GeometryContext::multiPointGeometry() {
  return getRuleContext<wktParser::MultiPointGeometryContext>(0);
}

wktParser::MultiLineStringGeometryContext*
wktParser::GeometryContext::multiLineStringGeometry() {
  return getRuleContext<wktParser::MultiLineStringGeometryContext>(0);
}

wktParser::MultiPolygonGeometryContext*
wktParser::GeometryContext::multiPolygonGeometry() {
  return getRuleContext<wktParser::MultiPolygonGeometryContext>(0);
}

wktParser::CircularStringGeometryContext*
wktParser::GeometryContext::circularStringGeometry() {
  return getRuleContext<wktParser::CircularStringGeometryContext>(0);
}

wktParser::GeometryCollectionContext* wktParser::GeometryContext::geometryCollection() {
  return getRuleContext<wktParser::GeometryCollectionContext>(0);
}

size_t wktParser::GeometryContext::getRuleIndex() const {
  return wktParser::RuleGeometry;
}

void wktParser::GeometryContext::enterRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterGeometry(this);
}

void wktParser::GeometryContext::exitRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitGeometry(this);
}

wktParser::GeometryContext* wktParser::geometry() {
  GeometryContext* _localctx = _tracker.createInstance<GeometryContext>(_ctx, getState());
  enterRule(_localctx, 2, wktParser::RuleGeometry);

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(51);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::POLYGON: {
        setState(43);
        polygonGeometry();
        break;
      }

      case wktParser::LINESTRING: {
        setState(44);
        lineStringGeometry();
        break;
      }

      case wktParser::POINT: {
        setState(45);
        pointGeometry();
        break;
      }

      case wktParser::MULTIPOINT: {
        setState(46);
        multiPointGeometry();
        break;
      }

      case wktParser::MULTILINESTRING: {
        setState(47);
        multiLineStringGeometry();
        break;
      }

      case wktParser::MULTIPOLYGON: {
        setState(48);
        multiPolygonGeometry();
        break;
      }

      case wktParser::CIRCULARSTRING: {
        setState(49);
        circularStringGeometry();
        break;
      }

      case wktParser::GEOMETRYCOLLECTION: {
        setState(50);
        geometryCollection();
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointGeometryContext
//------------------------------------------------------------------

wktParser::PointGeometryContext::PointGeometryContext(ParserRuleContext* parent,
                                                      size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::PointGeometryContext::POINT() {
  return getToken(wktParser::POINT, 0);
}

antlr4::tree::TerminalNode* wktParser::PointGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

antlr4::tree::TerminalNode* wktParser::PointGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

wktParser::PointContext* wktParser::PointGeometryContext::point() {
  return getRuleContext<wktParser::PointContext>(0);
}

antlr4::tree::TerminalNode* wktParser::PointGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

wktParser::NameContext* wktParser::PointGeometryContext::name() {
  return getRuleContext<wktParser::NameContext>(0);
}

size_t wktParser::PointGeometryContext::getRuleIndex() const {
  return wktParser::RulePointGeometry;
}

void wktParser::PointGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterPointGeometry(this);
}

void wktParser::PointGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitPointGeometry(this);
}

wktParser::PointGeometryContext* wktParser::pointGeometry() {
  PointGeometryContext* _localctx =
      _tracker.createInstance<PointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 4, wktParser::RulePointGeometry);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(53);
    match(wktParser::POINT);
    setState(62);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR:
      case wktParser::STRING: {
        setState(55);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == wktParser::STRING) {
          setState(54);
          name();
        }
        setState(57);
        match(wktParser::LPAR);
        setState(58);
        point();
        setState(59);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(61);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringGeometryContext
//------------------------------------------------------------------

wktParser::LineStringGeometryContext::LineStringGeometryContext(ParserRuleContext* parent,
                                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::LineStringGeometryContext::LINESTRING() {
  return getToken(wktParser::LINESTRING, 0);
}

wktParser::LineStringContext* wktParser::LineStringGeometryContext::lineString() {
  return getRuleContext<wktParser::LineStringContext>(0);
}

size_t wktParser::LineStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleLineStringGeometry;
}

void wktParser::LineStringGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterLineStringGeometry(this);
}

void wktParser::LineStringGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitLineStringGeometry(this);
}

wktParser::LineStringGeometryContext* wktParser::lineStringGeometry() {
  LineStringGeometryContext* _localctx =
      _tracker.createInstance<LineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 6, wktParser::RuleLineStringGeometry);

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(64);
    match(wktParser::LINESTRING);
    setState(65);
    lineString();
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonGeometryContext
//------------------------------------------------------------------

wktParser::PolygonGeometryContext::PolygonGeometryContext(ParserRuleContext* parent,
                                                          size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::PolygonGeometryContext::POLYGON() {
  return getToken(wktParser::POLYGON, 0);
}

wktParser::PolygonContext* wktParser::PolygonGeometryContext::polygon() {
  return getRuleContext<wktParser::PolygonContext>(0);
}

size_t wktParser::PolygonGeometryContext::getRuleIndex() const {
  return wktParser::RulePolygonGeometry;
}

void wktParser::PolygonGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterPolygonGeometry(this);
}

void wktParser::PolygonGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitPolygonGeometry(this);
}

wktParser::PolygonGeometryContext* wktParser::polygonGeometry() {
  PolygonGeometryContext* _localctx =
      _tracker.createInstance<PolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 8, wktParser::RulePolygonGeometry);

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(67);
    match(wktParser::POLYGON);
    setState(68);
    polygon();
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPointGeometryContext
//------------------------------------------------------------------

wktParser::MultiPointGeometryContext::MultiPointGeometryContext(ParserRuleContext* parent,
                                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::MultiPointGeometryContext::MULTIPOINT() {
  return getToken(wktParser::MULTIPOINT, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiPointGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiPointGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointOrClosedPointContext*>
wktParser::MultiPointGeometryContext::pointOrClosedPoint() {
  return getRuleContexts<wktParser::PointOrClosedPointContext>();
}

wktParser::PointOrClosedPointContext*
wktParser::MultiPointGeometryContext::pointOrClosedPoint(size_t i) {
  return getRuleContext<wktParser::PointOrClosedPointContext>(i);
}

antlr4::tree::TerminalNode* wktParser::MultiPointGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*> wktParser::MultiPointGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::MultiPointGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

size_t wktParser::MultiPointGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiPointGeometry;
}

void wktParser::MultiPointGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterMultiPointGeometry(this);
}

void wktParser::MultiPointGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitMultiPointGeometry(this);
}

wktParser::MultiPointGeometryContext* wktParser::multiPointGeometry() {
  MultiPointGeometryContext* _localctx =
      _tracker.createInstance<MultiPointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 10, wktParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(70);
    match(wktParser::MULTIPOINT);
    setState(83);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(71);
        match(wktParser::LPAR);
        setState(72);
        pointOrClosedPoint();
        setState(77);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(73);
          match(wktParser::COMMA);
          setState(74);
          pointOrClosedPoint();
          setState(79);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(80);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(82);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiLineStringGeometryContext
//------------------------------------------------------------------

wktParser::MultiLineStringGeometryContext::MultiLineStringGeometryContext(
    ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::MultiLineStringGeometryContext::MULTILINESTRING() {
  return getToken(wktParser::MULTILINESTRING, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiLineStringGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiLineStringGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::LineStringContext*>
wktParser::MultiLineStringGeometryContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::MultiLineStringGeometryContext::lineString(
    size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

antlr4::tree::TerminalNode* wktParser::MultiLineStringGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*>
wktParser::MultiLineStringGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::MultiLineStringGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

size_t wktParser::MultiLineStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiLineStringGeometry;
}

void wktParser::MultiLineStringGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterMultiLineStringGeometry(this);
}

void wktParser::MultiLineStringGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitMultiLineStringGeometry(this);
}

wktParser::MultiLineStringGeometryContext* wktParser::multiLineStringGeometry() {
  MultiLineStringGeometryContext* _localctx =
      _tracker.createInstance<MultiLineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 12, wktParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(85);
    match(wktParser::MULTILINESTRING);
    setState(98);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(86);
        match(wktParser::LPAR);
        setState(87);
        lineString();
        setState(92);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(88);
          match(wktParser::COMMA);
          setState(89);
          lineString();
          setState(94);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(95);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(97);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPolygonGeometryContext
//------------------------------------------------------------------

wktParser::MultiPolygonGeometryContext::MultiPolygonGeometryContext(
    ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::MultiPolygonGeometryContext::MULTIPOLYGON() {
  return getToken(wktParser::MULTIPOLYGON, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiPolygonGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

antlr4::tree::TerminalNode* wktParser::MultiPolygonGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PolygonContext*>
wktParser::MultiPolygonGeometryContext::polygon() {
  return getRuleContexts<wktParser::PolygonContext>();
}

wktParser::PolygonContext* wktParser::MultiPolygonGeometryContext::polygon(size_t i) {
  return getRuleContext<wktParser::PolygonContext>(i);
}

antlr4::tree::TerminalNode* wktParser::MultiPolygonGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*> wktParser::MultiPolygonGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::MultiPolygonGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

size_t wktParser::MultiPolygonGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiPolygonGeometry;
}

void wktParser::MultiPolygonGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterMultiPolygonGeometry(this);
}

void wktParser::MultiPolygonGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitMultiPolygonGeometry(this);
}

wktParser::MultiPolygonGeometryContext* wktParser::multiPolygonGeometry() {
  MultiPolygonGeometryContext* _localctx =
      _tracker.createInstance<MultiPolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 14, wktParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(100);
    match(wktParser::MULTIPOLYGON);
    setState(113);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(101);
        match(wktParser::LPAR);
        setState(102);
        polygon();
        setState(107);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(103);
          match(wktParser::COMMA);
          setState(104);
          polygon();
          setState(109);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(110);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(112);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CircularStringGeometryContext
//------------------------------------------------------------------

wktParser::CircularStringGeometryContext::CircularStringGeometryContext(
    ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::CircularStringGeometryContext::CIRCULARSTRING() {
  return getToken(wktParser::CIRCULARSTRING, 0);
}

antlr4::tree::TerminalNode* wktParser::CircularStringGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointContext*> wktParser::CircularStringGeometryContext::point() {
  return getRuleContexts<wktParser::PointContext>();
}

wktParser::PointContext* wktParser::CircularStringGeometryContext::point(size_t i) {
  return getRuleContext<wktParser::PointContext>(i);
}

antlr4::tree::TerminalNode* wktParser::CircularStringGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*>
wktParser::CircularStringGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::CircularStringGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

size_t wktParser::CircularStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleCircularStringGeometry;
}

void wktParser::CircularStringGeometryContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterCircularStringGeometry(this);
}

void wktParser::CircularStringGeometryContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitCircularStringGeometry(this);
}

wktParser::CircularStringGeometryContext* wktParser::circularStringGeometry() {
  CircularStringGeometryContext* _localctx =
      _tracker.createInstance<CircularStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 16, wktParser::RuleCircularStringGeometry);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(115);
    match(wktParser::CIRCULARSTRING);
    setState(116);
    match(wktParser::LPAR);
    setState(117);
    point();
    setState(122);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(118);
      match(wktParser::COMMA);
      setState(119);
      point();
      setState(124);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(125);
    match(wktParser::RPAR);
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointOrClosedPointContext
//------------------------------------------------------------------

wktParser::PointOrClosedPointContext::PointOrClosedPointContext(ParserRuleContext* parent,
                                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

wktParser::PointContext* wktParser::PointOrClosedPointContext::point() {
  return getRuleContext<wktParser::PointContext>(0);
}

antlr4::tree::TerminalNode* wktParser::PointOrClosedPointContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

antlr4::tree::TerminalNode* wktParser::PointOrClosedPointContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

size_t wktParser::PointOrClosedPointContext::getRuleIndex() const {
  return wktParser::RulePointOrClosedPoint;
}

void wktParser::PointOrClosedPointContext::enterRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterPointOrClosedPoint(this);
}

void wktParser::PointOrClosedPointContext::exitRule(
    antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitPointOrClosedPoint(this);
}

wktParser::PointOrClosedPointContext* wktParser::pointOrClosedPoint() {
  PointOrClosedPointContext* _localctx =
      _tracker.createInstance<PointOrClosedPointContext>(_ctx, getState());
  enterRule(_localctx, 18, wktParser::RulePointOrClosedPoint);

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    setState(132);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::DECIMAL: {
        enterOuterAlt(_localctx, 1);
        setState(127);
        point();
        break;
      }

      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 2);
        setState(128);
        match(wktParser::LPAR);
        setState(129);
        point();
        setState(130);
        match(wktParser::RPAR);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonContext
//------------------------------------------------------------------

wktParser::PolygonContext::PolygonContext(ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::PolygonContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::LineStringContext*> wktParser::PolygonContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::PolygonContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

antlr4::tree::TerminalNode* wktParser::PolygonContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*> wktParser::PolygonContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::PolygonContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

antlr4::tree::TerminalNode* wktParser::PolygonContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

size_t wktParser::PolygonContext::getRuleIndex() const { return wktParser::RulePolygon; }

void wktParser::PolygonContext::enterRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterPolygon(this);
}

void wktParser::PolygonContext::exitRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitPolygon(this);
}

wktParser::PolygonContext* wktParser::polygon() {
  PolygonContext* _localctx = _tracker.createInstance<PolygonContext>(_ctx, getState());
  enterRule(_localctx, 20, wktParser::RulePolygon);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    setState(146);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 1);
        setState(134);
        match(wktParser::LPAR);
        setState(135);
        lineString();
        setState(140);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(136);
          match(wktParser::COMMA);
          setState(137);
          lineString();
          setState(142);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(143);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        enterOuterAlt(_localctx, 2);
        setState(145);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringContext
//------------------------------------------------------------------

wktParser::LineStringContext::LineStringContext(ParserRuleContext* parent,
                                                size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::LineStringContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointContext*> wktParser::LineStringContext::point() {
  return getRuleContexts<wktParser::PointContext>();
}

wktParser::PointContext* wktParser::LineStringContext::point(size_t i) {
  return getRuleContext<wktParser::PointContext>(i);
}

antlr4::tree::TerminalNode* wktParser::LineStringContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<antlr4::tree::TerminalNode*> wktParser::LineStringContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

antlr4::tree::TerminalNode* wktParser::LineStringContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}

antlr4::tree::TerminalNode* wktParser::LineStringContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

size_t wktParser::LineStringContext::getRuleIndex() const {
  return wktParser::RuleLineString;
}

void wktParser::LineStringContext::enterRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterLineString(this);
}

void wktParser::LineStringContext::exitRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitLineString(this);
}

wktParser::LineStringContext* wktParser::lineString() {
  LineStringContext* _localctx =
      _tracker.createInstance<LineStringContext>(_ctx, getState());
  enterRule(_localctx, 22, wktParser::RuleLineString);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    setState(160);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 1);
        setState(148);
        match(wktParser::LPAR);
        setState(149);
        point();
        setState(154);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(150);
          match(wktParser::COMMA);
          setState(151);
          point();
          setState(156);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(157);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        enterOuterAlt(_localctx, 2);
        setState(159);
        match(wktParser::EMPTY);
        break;
      }

      default:
        throw antlr4::NoViableAltException(this);
    }
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointContext
//------------------------------------------------------------------

wktParser::PointContext::PointContext(ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

std::vector<antlr4::tree::TerminalNode*> wktParser::PointContext::DECIMAL() {
  return getTokens(wktParser::DECIMAL);
}

antlr4::tree::TerminalNode* wktParser::PointContext::DECIMAL(size_t i) {
  return getToken(wktParser::DECIMAL, i);
}

size_t wktParser::PointContext::getRuleIndex() const { return wktParser::RulePoint; }

void wktParser::PointContext::enterRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterPoint(this);
}

void wktParser::PointContext::exitRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitPoint(this);
}

wktParser::PointContext* wktParser::point() {
  PointContext* _localctx = _tracker.createInstance<PointContext>(_ctx, getState());
  enterRule(_localctx, 24, wktParser::RulePoint);
  size_t _la = 0;

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(163);
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(162);
      match(wktParser::DECIMAL);
      setState(165);
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == wktParser::DECIMAL);
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NameContext
//------------------------------------------------------------------

wktParser::NameContext::NameContext(ParserRuleContext* parent, size_t invokingState)
    : ParserRuleContext(parent, invokingState) {}

antlr4::tree::TerminalNode* wktParser::NameContext::STRING() {
  return getToken(wktParser::STRING, 0);
}

size_t wktParser::NameContext::getRuleIndex() const { return wktParser::RuleName; }

void wktParser::NameContext::enterRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->enterName(this);
}

void wktParser::NameContext::exitRule(antlr4::tree::ParseTreeListener* listener) {
  auto parserListener = dynamic_cast<wktListener*>(listener);
  if (parserListener != nullptr) parserListener->exitName(this);
}

wktParser::NameContext* wktParser::name() {
  NameContext* _localctx = _tracker.createInstance<NameContext>(_ctx, getState());
  enterRule(_localctx, 26, wktParser::RuleName);

  auto onExit = antlrcpp::finally([=] { exitRule(); });
  try {
    enterOuterAlt(_localctx, 1);
    setState(167);
    match(wktParser::STRING);
  } catch (antlr4::RecognitionException& e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<antlr4::dfa::DFA> wktParser::_decisionToDFA;
antlr4::atn::PredictionContextCache wktParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
antlr4::atn::ATN wktParser::_atn;
std::vector<uint16_t> wktParser::_serializedATN;

std::vector<std::string> wktParser::_ruleNames = {"geometryCollection",
                                                  "geometry",
                                                  "pointGeometry",
                                                  "lineStringGeometry",
                                                  "polygonGeometry",
                                                  "multiPointGeometry",
                                                  "multiLineStringGeometry",
                                                  "multiPolygonGeometry",
                                                  "circularStringGeometry",
                                                  "pointOrClosedPoint",
                                                  "polygon",
                                                  "lineString",
                                                  "point",
                                                  "name"};

std::vector<std::string> wktParser::_literalNames = {"", "", "", "", "','", "'('", "')'"};

std::vector<std::string> wktParser::_symbolicNames = {"",
                                                      "DECIMAL",
                                                      "INTEGERPART",
                                                      "DECIMALPART",
                                                      "COMMA",
                                                      "LPAR",
                                                      "RPAR",
                                                      "POINT",
                                                      "LINESTRING",
                                                      "POLYGON",
                                                      "MULTIPOINT",
                                                      "MULTILINESTRING",
                                                      "MULTIPOLYGON",
                                                      "GEOMETRYCOLLECTION",
                                                      "EMPTY",
                                                      "CIRCULARSTRING",
                                                      "COMPOUNDCURVE",
                                                      "CURVEPOLYGON",
                                                      "MULTICURVE",
                                                      "TRIANGLE",
                                                      "TIN",
                                                      "POLYHEDRALSURFACE",
                                                      "STRING",
                                                      "WS"};

antlr4::dfa::Vocabulary wktParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> wktParser::_tokenNames;

wktParser::Initializer::Initializer() {
  for (size_t i = 0; i < _symbolicNames.size(); ++i) {
    std::string name = _vocabulary.getLiteralName(i);
    if (name.empty()) {
      name = _vocabulary.getSymbolicName(i);
    }

    if (name.empty()) {
      _tokenNames.push_back("<INVALID>");
    } else {
      _tokenNames.push_back(name);
    }
  }

  _serializedATN = {
      0x3,  0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 0x3,  0x19,
      0xac, 0x4,    0x2,    0x9,    0x2,    0x4,    0x3,    0x9,    0x3,    0x4,  0x4,
      0x9,  0x4,    0x4,    0x5,    0x9,    0x5,    0x4,    0x6,    0x9,    0x6,  0x4,
      0x7,  0x9,    0x7,    0x4,    0x8,    0x9,    0x8,    0x4,    0x9,    0x9,  0x9,
      0x4,  0xa,    0x9,    0xa,    0x4,    0xb,    0x9,    0xb,    0x4,    0xc,  0x9,
      0xc,  0x4,    0xd,    0x9,    0xd,    0x4,    0xe,    0x9,    0xe,    0x4,  0xf,
      0x9,  0xf,    0x3,    0x2,    0x3,    0x2,    0x3,    0x2,    0x3,    0x2,  0x3,
      0x2,  0x7,    0x2,    0x24,   0xa,    0x2,    0xc,    0x2,    0xe,    0x2,  0x27,
      0xb,  0x2,    0x3,    0x2,    0x3,    0x2,    0x3,    0x2,    0x5,    0x2,  0x2c,
      0xa,  0x2,    0x3,    0x3,    0x3,    0x3,    0x3,    0x3,    0x3,    0x3,  0x3,
      0x3,  0x3,    0x3,    0x3,    0x3,    0x3,    0x3,    0x5,    0x3,    0x36, 0xa,
      0x3,  0x3,    0x4,    0x3,    0x4,    0x5,    0x4,    0x3a,   0xa,    0x4,  0x3,
      0x4,  0x3,    0x4,    0x3,    0x4,    0x3,    0x4,    0x3,    0x4,    0x5,  0x4,
      0x41, 0xa,    0x4,    0x3,    0x5,    0x3,    0x5,    0x3,    0x5,    0x3,  0x6,
      0x3,  0x6,    0x3,    0x6,    0x3,    0x7,    0x3,    0x7,    0x3,    0x7,  0x3,
      0x7,  0x3,    0x7,    0x7,    0x7,    0x4e,   0xa,    0x7,    0xc,    0x7,  0xe,
      0x7,  0x51,   0xb,    0x7,    0x3,    0x7,    0x3,    0x7,    0x3,    0x7,  0x5,
      0x7,  0x56,   0xa,    0x7,    0x3,    0x8,    0x3,    0x8,    0x3,    0x8,  0x3,
      0x8,  0x3,    0x8,    0x7,    0x8,    0x5d,   0xa,    0x8,    0xc,    0x8,  0xe,
      0x8,  0x60,   0xb,    0x8,    0x3,    0x8,    0x3,    0x8,    0x3,    0x8,  0x5,
      0x8,  0x65,   0xa,    0x8,    0x3,    0x9,    0x3,    0x9,    0x3,    0x9,  0x3,
      0x9,  0x3,    0x9,    0x7,    0x9,    0x6c,   0xa,    0x9,    0xc,    0x9,  0xe,
      0x9,  0x6f,   0xb,    0x9,    0x3,    0x9,    0x3,    0x9,    0x3,    0x9,  0x5,
      0x9,  0x74,   0xa,    0x9,    0x3,    0xa,    0x3,    0xa,    0x3,    0xa,  0x3,
      0xa,  0x3,    0xa,    0x7,    0xa,    0x7b,   0xa,    0xa,    0xc,    0xa,  0xe,
      0xa,  0x7e,   0xb,    0xa,    0x3,    0xa,    0x3,    0xa,    0x3,    0xb,  0x3,
      0xb,  0x3,    0xb,    0x3,    0xb,    0x3,    0xb,    0x5,    0xb,    0x87, 0xa,
      0xb,  0x3,    0xc,    0x3,    0xc,    0x3,    0xc,    0x3,    0xc,    0x7,  0xc,
      0x8d, 0xa,    0xc,    0xc,    0xc,    0xe,    0xc,    0x90,   0xb,    0xc,  0x3,
      0xc,  0x3,    0xc,    0x3,    0xc,    0x5,    0xc,    0x95,   0xa,    0xc,  0x3,
      0xd,  0x3,    0xd,    0x3,    0xd,    0x3,    0xd,    0x7,    0xd,    0x9b, 0xa,
      0xd,  0xc,    0xd,    0xe,    0xd,    0x9e,   0xb,    0xd,    0x3,    0xd,  0x3,
      0xd,  0x3,    0xd,    0x5,    0xd,    0xa3,   0xa,    0xd,    0x3,    0xe,  0x6,
      0xe,  0xa6,   0xa,    0xe,    0xd,    0xe,    0xe,    0xe,    0xa7,   0x3,  0xf,
      0x3,  0xf,    0x3,    0xf,    0x2,    0x2,    0x10,   0x2,    0x4,    0x6,  0x8,
      0xa,  0xc,    0xe,    0x10,   0x12,   0x14,   0x16,   0x18,   0x1a,   0x1c, 0x2,
      0x2,  0x2,    0xb5,   0x2,    0x1e,   0x3,    0x2,    0x2,    0x2,    0x4,  0x35,
      0x3,  0x2,    0x2,    0x2,    0x6,    0x37,   0x3,    0x2,    0x2,    0x2,  0x8,
      0x42, 0x3,    0x2,    0x2,    0x2,    0xa,    0x45,   0x3,    0x2,    0x2,  0x2,
      0xc,  0x48,   0x3,    0x2,    0x2,    0x2,    0xe,    0x57,   0x3,    0x2,  0x2,
      0x2,  0x10,   0x66,   0x3,    0x2,    0x2,    0x2,    0x12,   0x75,   0x3,  0x2,
      0x2,  0x2,    0x14,   0x86,   0x3,    0x2,    0x2,    0x2,    0x16,   0x94, 0x3,
      0x2,  0x2,    0x2,    0x18,   0xa2,   0x3,    0x2,    0x2,    0x2,    0x1a, 0xa5,
      0x3,  0x2,    0x2,    0x2,    0x1c,   0xa9,   0x3,    0x2,    0x2,    0x2,  0x1e,
      0x2b, 0x7,    0xf,    0x2,    0x2,    0x1f,   0x20,   0x7,    0x7,    0x2,  0x2,
      0x20, 0x25,   0x5,    0x4,    0x3,    0x2,    0x21,   0x22,   0x7,    0x6,  0x2,
      0x2,  0x22,   0x24,   0x5,    0x4,    0x3,    0x2,    0x23,   0x21,   0x3,  0x2,
      0x2,  0x2,    0x24,   0x27,   0x3,    0x2,    0x2,    0x2,    0x25,   0x23, 0x3,
      0x2,  0x2,    0x2,    0x25,   0x26,   0x3,    0x2,    0x2,    0x2,    0x26, 0x28,
      0x3,  0x2,    0x2,    0x2,    0x27,   0x25,   0x3,    0x2,    0x2,    0x2,  0x28,
      0x29, 0x7,    0x8,    0x2,    0x2,    0x29,   0x2c,   0x3,    0x2,    0x2,  0x2,
      0x2a, 0x2c,   0x7,    0x10,   0x2,    0x2,    0x2b,   0x1f,   0x3,    0x2,  0x2,
      0x2,  0x2b,   0x2a,   0x3,    0x2,    0x2,    0x2,    0x2c,   0x3,    0x3,  0x2,
      0x2,  0x2,    0x2d,   0x36,   0x5,    0xa,    0x6,    0x2,    0x2e,   0x36, 0x5,
      0x8,  0x5,    0x2,    0x2f,   0x36,   0x5,    0x6,    0x4,    0x2,    0x30, 0x36,
      0x5,  0xc,    0x7,    0x2,    0x31,   0x36,   0x5,    0xe,    0x8,    0x2,  0x32,
      0x36, 0x5,    0x10,   0x9,    0x2,    0x33,   0x36,   0x5,    0x12,   0xa,  0x2,
      0x34, 0x36,   0x5,    0x2,    0x2,    0x2,    0x35,   0x2d,   0x3,    0x2,  0x2,
      0x2,  0x35,   0x2e,   0x3,    0x2,    0x2,    0x2,    0x35,   0x2f,   0x3,  0x2,
      0x2,  0x2,    0x35,   0x30,   0x3,    0x2,    0x2,    0x2,    0x35,   0x31, 0x3,
      0x2,  0x2,    0x2,    0x35,   0x32,   0x3,    0x2,    0x2,    0x2,    0x35, 0x33,
      0x3,  0x2,    0x2,    0x2,    0x35,   0x34,   0x3,    0x2,    0x2,    0x2,  0x36,
      0x5,  0x3,    0x2,    0x2,    0x2,    0x37,   0x40,   0x7,    0x9,    0x2,  0x2,
      0x38, 0x3a,   0x5,    0x1c,   0xf,    0x2,    0x39,   0x38,   0x3,    0x2,  0x2,
      0x2,  0x39,   0x3a,   0x3,    0x2,    0x2,    0x2,    0x3a,   0x3b,   0x3,  0x2,
      0x2,  0x2,    0x3b,   0x3c,   0x7,    0x7,    0x2,    0x2,    0x3c,   0x3d, 0x5,
      0x1a, 0xe,    0x2,    0x3d,   0x3e,   0x7,    0x8,    0x2,    0x2,    0x3e, 0x41,
      0x3,  0x2,    0x2,    0x2,    0x3f,   0x41,   0x7,    0x10,   0x2,    0x2,  0x40,
      0x39, 0x3,    0x2,    0x2,    0x2,    0x40,   0x3f,   0x3,    0x2,    0x2,  0x2,
      0x41, 0x7,    0x3,    0x2,    0x2,    0x2,    0x42,   0x43,   0x7,    0xa,  0x2,
      0x2,  0x43,   0x44,   0x5,    0x18,   0xd,    0x2,    0x44,   0x9,    0x3,  0x2,
      0x2,  0x2,    0x45,   0x46,   0x7,    0xb,    0x2,    0x2,    0x46,   0x47, 0x5,
      0x16, 0xc,    0x2,    0x47,   0xb,    0x3,    0x2,    0x2,    0x2,    0x48, 0x55,
      0x7,  0xc,    0x2,    0x2,    0x49,   0x4a,   0x7,    0x7,    0x2,    0x2,  0x4a,
      0x4f, 0x5,    0x14,   0xb,    0x2,    0x4b,   0x4c,   0x7,    0x6,    0x2,  0x2,
      0x4c, 0x4e,   0x5,    0x14,   0xb,    0x2,    0x4d,   0x4b,   0x3,    0x2,  0x2,
      0x2,  0x4e,   0x51,   0x3,    0x2,    0x2,    0x2,    0x4f,   0x4d,   0x3,  0x2,
      0x2,  0x2,    0x4f,   0x50,   0x3,    0x2,    0x2,    0x2,    0x50,   0x52, 0x3,
      0x2,  0x2,    0x2,    0x51,   0x4f,   0x3,    0x2,    0x2,    0x2,    0x52, 0x53,
      0x7,  0x8,    0x2,    0x2,    0x53,   0x56,   0x3,    0x2,    0x2,    0x2,  0x54,
      0x56, 0x7,    0x10,   0x2,    0x2,    0x55,   0x49,   0x3,    0x2,    0x2,  0x2,
      0x55, 0x54,   0x3,    0x2,    0x2,    0x2,    0x56,   0xd,    0x3,    0x2,  0x2,
      0x2,  0x57,   0x64,   0x7,    0xd,    0x2,    0x2,    0x58,   0x59,   0x7,  0x7,
      0x2,  0x2,    0x59,   0x5e,   0x5,    0x18,   0xd,    0x2,    0x5a,   0x5b, 0x7,
      0x6,  0x2,    0x2,    0x5b,   0x5d,   0x5,    0x18,   0xd,    0x2,    0x5c, 0x5a,
      0x3,  0x2,    0x2,    0x2,    0x5d,   0x60,   0x3,    0x2,    0x2,    0x2,  0x5e,
      0x5c, 0x3,    0x2,    0x2,    0x2,    0x5e,   0x5f,   0x3,    0x2,    0x2,  0x2,
      0x5f, 0x61,   0x3,    0x2,    0x2,    0x2,    0x60,   0x5e,   0x3,    0x2,  0x2,
      0x2,  0x61,   0x62,   0x7,    0x8,    0x2,    0x2,    0x62,   0x65,   0x3,  0x2,
      0x2,  0x2,    0x63,   0x65,   0x7,    0x10,   0x2,    0x2,    0x64,   0x58, 0x3,
      0x2,  0x2,    0x2,    0x64,   0x63,   0x3,    0x2,    0x2,    0x2,    0x65, 0xf,
      0x3,  0x2,    0x2,    0x2,    0x66,   0x73,   0x7,    0xe,    0x2,    0x2,  0x67,
      0x68, 0x7,    0x7,    0x2,    0x2,    0x68,   0x6d,   0x5,    0x16,   0xc,  0x2,
      0x69, 0x6a,   0x7,    0x6,    0x2,    0x2,    0x6a,   0x6c,   0x5,    0x16, 0xc,
      0x2,  0x6b,   0x69,   0x3,    0x2,    0x2,    0x2,    0x6c,   0x6f,   0x3,  0x2,
      0x2,  0x2,    0x6d,   0x6b,   0x3,    0x2,    0x2,    0x2,    0x6d,   0x6e, 0x3,
      0x2,  0x2,    0x2,    0x6e,   0x70,   0x3,    0x2,    0x2,    0x2,    0x6f, 0x6d,
      0x3,  0x2,    0x2,    0x2,    0x70,   0x71,   0x7,    0x8,    0x2,    0x2,  0x71,
      0x74, 0x3,    0x2,    0x2,    0x2,    0x72,   0x74,   0x7,    0x10,   0x2,  0x2,
      0x73, 0x67,   0x3,    0x2,    0x2,    0x2,    0x73,   0x72,   0x3,    0x2,  0x2,
      0x2,  0x74,   0x11,   0x3,    0x2,    0x2,    0x2,    0x75,   0x76,   0x7,  0x11,
      0x2,  0x2,    0x76,   0x77,   0x7,    0x7,    0x2,    0x2,    0x77,   0x7c, 0x5,
      0x1a, 0xe,    0x2,    0x78,   0x79,   0x7,    0x6,    0x2,    0x2,    0x79, 0x7b,
      0x5,  0x1a,   0xe,    0x2,    0x7a,   0x78,   0x3,    0x2,    0x2,    0x2,  0x7b,
      0x7e, 0x3,    0x2,    0x2,    0x2,    0x7c,   0x7a,   0x3,    0x2,    0x2,  0x2,
      0x7c, 0x7d,   0x3,    0x2,    0x2,    0x2,    0x7d,   0x7f,   0x3,    0x2,  0x2,
      0x2,  0x7e,   0x7c,   0x3,    0x2,    0x2,    0x2,    0x7f,   0x80,   0x7,  0x8,
      0x2,  0x2,    0x80,   0x13,   0x3,    0x2,    0x2,    0x2,    0x81,   0x87, 0x5,
      0x1a, 0xe,    0x2,    0x82,   0x83,   0x7,    0x7,    0x2,    0x2,    0x83, 0x84,
      0x5,  0x1a,   0xe,    0x2,    0x84,   0x85,   0x7,    0x8,    0x2,    0x2,  0x85,
      0x87, 0x3,    0x2,    0x2,    0x2,    0x86,   0x81,   0x3,    0x2,    0x2,  0x2,
      0x86, 0x82,   0x3,    0x2,    0x2,    0x2,    0x87,   0x15,   0x3,    0x2,  0x2,
      0x2,  0x88,   0x89,   0x7,    0x7,    0x2,    0x2,    0x89,   0x8e,   0x5,  0x18,
      0xd,  0x2,    0x8a,   0x8b,   0x7,    0x6,    0x2,    0x2,    0x8b,   0x8d, 0x5,
      0x18, 0xd,    0x2,    0x8c,   0x8a,   0x3,    0x2,    0x2,    0x2,    0x8d, 0x90,
      0x3,  0x2,    0x2,    0x2,    0x8e,   0x8c,   0x3,    0x2,    0x2,    0x2,  0x8e,
      0x8f, 0x3,    0x2,    0x2,    0x2,    0x8f,   0x91,   0x3,    0x2,    0x2,  0x2,
      0x90, 0x8e,   0x3,    0x2,    0x2,    0x2,    0x91,   0x92,   0x7,    0x8,  0x2,
      0x2,  0x92,   0x95,   0x3,    0x2,    0x2,    0x2,    0x93,   0x95,   0x7,  0x10,
      0x2,  0x2,    0x94,   0x88,   0x3,    0x2,    0x2,    0x2,    0x94,   0x93, 0x3,
      0x2,  0x2,    0x2,    0x95,   0x17,   0x3,    0x2,    0x2,    0x2,    0x96, 0x97,
      0x7,  0x7,    0x2,    0x2,    0x97,   0x9c,   0x5,    0x1a,   0xe,    0x2,  0x98,
      0x99, 0x7,    0x6,    0x2,    0x2,    0x99,   0x9b,   0x5,    0x1a,   0xe,  0x2,
      0x9a, 0x98,   0x3,    0x2,    0x2,    0x2,    0x9b,   0x9e,   0x3,    0x2,  0x2,
      0x2,  0x9c,   0x9a,   0x3,    0x2,    0x2,    0x2,    0x9c,   0x9d,   0x3,  0x2,
      0x2,  0x2,    0x9d,   0x9f,   0x3,    0x2,    0x2,    0x2,    0x9e,   0x9c, 0x3,
      0x2,  0x2,    0x2,    0x9f,   0xa0,   0x7,    0x8,    0x2,    0x2,    0xa0, 0xa3,
      0x3,  0x2,    0x2,    0x2,    0xa1,   0xa3,   0x7,    0x10,   0x2,    0x2,  0xa2,
      0x96, 0x3,    0x2,    0x2,    0x2,    0xa2,   0xa1,   0x3,    0x2,    0x2,  0x2,
      0xa3, 0x19,   0x3,    0x2,    0x2,    0x2,    0xa4,   0xa6,   0x7,    0x3,  0x2,
      0x2,  0xa5,   0xa4,   0x3,    0x2,    0x2,    0x2,    0xa6,   0xa7,   0x3,  0x2,
      0x2,  0x2,    0xa7,   0xa5,   0x3,    0x2,    0x2,    0x2,    0xa7,   0xa8, 0x3,
      0x2,  0x2,    0x2,    0xa8,   0x1b,   0x3,    0x2,    0x2,    0x2,    0xa9, 0xaa,
      0x7,  0x18,   0x2,    0x2,    0xaa,   0x1d,   0x3,    0x2,    0x2,    0x2,  0x14,
      0x25, 0x2b,   0x35,   0x39,   0x40,   0x4f,   0x55,   0x5e,   0x64,   0x6d, 0x73,
      0x7c, 0x86,   0x8e,   0x94,   0x9c,   0xa2,   0xa7,
  };

  antlr4::atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) {
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

wktParser::Initializer wktParser::_init;
