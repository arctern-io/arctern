
// Generated from wkt.g4 by ANTLR 4.7


#include "wktListener.h"

#include "wktParser.h"


using namespace antlrcpp;
using namespace antlr4;

wktParser::wktParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

wktParser::~wktParser() {
  delete _interpreter;
}

std::string wktParser::getGrammarFileName() const {
  return "wkt.g4";
}

const std::vector<std::string>& wktParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& wktParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- GeometryContext ------------------------------------------------------------------

wktParser::GeometryContext::GeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::GeometryContext::EOF() {
  return getToken(wktParser::EOF, 0);
}

std::vector<wktParser::PolygonGeometryContext *> wktParser::GeometryContext::polygonGeometry() {
  return getRuleContexts<wktParser::PolygonGeometryContext>();
}

wktParser::PolygonGeometryContext* wktParser::GeometryContext::polygonGeometry(size_t i) {
  return getRuleContext<wktParser::PolygonGeometryContext>(i);
}

std::vector<wktParser::LineStringGeometryContext *> wktParser::GeometryContext::lineStringGeometry() {
  return getRuleContexts<wktParser::LineStringGeometryContext>();
}

wktParser::LineStringGeometryContext* wktParser::GeometryContext::lineStringGeometry(size_t i) {
  return getRuleContext<wktParser::LineStringGeometryContext>(i);
}

std::vector<wktParser::PointGeometryContext *> wktParser::GeometryContext::pointGeometry() {
  return getRuleContexts<wktParser::PointGeometryContext>();
}

wktParser::PointGeometryContext* wktParser::GeometryContext::pointGeometry(size_t i) {
  return getRuleContext<wktParser::PointGeometryContext>(i);
}

std::vector<wktParser::MultiPointGeometryContext *> wktParser::GeometryContext::multiPointGeometry() {
  return getRuleContexts<wktParser::MultiPointGeometryContext>();
}

wktParser::MultiPointGeometryContext* wktParser::GeometryContext::multiPointGeometry(size_t i) {
  return getRuleContext<wktParser::MultiPointGeometryContext>(i);
}

std::vector<wktParser::MultiLineStringGeometryContext *> wktParser::GeometryContext::multiLineStringGeometry() {
  return getRuleContexts<wktParser::MultiLineStringGeometryContext>();
}

wktParser::MultiLineStringGeometryContext* wktParser::GeometryContext::multiLineStringGeometry(size_t i) {
  return getRuleContext<wktParser::MultiLineStringGeometryContext>(i);
}

std::vector<wktParser::MultiPolygonGeometryContext *> wktParser::GeometryContext::multiPolygonGeometry() {
  return getRuleContexts<wktParser::MultiPolygonGeometryContext>();
}

wktParser::MultiPolygonGeometryContext* wktParser::GeometryContext::multiPolygonGeometry(size_t i) {
  return getRuleContext<wktParser::MultiPolygonGeometryContext>(i);
}

std::vector<wktParser::CircularStringGeometryContext *> wktParser::GeometryContext::circularStringGeometry() {
  return getRuleContexts<wktParser::CircularStringGeometryContext>();
}

wktParser::CircularStringGeometryContext* wktParser::GeometryContext::circularStringGeometry(size_t i) {
  return getRuleContext<wktParser::CircularStringGeometryContext>(i);
}


size_t wktParser::GeometryContext::getRuleIndex() const {
  return wktParser::RuleGeometry;
}

void wktParser::GeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGeometry(this);
}

void wktParser::GeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGeometry(this);
}

wktParser::GeometryContext* wktParser::geometry() {
  GeometryContext *_localctx = _tracker.createInstance<GeometryContext>(_ctx, getState());
  enterRule(_localctx, 0, wktParser::RuleGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(33); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(33);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case wktParser::POLYGON: {
          setState(26);
          polygonGeometry();
          break;
        }

        case wktParser::LINESTRING: {
          setState(27);
          lineStringGeometry();
          break;
        }

        case wktParser::POINT: {
          setState(28);
          pointGeometry();
          break;
        }

        case wktParser::MULTIPOINT: {
          setState(29);
          multiPointGeometry();
          break;
        }

        case wktParser::MULTILINESTRING: {
          setState(30);
          multiLineStringGeometry();
          break;
        }

        case wktParser::MULTIPOLYGON: {
          setState(31);
          multiPolygonGeometry();
          break;
        }

        case wktParser::CIRCULARSTRING: {
          setState(32);
          circularStringGeometry();
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(35); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << wktParser::POINT)
      | (1ULL << wktParser::LINESTRING)
      | (1ULL << wktParser::POLYGON)
      | (1ULL << wktParser::MULTIPOINT)
      | (1ULL << wktParser::MULTILINESTRING)
      | (1ULL << wktParser::MULTIPOLYGON)
      | (1ULL << wktParser::CIRCULARSTRING))) != 0));
    setState(37);
    match(wktParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointGeometryContext ------------------------------------------------------------------

wktParser::PointGeometryContext::PointGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::PointGeometryContext::POINT() {
  return getToken(wktParser::POINT, 0);
}

tree::TerminalNode* wktParser::PointGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::PointGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

wktParser::PointContext* wktParser::PointGeometryContext::point() {
  return getRuleContext<wktParser::PointContext>(0);
}

tree::TerminalNode* wktParser::PointGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

wktParser::NameContext* wktParser::PointGeometryContext::name() {
  return getRuleContext<wktParser::NameContext>(0);
}


size_t wktParser::PointGeometryContext::getRuleIndex() const {
  return wktParser::RulePointGeometry;
}

void wktParser::PointGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPointGeometry(this);
}

void wktParser::PointGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPointGeometry(this);
}

wktParser::PointGeometryContext* wktParser::pointGeometry() {
  PointGeometryContext *_localctx = _tracker.createInstance<PointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 2, wktParser::RulePointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(39);
    match(wktParser::POINT);
    setState(48);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR:
      case wktParser::STRING: {
        setState(41);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == wktParser::STRING) {
          setState(40);
          name();
        }
        setState(43);
        match(wktParser::LPAR);
        setState(44);
        point();
        setState(45);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(47);
        match(wktParser::EMPTY);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringGeometryContext ------------------------------------------------------------------

wktParser::LineStringGeometryContext::LineStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::LineStringGeometryContext::LINESTRING() {
  return getToken(wktParser::LINESTRING, 0);
}

wktParser::LineStringContext* wktParser::LineStringGeometryContext::lineString() {
  return getRuleContext<wktParser::LineStringContext>(0);
}


size_t wktParser::LineStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleLineStringGeometry;
}

void wktParser::LineStringGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLineStringGeometry(this);
}

void wktParser::LineStringGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLineStringGeometry(this);
}

wktParser::LineStringGeometryContext* wktParser::lineStringGeometry() {
  LineStringGeometryContext *_localctx = _tracker.createInstance<LineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 4, wktParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(50);
    match(wktParser::LINESTRING);
    setState(51);
    lineString();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonGeometryContext ------------------------------------------------------------------

wktParser::PolygonGeometryContext::PolygonGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::PolygonGeometryContext::POLYGON() {
  return getToken(wktParser::POLYGON, 0);
}

wktParser::PolygonContext* wktParser::PolygonGeometryContext::polygon() {
  return getRuleContext<wktParser::PolygonContext>(0);
}


size_t wktParser::PolygonGeometryContext::getRuleIndex() const {
  return wktParser::RulePolygonGeometry;
}

void wktParser::PolygonGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPolygonGeometry(this);
}

void wktParser::PolygonGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPolygonGeometry(this);
}

wktParser::PolygonGeometryContext* wktParser::polygonGeometry() {
  PolygonGeometryContext *_localctx = _tracker.createInstance<PolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 6, wktParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(53);
    match(wktParser::POLYGON);
    setState(54);
    polygon();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPointGeometryContext ------------------------------------------------------------------

wktParser::MultiPointGeometryContext::MultiPointGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::MULTIPOINT() {
  return getToken(wktParser::MULTIPOINT, 0);
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointOrClosedPointContext *> wktParser::MultiPointGeometryContext::pointOrClosedPoint() {
  return getRuleContexts<wktParser::PointOrClosedPointContext>();
}

wktParser::PointOrClosedPointContext* wktParser::MultiPointGeometryContext::pointOrClosedPoint(size_t i) {
  return getRuleContext<wktParser::PointOrClosedPointContext>(i);
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::MultiPointGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiPointGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiPointGeometry;
}

void wktParser::MultiPointGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiPointGeometry(this);
}

void wktParser::MultiPointGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiPointGeometry(this);
}

wktParser::MultiPointGeometryContext* wktParser::multiPointGeometry() {
  MultiPointGeometryContext *_localctx = _tracker.createInstance<MultiPointGeometryContext>(_ctx, getState());
  enterRule(_localctx, 8, wktParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(56);
    match(wktParser::MULTIPOINT);
    setState(57);
    match(wktParser::LPAR);
    setState(58);
    pointOrClosedPoint();
    setState(63);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(59);
      match(wktParser::COMMA);
      setState(60);
      pointOrClosedPoint();
      setState(65);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(66);
    match(wktParser::RPAR);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiLineStringGeometryContext ------------------------------------------------------------------

wktParser::MultiLineStringGeometryContext::MultiLineStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::MULTILINESTRING() {
  return getToken(wktParser::MULTILINESTRING, 0);
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::LineStringContext *> wktParser::MultiLineStringGeometryContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::MultiLineStringGeometryContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::MultiLineStringGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiLineStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiLineStringGeometry;
}

void wktParser::MultiLineStringGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiLineStringGeometry(this);
}

void wktParser::MultiLineStringGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiLineStringGeometry(this);
}

wktParser::MultiLineStringGeometryContext* wktParser::multiLineStringGeometry() {
  MultiLineStringGeometryContext *_localctx = _tracker.createInstance<MultiLineStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 10, wktParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(68);
    match(wktParser::MULTILINESTRING);
    setState(69);
    match(wktParser::LPAR);
    setState(70);
    lineString();
    setState(75);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(71);
      match(wktParser::COMMA);
      setState(72);
      lineString();
      setState(77);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(78);
    match(wktParser::RPAR);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiPolygonGeometryContext ------------------------------------------------------------------

wktParser::MultiPolygonGeometryContext::MultiPolygonGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiPolygonGeometryContext::MULTIPOLYGON() {
  return getToken(wktParser::MULTIPOLYGON, 0);
}

tree::TerminalNode* wktParser::MultiPolygonGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::MultiPolygonGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PolygonContext *> wktParser::MultiPolygonGeometryContext::polygon() {
  return getRuleContexts<wktParser::PolygonContext>();
}

wktParser::PolygonContext* wktParser::MultiPolygonGeometryContext::polygon(size_t i) {
  return getRuleContext<wktParser::PolygonContext>(i);
}

tree::TerminalNode* wktParser::MultiPolygonGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::MultiPolygonGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiPolygonGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiPolygonGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiPolygonGeometry;
}

void wktParser::MultiPolygonGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiPolygonGeometry(this);
}

void wktParser::MultiPolygonGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiPolygonGeometry(this);
}

wktParser::MultiPolygonGeometryContext* wktParser::multiPolygonGeometry() {
  MultiPolygonGeometryContext *_localctx = _tracker.createInstance<MultiPolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 12, wktParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(80);
    match(wktParser::MULTIPOLYGON);
    setState(93);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(81);
        match(wktParser::LPAR);
        setState(82);
        polygon();
        setState(87);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(83);
          match(wktParser::COMMA);
          setState(84);
          polygon();
          setState(89);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(90);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(92);
        match(wktParser::EMPTY);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CircularStringGeometryContext ------------------------------------------------------------------

wktParser::CircularStringGeometryContext::CircularStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::CircularStringGeometryContext::CIRCULARSTRING() {
  return getToken(wktParser::CIRCULARSTRING, 0);
}

tree::TerminalNode* wktParser::CircularStringGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointContext *> wktParser::CircularStringGeometryContext::point() {
  return getRuleContexts<wktParser::PointContext>();
}

wktParser::PointContext* wktParser::CircularStringGeometryContext::point(size_t i) {
  return getRuleContext<wktParser::PointContext>(i);
}

tree::TerminalNode* wktParser::CircularStringGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::CircularStringGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::CircularStringGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::CircularStringGeometryContext::getRuleIndex() const {
  return wktParser::RuleCircularStringGeometry;
}

void wktParser::CircularStringGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCircularStringGeometry(this);
}

void wktParser::CircularStringGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCircularStringGeometry(this);
}

wktParser::CircularStringGeometryContext* wktParser::circularStringGeometry() {
  CircularStringGeometryContext *_localctx = _tracker.createInstance<CircularStringGeometryContext>(_ctx, getState());
  enterRule(_localctx, 14, wktParser::RuleCircularStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(95);
    match(wktParser::CIRCULARSTRING);
    setState(96);
    match(wktParser::LPAR);
    setState(97);
    point();
    setState(102);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(98);
      match(wktParser::COMMA);
      setState(99);
      point();
      setState(104);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(105);
    match(wktParser::RPAR);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointOrClosedPointContext ------------------------------------------------------------------

wktParser::PointOrClosedPointContext::PointOrClosedPointContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

wktParser::PointContext* wktParser::PointOrClosedPointContext::point() {
  return getRuleContext<wktParser::PointContext>(0);
}

tree::TerminalNode* wktParser::PointOrClosedPointContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

tree::TerminalNode* wktParser::PointOrClosedPointContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}


size_t wktParser::PointOrClosedPointContext::getRuleIndex() const {
  return wktParser::RulePointOrClosedPoint;
}

void wktParser::PointOrClosedPointContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPointOrClosedPoint(this);
}

void wktParser::PointOrClosedPointContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPointOrClosedPoint(this);
}

wktParser::PointOrClosedPointContext* wktParser::pointOrClosedPoint() {
  PointOrClosedPointContext *_localctx = _tracker.createInstance<PointOrClosedPointContext>(_ctx, getState());
  enterRule(_localctx, 16, wktParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(112);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::DECIMAL: {
        enterOuterAlt(_localctx, 1);
        setState(107);
        point();
        break;
      }

      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 2);
        setState(108);
        match(wktParser::LPAR);
        setState(109);
        point();
        setState(110);
        match(wktParser::RPAR);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PolygonContext ------------------------------------------------------------------

wktParser::PolygonContext::PolygonContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::PolygonContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::LineStringContext *> wktParser::PolygonContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::PolygonContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

tree::TerminalNode* wktParser::PolygonContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::PolygonContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::PolygonContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::PolygonContext::getRuleIndex() const {
  return wktParser::RulePolygon;
}

void wktParser::PolygonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPolygon(this);
}

void wktParser::PolygonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPolygon(this);
}

wktParser::PolygonContext* wktParser::polygon() {
  PolygonContext *_localctx = _tracker.createInstance<PolygonContext>(_ctx, getState());
  enterRule(_localctx, 18, wktParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    match(wktParser::LPAR);
    setState(115);
    lineString();
    setState(120);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(116);
      match(wktParser::COMMA);
      setState(117);
      lineString();
      setState(122);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(123);
    match(wktParser::RPAR);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LineStringContext ------------------------------------------------------------------

wktParser::LineStringContext::LineStringContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::LineStringContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PointContext *> wktParser::LineStringContext::point() {
  return getRuleContexts<wktParser::PointContext>();
}

wktParser::PointContext* wktParser::LineStringContext::point(size_t i) {
  return getRuleContext<wktParser::PointContext>(i);
}

tree::TerminalNode* wktParser::LineStringContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::LineStringContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::LineStringContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::LineStringContext::getRuleIndex() const {
  return wktParser::RuleLineString;
}

void wktParser::LineStringContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLineString(this);
}

void wktParser::LineStringContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLineString(this);
}

wktParser::LineStringContext* wktParser::lineString() {
  LineStringContext *_localctx = _tracker.createInstance<LineStringContext>(_ctx, getState());
  enterRule(_localctx, 20, wktParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(125);
    match(wktParser::LPAR);
    setState(126);
    point();
    setState(131);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(127);
      match(wktParser::COMMA);
      setState(128);
      point();
      setState(133);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(134);
    match(wktParser::RPAR);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PointContext ------------------------------------------------------------------

wktParser::PointContext::PointContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> wktParser::PointContext::DECIMAL() {
  return getTokens(wktParser::DECIMAL);
}

tree::TerminalNode* wktParser::PointContext::DECIMAL(size_t i) {
  return getToken(wktParser::DECIMAL, i);
}


size_t wktParser::PointContext::getRuleIndex() const {
  return wktParser::RulePoint;
}

void wktParser::PointContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPoint(this);
}

void wktParser::PointContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPoint(this);
}

wktParser::PointContext* wktParser::point() {
  PointContext *_localctx = _tracker.createInstance<PointContext>(_ctx, getState());
  enterRule(_localctx, 22, wktParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(137); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(136);
      match(wktParser::DECIMAL);
      setState(139); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == wktParser::DECIMAL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NameContext ------------------------------------------------------------------

wktParser::NameContext::NameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::NameContext::STRING() {
  return getToken(wktParser::STRING, 0);
}


size_t wktParser::NameContext::getRuleIndex() const {
  return wktParser::RuleName;
}

void wktParser::NameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterName(this);
}

void wktParser::NameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitName(this);
}

wktParser::NameContext* wktParser::name() {
  NameContext *_localctx = _tracker.createInstance<NameContext>(_ctx, getState());
  enterRule(_localctx, 24, wktParser::RuleName);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(141);
    match(wktParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> wktParser::_decisionToDFA;
atn::PredictionContextCache wktParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN wktParser::_atn;
std::vector<uint16_t> wktParser::_serializedATN;

std::vector<std::string> wktParser::_ruleNames = {
  "geometry", "pointGeometry", "lineStringGeometry", "polygonGeometry", 
  "multiPointGeometry", "multiLineStringGeometry", "multiPolygonGeometry", 
  "circularStringGeometry", "pointOrClosedPoint", "polygon", "lineString", 
  "point", "name"
};

std::vector<std::string> wktParser::_literalNames = {
  "", "", "", "", "','", "'('", "')'"
};

std::vector<std::string> wktParser::_symbolicNames = {
  "", "DECIMAL", "INTEGERPART", "DECIMALPART", "COMMA", "LPAR", "RPAR", 
  "POINT", "LINESTRING", "POLYGON", "MULTIPOINT", "MULTILINESTRING", "MULTIPOLYGON", 
  "GEOMETRYCOLLECTION", "EMPTY", "CIRCULARSTRING", "COMPOUNDCURVE", "CURVEPOLYGON", 
  "MULTICURVE", "TRIANGLE", "TIN", "POLYHEDRALSURFACE", "STRING", "WS"
};

dfa::Vocabulary wktParser::_vocabulary(_literalNames, _symbolicNames);

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
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x19, 0x92, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 0x9, 
    0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 0x4, 
    0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 0x9, 
    0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 0x3, 
    0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x6, 
    0x2, 0x24, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 0x25, 0x3, 0x2, 0x3, 0x2, 0x3, 
    0x3, 0x3, 0x3, 0x5, 0x3, 0x2c, 0xa, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x33, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x7, 0x6, 0x40, 0xa, 0x6, 0xc, 0x6, 0xe, 0x6, 0x43, 0xb, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x7, 0x7, 0x4c, 0xa, 0x7, 0xc, 0x7, 0xe, 0x7, 0x4f, 0xb, 0x7, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x7, 
    0x8, 0x58, 0xa, 0x8, 0xc, 0x8, 0xe, 0x8, 0x5b, 0xb, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x5, 0x8, 0x60, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x7, 0x9, 0x67, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x6a, 
    0xb, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x5, 0xa, 0x73, 0xa, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
    0xb, 0x7, 0xb, 0x79, 0xa, 0xb, 0xc, 0xb, 0xe, 0xb, 0x7c, 0xb, 0xb, 0x3, 
    0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x7, 0xc, 0x84, 
    0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0x87, 0xb, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xd, 0x6, 0xd, 0x8c, 0xa, 0xd, 0xd, 0xd, 0xe, 0xd, 0x8d, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x2, 0x2, 0xf, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 
    0x12, 0x14, 0x16, 0x18, 0x1a, 0x2, 0x2, 0x2, 0x96, 0x2, 0x23, 0x3, 0x2, 
    0x2, 0x2, 0x4, 0x29, 0x3, 0x2, 0x2, 0x2, 0x6, 0x34, 0x3, 0x2, 0x2, 0x2, 
    0x8, 0x37, 0x3, 0x2, 0x2, 0x2, 0xa, 0x3a, 0x3, 0x2, 0x2, 0x2, 0xc, 0x46, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x52, 0x3, 0x2, 0x2, 0x2, 0x10, 0x61, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0x72, 0x3, 0x2, 0x2, 0x2, 0x14, 0x74, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x18, 0x8b, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x24, 0x5, 0x8, 0x5, 0x2, 
    0x1d, 0x24, 0x5, 0x6, 0x4, 0x2, 0x1e, 0x24, 0x5, 0x4, 0x3, 0x2, 0x1f, 
    0x24, 0x5, 0xa, 0x6, 0x2, 0x20, 0x24, 0x5, 0xc, 0x7, 0x2, 0x21, 0x24, 
    0x5, 0xe, 0x8, 0x2, 0x22, 0x24, 0x5, 0x10, 0x9, 0x2, 0x23, 0x1c, 0x3, 
    0x2, 0x2, 0x2, 0x23, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x23, 0x1e, 0x3, 0x2, 
    0x2, 0x2, 0x23, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x23, 0x20, 0x3, 0x2, 0x2, 
    0x2, 0x23, 0x21, 0x3, 0x2, 0x2, 0x2, 0x23, 0x22, 0x3, 0x2, 0x2, 0x2, 
    0x24, 0x25, 0x3, 0x2, 0x2, 0x2, 0x25, 0x23, 0x3, 0x2, 0x2, 0x2, 0x25, 
    0x26, 0x3, 0x2, 0x2, 0x2, 0x26, 0x27, 0x3, 0x2, 0x2, 0x2, 0x27, 0x28, 
    0x7, 0x2, 0x2, 0x3, 0x28, 0x3, 0x3, 0x2, 0x2, 0x2, 0x29, 0x32, 0x7, 
    0x9, 0x2, 0x2, 0x2a, 0x2c, 0x5, 0x1a, 0xe, 0x2, 0x2b, 0x2a, 0x3, 0x2, 
    0x2, 0x2, 0x2b, 0x2c, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x2d, 0x3, 0x2, 0x2, 
    0x2, 0x2d, 0x2e, 0x7, 0x7, 0x2, 0x2, 0x2e, 0x2f, 0x5, 0x18, 0xd, 0x2, 
    0x2f, 0x30, 0x7, 0x8, 0x2, 0x2, 0x30, 0x33, 0x3, 0x2, 0x2, 0x2, 0x31, 
    0x33, 0x7, 0x10, 0x2, 0x2, 0x32, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x32, 0x31, 
    0x3, 0x2, 0x2, 0x2, 0x33, 0x5, 0x3, 0x2, 0x2, 0x2, 0x34, 0x35, 0x7, 
    0xa, 0x2, 0x2, 0x35, 0x36, 0x5, 0x16, 0xc, 0x2, 0x36, 0x7, 0x3, 0x2, 
    0x2, 0x2, 0x37, 0x38, 0x7, 0xb, 0x2, 0x2, 0x38, 0x39, 0x5, 0x14, 0xb, 
    0x2, 0x39, 0x9, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x3b, 0x7, 0xc, 0x2, 0x2, 
    0x3b, 0x3c, 0x7, 0x7, 0x2, 0x2, 0x3c, 0x41, 0x5, 0x12, 0xa, 0x2, 0x3d, 
    0x3e, 0x7, 0x6, 0x2, 0x2, 0x3e, 0x40, 0x5, 0x12, 0xa, 0x2, 0x3f, 0x3d, 
    0x3, 0x2, 0x2, 0x2, 0x40, 0x43, 0x3, 0x2, 0x2, 0x2, 0x41, 0x3f, 0x3, 
    0x2, 0x2, 0x2, 0x41, 0x42, 0x3, 0x2, 0x2, 0x2, 0x42, 0x44, 0x3, 0x2, 
    0x2, 0x2, 0x43, 0x41, 0x3, 0x2, 0x2, 0x2, 0x44, 0x45, 0x7, 0x8, 0x2, 
    0x2, 0x45, 0xb, 0x3, 0x2, 0x2, 0x2, 0x46, 0x47, 0x7, 0xd, 0x2, 0x2, 
    0x47, 0x48, 0x7, 0x7, 0x2, 0x2, 0x48, 0x4d, 0x5, 0x16, 0xc, 0x2, 0x49, 
    0x4a, 0x7, 0x6, 0x2, 0x2, 0x4a, 0x4c, 0x5, 0x16, 0xc, 0x2, 0x4b, 0x49, 
    0x3, 0x2, 0x2, 0x2, 0x4c, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x4d, 0x4b, 0x3, 
    0x2, 0x2, 0x2, 0x4d, 0x4e, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x50, 0x3, 0x2, 
    0x2, 0x2, 0x4f, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x50, 0x51, 0x7, 0x8, 0x2, 
    0x2, 0x51, 0xd, 0x3, 0x2, 0x2, 0x2, 0x52, 0x5f, 0x7, 0xe, 0x2, 0x2, 
    0x53, 0x54, 0x7, 0x7, 0x2, 0x2, 0x54, 0x59, 0x5, 0x14, 0xb, 0x2, 0x55, 
    0x56, 0x7, 0x6, 0x2, 0x2, 0x56, 0x58, 0x5, 0x14, 0xb, 0x2, 0x57, 0x55, 
    0x3, 0x2, 0x2, 0x2, 0x58, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x59, 0x57, 0x3, 
    0x2, 0x2, 0x2, 0x59, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5c, 0x3, 0x2, 
    0x2, 0x2, 0x5b, 0x59, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5d, 0x7, 0x8, 0x2, 
    0x2, 0x5d, 0x60, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x60, 0x7, 0x10, 0x2, 0x2, 
    0x5f, 0x53, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5e, 0x3, 0x2, 0x2, 0x2, 0x60, 
    0xf, 0x3, 0x2, 0x2, 0x2, 0x61, 0x62, 0x7, 0x11, 0x2, 0x2, 0x62, 0x63, 
    0x7, 0x7, 0x2, 0x2, 0x63, 0x68, 0x5, 0x18, 0xd, 0x2, 0x64, 0x65, 0x7, 
    0x6, 0x2, 0x2, 0x65, 0x67, 0x5, 0x18, 0xd, 0x2, 0x66, 0x64, 0x3, 0x2, 
    0x2, 0x2, 0x67, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x68, 0x66, 0x3, 0x2, 0x2, 
    0x2, 0x68, 0x69, 0x3, 0x2, 0x2, 0x2, 0x69, 0x6b, 0x3, 0x2, 0x2, 0x2, 
    0x6a, 0x68, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x6c, 0x7, 0x8, 0x2, 0x2, 0x6c, 
    0x11, 0x3, 0x2, 0x2, 0x2, 0x6d, 0x73, 0x5, 0x18, 0xd, 0x2, 0x6e, 0x6f, 
    0x7, 0x7, 0x2, 0x2, 0x6f, 0x70, 0x5, 0x18, 0xd, 0x2, 0x70, 0x71, 0x7, 
    0x8, 0x2, 0x2, 0x71, 0x73, 0x3, 0x2, 0x2, 0x2, 0x72, 0x6d, 0x3, 0x2, 
    0x2, 0x2, 0x72, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x73, 0x13, 0x3, 0x2, 0x2, 
    0x2, 0x74, 0x75, 0x7, 0x7, 0x2, 0x2, 0x75, 0x7a, 0x5, 0x16, 0xc, 0x2, 
    0x76, 0x77, 0x7, 0x6, 0x2, 0x2, 0x77, 0x79, 0x5, 0x16, 0xc, 0x2, 0x78, 
    0x76, 0x3, 0x2, 0x2, 0x2, 0x79, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x78, 
    0x3, 0x2, 0x2, 0x2, 0x7a, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x7b, 0x7d, 0x3, 
    0x2, 0x2, 0x2, 0x7c, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7e, 0x7, 0x8, 
    0x2, 0x2, 0x7e, 0x15, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x80, 0x7, 0x7, 0x2, 
    0x2, 0x80, 0x85, 0x5, 0x18, 0xd, 0x2, 0x81, 0x82, 0x7, 0x6, 0x2, 0x2, 
    0x82, 0x84, 0x5, 0x18, 0xd, 0x2, 0x83, 0x81, 0x3, 0x2, 0x2, 0x2, 0x84, 
    0x87, 0x3, 0x2, 0x2, 0x2, 0x85, 0x83, 0x3, 0x2, 0x2, 0x2, 0x85, 0x86, 
    0x3, 0x2, 0x2, 0x2, 0x86, 0x88, 0x3, 0x2, 0x2, 0x2, 0x87, 0x85, 0x3, 
    0x2, 0x2, 0x2, 0x88, 0x89, 0x7, 0x8, 0x2, 0x2, 0x89, 0x17, 0x3, 0x2, 
    0x2, 0x2, 0x8a, 0x8c, 0x7, 0x3, 0x2, 0x2, 0x8b, 0x8a, 0x3, 0x2, 0x2, 
    0x2, 0x8c, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8b, 0x3, 0x2, 0x2, 0x2, 
    0x8d, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x19, 0x3, 0x2, 0x2, 0x2, 0x8f, 
    0x90, 0x7, 0x18, 0x2, 0x2, 0x90, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xf, 0x23, 
    0x25, 0x2b, 0x32, 0x41, 0x4d, 0x59, 0x5f, 0x68, 0x72, 0x7a, 0x85, 0x8d, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

wktParser::Initializer wktParser::_init;
