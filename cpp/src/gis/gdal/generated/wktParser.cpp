
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


//----------------- GeometryCollectionContext ------------------------------------------------------------------

wktParser::GeometryCollectionContext::GeometryCollectionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::GeometryCollectionContext::GEOMETRYCOLLECTION() {
  return getToken(wktParser::GEOMETRYCOLLECTION, 0);
}

tree::TerminalNode* wktParser::GeometryCollectionContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::GeometryContext *> wktParser::GeometryCollectionContext::geometry() {
  return getRuleContexts<wktParser::GeometryContext>();
}

wktParser::GeometryContext* wktParser::GeometryCollectionContext::geometry(size_t i) {
  return getRuleContext<wktParser::GeometryContext>(i);
}

tree::TerminalNode* wktParser::GeometryCollectionContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

tree::TerminalNode* wktParser::GeometryCollectionContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

std::vector<tree::TerminalNode *> wktParser::GeometryCollectionContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::GeometryCollectionContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::GeometryCollectionContext::getRuleIndex() const {
  return wktParser::RuleGeometryCollection;
}

void wktParser::GeometryCollectionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGeometryCollection(this);
}

void wktParser::GeometryCollectionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGeometryCollection(this);
}

wktParser::GeometryCollectionContext* wktParser::geometryCollection() {
  GeometryCollectionContext *_localctx = _tracker.createInstance<GeometryCollectionContext>(_ctx, getState());
  enterRule(_localctx, 0, wktParser::RuleGeometryCollection);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(40);
    match(wktParser::GEOMETRYCOLLECTION);
    setState(53);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(41);
        match(wktParser::LPAR);
        setState(42);
        geometry();
        setState(47);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(43);
          match(wktParser::COMMA);
          setState(44);
          geometry();
          setState(49);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(50);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(52);
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

//----------------- GeometryContext ------------------------------------------------------------------

wktParser::GeometryContext::GeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

wktParser::PolygonGeometryContext* wktParser::GeometryContext::polygonGeometry() {
  return getRuleContext<wktParser::PolygonGeometryContext>(0);
}

wktParser::LineStringGeometryContext* wktParser::GeometryContext::lineStringGeometry() {
  return getRuleContext<wktParser::LineStringGeometryContext>(0);
}

wktParser::PointGeometryContext* wktParser::GeometryContext::pointGeometry() {
  return getRuleContext<wktParser::PointGeometryContext>(0);
}

wktParser::CompoundCurveGeometryContext* wktParser::GeometryContext::compoundCurveGeometry() {
  return getRuleContext<wktParser::CompoundCurveGeometryContext>(0);
}

wktParser::CurvePolygonGeometryContext* wktParser::GeometryContext::curvePolygonGeometry() {
  return getRuleContext<wktParser::CurvePolygonGeometryContext>(0);
}

wktParser::MultiSurfaceGeometryContext* wktParser::GeometryContext::multiSurfaceGeometry() {
  return getRuleContext<wktParser::MultiSurfaceGeometryContext>(0);
}

wktParser::MultiCurveGeometryContext* wktParser::GeometryContext::multiCurveGeometry() {
  return getRuleContext<wktParser::MultiCurveGeometryContext>(0);
}

wktParser::MultiPointGeometryContext* wktParser::GeometryContext::multiPointGeometry() {
  return getRuleContext<wktParser::MultiPointGeometryContext>(0);
}

wktParser::MultiLineStringGeometryContext* wktParser::GeometryContext::multiLineStringGeometry() {
  return getRuleContext<wktParser::MultiLineStringGeometryContext>(0);
}

wktParser::MultiPolygonGeometryContext* wktParser::GeometryContext::multiPolygonGeometry() {
  return getRuleContext<wktParser::MultiPolygonGeometryContext>(0);
}

wktParser::CircularStringGeometryContext* wktParser::GeometryContext::circularStringGeometry() {
  return getRuleContext<wktParser::CircularStringGeometryContext>(0);
}

wktParser::MultiPolyhedralSurfaceGeometryContext* wktParser::GeometryContext::multiPolyhedralSurfaceGeometry() {
  return getRuleContext<wktParser::MultiPolyhedralSurfaceGeometryContext>(0);
}

wktParser::MultiTinGeometryContext* wktParser::GeometryContext::multiTinGeometry() {
  return getRuleContext<wktParser::MultiTinGeometryContext>(0);
}

wktParser::GeometryCollectionContext* wktParser::GeometryContext::geometryCollection() {
  return getRuleContext<wktParser::GeometryCollectionContext>(0);
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
  enterRule(_localctx, 2, wktParser::RuleGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(69);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::POLYGON: {
        setState(55);
        polygonGeometry();
        break;
      }

      case wktParser::LINESTRING: {
        setState(56);
        lineStringGeometry();
        break;
      }

      case wktParser::POINT: {
        setState(57);
        pointGeometry();
        break;
      }

      case wktParser::COMPOUNDCURVE: {
        setState(58);
        compoundCurveGeometry();
        break;
      }

      case wktParser::CURVEPOLYGON: {
        setState(59);
        curvePolygonGeometry();
        break;
      }

      case wktParser::MULTISURFACE: {
        setState(60);
        multiSurfaceGeometry();
        break;
      }

      case wktParser::MULTICURVE: {
        setState(61);
        multiCurveGeometry();
        break;
      }

      case wktParser::MULTIPOINT: {
        setState(62);
        multiPointGeometry();
        break;
      }

      case wktParser::MULTILINESTRING: {
        setState(63);
        multiLineStringGeometry();
        break;
      }

      case wktParser::MULTIPOLYGON: {
        setState(64);
        multiPolygonGeometry();
        break;
      }

      case wktParser::CIRCULARSTRING: {
        setState(65);
        circularStringGeometry();
        break;
      }

      case wktParser::POLYHEDRALSURFACE: {
        setState(66);
        multiPolyhedralSurfaceGeometry();
        break;
      }

      case wktParser::TIN: {
        setState(67);
        multiTinGeometry();
        break;
      }

      case wktParser::GEOMETRYCOLLECTION: {
        setState(68);
        geometryCollection();
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
  enterRule(_localctx, 4, wktParser::RulePointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(71);
    match(wktParser::POINT);
    setState(80);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR:
      case wktParser::STRING: {
        setState(73);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == wktParser::STRING) {
          setState(72);
          name();
        }
        setState(75);
        match(wktParser::LPAR);
        setState(76);
        point();
        setState(77);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(79);
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
  enterRule(_localctx, 6, wktParser::RuleLineStringGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(82);
    match(wktParser::LINESTRING);
    setState(83);
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
  enterRule(_localctx, 8, wktParser::RulePolygonGeometry);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(85);
    match(wktParser::POLYGON);
    setState(86);
    polygon();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiCurveGeometryContext ------------------------------------------------------------------

wktParser::MultiCurveGeometryContext::MultiCurveGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiCurveGeometryContext::MULTICURVE() {
  return getToken(wktParser::MULTICURVE, 0);
}

tree::TerminalNode* wktParser::MultiCurveGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::MultiCurveGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

tree::TerminalNode* wktParser::MultiCurveGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<wktParser::LineStringContext *> wktParser::MultiCurveGeometryContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::MultiCurveGeometryContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

std::vector<wktParser::CircularStringGeometryContext *> wktParser::MultiCurveGeometryContext::circularStringGeometry() {
  return getRuleContexts<wktParser::CircularStringGeometryContext>();
}

wktParser::CircularStringGeometryContext* wktParser::MultiCurveGeometryContext::circularStringGeometry(size_t i) {
  return getRuleContext<wktParser::CircularStringGeometryContext>(i);
}

std::vector<wktParser::CompoundCurveGeometryContext *> wktParser::MultiCurveGeometryContext::compoundCurveGeometry() {
  return getRuleContexts<wktParser::CompoundCurveGeometryContext>();
}

wktParser::CompoundCurveGeometryContext* wktParser::MultiCurveGeometryContext::compoundCurveGeometry(size_t i) {
  return getRuleContext<wktParser::CompoundCurveGeometryContext>(i);
}

std::vector<tree::TerminalNode *> wktParser::MultiCurveGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiCurveGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiCurveGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiCurveGeometry;
}

void wktParser::MultiCurveGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiCurveGeometry(this);
}

void wktParser::MultiCurveGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiCurveGeometry(this);
}

wktParser::MultiCurveGeometryContext* wktParser::multiCurveGeometry() {
  MultiCurveGeometryContext *_localctx = _tracker.createInstance<MultiCurveGeometryContext>(_ctx, getState());
  enterRule(_localctx, 10, wktParser::RuleMultiCurveGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(88);
    match(wktParser::MULTICURVE);
    setState(109);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(89);
        match(wktParser::LPAR);
        setState(93);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case wktParser::LPAR:
          case wktParser::EMPTY: {
            setState(90);
            lineString();
            break;
          }

          case wktParser::CIRCULARSTRING: {
            setState(91);
            circularStringGeometry();
            break;
          }

          case wktParser::COMPOUNDCURVE: {
            setState(92);
            compoundCurveGeometry();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(103);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(95);
          match(wktParser::COMMA);
          setState(99);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case wktParser::CIRCULARSTRING: {
              setState(96);
              circularStringGeometry();
              break;
            }

            case wktParser::LPAR:
            case wktParser::EMPTY: {
              setState(97);
              lineString();
              break;
            }

            case wktParser::COMPOUNDCURVE: {
              setState(98);
              compoundCurveGeometry();
              break;
            }

          default:
            throw NoViableAltException(this);
          }
          setState(105);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(106);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(108);
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

//----------------- MultiSurfaceGeometryContext ------------------------------------------------------------------

wktParser::MultiSurfaceGeometryContext::MultiSurfaceGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiSurfaceGeometryContext::MULTISURFACE() {
  return getToken(wktParser::MULTISURFACE, 0);
}

tree::TerminalNode* wktParser::MultiSurfaceGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::MultiSurfaceGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

tree::TerminalNode* wktParser::MultiSurfaceGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<wktParser::PolygonContext *> wktParser::MultiSurfaceGeometryContext::polygon() {
  return getRuleContexts<wktParser::PolygonContext>();
}

wktParser::PolygonContext* wktParser::MultiSurfaceGeometryContext::polygon(size_t i) {
  return getRuleContext<wktParser::PolygonContext>(i);
}

std::vector<wktParser::CurvePolygonGeometryContext *> wktParser::MultiSurfaceGeometryContext::curvePolygonGeometry() {
  return getRuleContexts<wktParser::CurvePolygonGeometryContext>();
}

wktParser::CurvePolygonGeometryContext* wktParser::MultiSurfaceGeometryContext::curvePolygonGeometry(size_t i) {
  return getRuleContext<wktParser::CurvePolygonGeometryContext>(i);
}

std::vector<tree::TerminalNode *> wktParser::MultiSurfaceGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiSurfaceGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiSurfaceGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiSurfaceGeometry;
}

void wktParser::MultiSurfaceGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiSurfaceGeometry(this);
}

void wktParser::MultiSurfaceGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiSurfaceGeometry(this);
}

wktParser::MultiSurfaceGeometryContext* wktParser::multiSurfaceGeometry() {
  MultiSurfaceGeometryContext *_localctx = _tracker.createInstance<MultiSurfaceGeometryContext>(_ctx, getState());
  enterRule(_localctx, 12, wktParser::RuleMultiSurfaceGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(111);
    match(wktParser::MULTISURFACE);
    setState(130);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(112);
        match(wktParser::LPAR);
        setState(115);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case wktParser::LPAR:
          case wktParser::EMPTY: {
            setState(113);
            polygon();
            break;
          }

          case wktParser::CURVEPOLYGON: {
            setState(114);
            curvePolygonGeometry();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(124);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(117);
          match(wktParser::COMMA);
          setState(120);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case wktParser::LPAR:
            case wktParser::EMPTY: {
              setState(118);
              polygon();
              break;
            }

            case wktParser::CURVEPOLYGON: {
              setState(119);
              curvePolygonGeometry();
              break;
            }

          default:
            throw NoViableAltException(this);
          }
          setState(126);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(127);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(129);
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

//----------------- CurvePolygonGeometryContext ------------------------------------------------------------------

wktParser::CurvePolygonGeometryContext::CurvePolygonGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::CurvePolygonGeometryContext::CURVEPOLYGON() {
  return getToken(wktParser::CURVEPOLYGON, 0);
}

tree::TerminalNode* wktParser::CurvePolygonGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::CurvePolygonGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

tree::TerminalNode* wktParser::CurvePolygonGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<wktParser::LineStringContext *> wktParser::CurvePolygonGeometryContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::CurvePolygonGeometryContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

std::vector<wktParser::CircularStringGeometryContext *> wktParser::CurvePolygonGeometryContext::circularStringGeometry() {
  return getRuleContexts<wktParser::CircularStringGeometryContext>();
}

wktParser::CircularStringGeometryContext* wktParser::CurvePolygonGeometryContext::circularStringGeometry(size_t i) {
  return getRuleContext<wktParser::CircularStringGeometryContext>(i);
}

std::vector<wktParser::CompoundCurveGeometryContext *> wktParser::CurvePolygonGeometryContext::compoundCurveGeometry() {
  return getRuleContexts<wktParser::CompoundCurveGeometryContext>();
}

wktParser::CompoundCurveGeometryContext* wktParser::CurvePolygonGeometryContext::compoundCurveGeometry(size_t i) {
  return getRuleContext<wktParser::CompoundCurveGeometryContext>(i);
}

std::vector<tree::TerminalNode *> wktParser::CurvePolygonGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::CurvePolygonGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::CurvePolygonGeometryContext::getRuleIndex() const {
  return wktParser::RuleCurvePolygonGeometry;
}

void wktParser::CurvePolygonGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCurvePolygonGeometry(this);
}

void wktParser::CurvePolygonGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCurvePolygonGeometry(this);
}

wktParser::CurvePolygonGeometryContext* wktParser::curvePolygonGeometry() {
  CurvePolygonGeometryContext *_localctx = _tracker.createInstance<CurvePolygonGeometryContext>(_ctx, getState());
  enterRule(_localctx, 14, wktParser::RuleCurvePolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(132);
    match(wktParser::CURVEPOLYGON);
    setState(153);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(133);
        match(wktParser::LPAR);
        setState(137);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case wktParser::LPAR:
          case wktParser::EMPTY: {
            setState(134);
            lineString();
            break;
          }

          case wktParser::CIRCULARSTRING: {
            setState(135);
            circularStringGeometry();
            break;
          }

          case wktParser::COMPOUNDCURVE: {
            setState(136);
            compoundCurveGeometry();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(147);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(139);
          match(wktParser::COMMA);
          setState(143);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case wktParser::CIRCULARSTRING: {
              setState(140);
              circularStringGeometry();
              break;
            }

            case wktParser::LPAR:
            case wktParser::EMPTY: {
              setState(141);
              lineString();
              break;
            }

            case wktParser::COMPOUNDCURVE: {
              setState(142);
              compoundCurveGeometry();
              break;
            }

          default:
            throw NoViableAltException(this);
          }
          setState(149);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(150);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(152);
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

//----------------- CompoundCurveGeometryContext ------------------------------------------------------------------

wktParser::CompoundCurveGeometryContext::CompoundCurveGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::CompoundCurveGeometryContext::COMPOUNDCURVE() {
  return getToken(wktParser::COMPOUNDCURVE, 0);
}

tree::TerminalNode* wktParser::CompoundCurveGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::CompoundCurveGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

tree::TerminalNode* wktParser::CompoundCurveGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<wktParser::LineStringContext *> wktParser::CompoundCurveGeometryContext::lineString() {
  return getRuleContexts<wktParser::LineStringContext>();
}

wktParser::LineStringContext* wktParser::CompoundCurveGeometryContext::lineString(size_t i) {
  return getRuleContext<wktParser::LineStringContext>(i);
}

std::vector<wktParser::CircularStringGeometryContext *> wktParser::CompoundCurveGeometryContext::circularStringGeometry() {
  return getRuleContexts<wktParser::CircularStringGeometryContext>();
}

wktParser::CircularStringGeometryContext* wktParser::CompoundCurveGeometryContext::circularStringGeometry(size_t i) {
  return getRuleContext<wktParser::CircularStringGeometryContext>(i);
}

std::vector<tree::TerminalNode *> wktParser::CompoundCurveGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::CompoundCurveGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::CompoundCurveGeometryContext::getRuleIndex() const {
  return wktParser::RuleCompoundCurveGeometry;
}

void wktParser::CompoundCurveGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompoundCurveGeometry(this);
}

void wktParser::CompoundCurveGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompoundCurveGeometry(this);
}

wktParser::CompoundCurveGeometryContext* wktParser::compoundCurveGeometry() {
  CompoundCurveGeometryContext *_localctx = _tracker.createInstance<CompoundCurveGeometryContext>(_ctx, getState());
  enterRule(_localctx, 16, wktParser::RuleCompoundCurveGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(155);
    match(wktParser::COMPOUNDCURVE);
    setState(174);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(156);
        match(wktParser::LPAR);
        setState(159);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case wktParser::LPAR:
          case wktParser::EMPTY: {
            setState(157);
            lineString();
            break;
          }

          case wktParser::CIRCULARSTRING: {
            setState(158);
            circularStringGeometry();
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        setState(168);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(161);
          match(wktParser::COMMA);
          setState(164);
          _errHandler->sync(this);
          switch (_input->LA(1)) {
            case wktParser::CIRCULARSTRING: {
              setState(162);
              circularStringGeometry();
              break;
            }

            case wktParser::LPAR:
            case wktParser::EMPTY: {
              setState(163);
              lineString();
              break;
            }

          default:
            throw NoViableAltException(this);
          }
          setState(170);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(171);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(173);
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

//----------------- MultiPointGeometryContext ------------------------------------------------------------------

wktParser::MultiPointGeometryContext::MultiPointGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::MULTIPOINT() {
  return getToken(wktParser::MULTIPOINT, 0);
}

tree::TerminalNode* wktParser::MultiPointGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
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
  enterRule(_localctx, 18, wktParser::RuleMultiPointGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(176);
    match(wktParser::MULTIPOINT);
    setState(189);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(177);
        match(wktParser::LPAR);
        setState(178);
        pointOrClosedPoint();
        setState(183);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(179);
          match(wktParser::COMMA);
          setState(180);
          pointOrClosedPoint();
          setState(185);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(186);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(188);
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

//----------------- MultiLineStringGeometryContext ------------------------------------------------------------------

wktParser::MultiLineStringGeometryContext::MultiLineStringGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::MULTILINESTRING() {
  return getToken(wktParser::MULTILINESTRING, 0);
}

tree::TerminalNode* wktParser::MultiLineStringGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
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
  enterRule(_localctx, 20, wktParser::RuleMultiLineStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(191);
    match(wktParser::MULTILINESTRING);
    setState(204);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(192);
        match(wktParser::LPAR);
        setState(193);
        lineString();
        setState(198);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(194);
          match(wktParser::COMMA);
          setState(195);
          lineString();
          setState(200);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(201);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(203);
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
  enterRule(_localctx, 22, wktParser::RuleMultiPolygonGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(206);
    match(wktParser::MULTIPOLYGON);
    setState(219);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(207);
        match(wktParser::LPAR);
        setState(208);
        polygon();
        setState(213);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(209);
          match(wktParser::COMMA);
          setState(210);
          polygon();
          setState(215);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(216);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(218);
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

//----------------- MultiPolyhedralSurfaceGeometryContext ------------------------------------------------------------------

wktParser::MultiPolyhedralSurfaceGeometryContext::MultiPolyhedralSurfaceGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiPolyhedralSurfaceGeometryContext::POLYHEDRALSURFACE() {
  return getToken(wktParser::POLYHEDRALSURFACE, 0);
}

tree::TerminalNode* wktParser::MultiPolyhedralSurfaceGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::MultiPolyhedralSurfaceGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PolygonContext *> wktParser::MultiPolyhedralSurfaceGeometryContext::polygon() {
  return getRuleContexts<wktParser::PolygonContext>();
}

wktParser::PolygonContext* wktParser::MultiPolyhedralSurfaceGeometryContext::polygon(size_t i) {
  return getRuleContext<wktParser::PolygonContext>(i);
}

tree::TerminalNode* wktParser::MultiPolyhedralSurfaceGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::MultiPolyhedralSurfaceGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiPolyhedralSurfaceGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiPolyhedralSurfaceGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiPolyhedralSurfaceGeometry;
}

void wktParser::MultiPolyhedralSurfaceGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiPolyhedralSurfaceGeometry(this);
}

void wktParser::MultiPolyhedralSurfaceGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiPolyhedralSurfaceGeometry(this);
}

wktParser::MultiPolyhedralSurfaceGeometryContext* wktParser::multiPolyhedralSurfaceGeometry() {
  MultiPolyhedralSurfaceGeometryContext *_localctx = _tracker.createInstance<MultiPolyhedralSurfaceGeometryContext>(_ctx, getState());
  enterRule(_localctx, 24, wktParser::RuleMultiPolyhedralSurfaceGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(221);
    match(wktParser::POLYHEDRALSURFACE);
    setState(234);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(222);
        match(wktParser::LPAR);
        setState(223);
        polygon();
        setState(228);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(224);
          match(wktParser::COMMA);
          setState(225);
          polygon();
          setState(230);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(231);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(233);
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

//----------------- MultiTinGeometryContext ------------------------------------------------------------------

wktParser::MultiTinGeometryContext::MultiTinGeometryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* wktParser::MultiTinGeometryContext::TIN() {
  return getToken(wktParser::TIN, 0);
}

tree::TerminalNode* wktParser::MultiTinGeometryContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
}

tree::TerminalNode* wktParser::MultiTinGeometryContext::LPAR() {
  return getToken(wktParser::LPAR, 0);
}

std::vector<wktParser::PolygonContext *> wktParser::MultiTinGeometryContext::polygon() {
  return getRuleContexts<wktParser::PolygonContext>();
}

wktParser::PolygonContext* wktParser::MultiTinGeometryContext::polygon(size_t i) {
  return getRuleContext<wktParser::PolygonContext>(i);
}

tree::TerminalNode* wktParser::MultiTinGeometryContext::RPAR() {
  return getToken(wktParser::RPAR, 0);
}

std::vector<tree::TerminalNode *> wktParser::MultiTinGeometryContext::COMMA() {
  return getTokens(wktParser::COMMA);
}

tree::TerminalNode* wktParser::MultiTinGeometryContext::COMMA(size_t i) {
  return getToken(wktParser::COMMA, i);
}


size_t wktParser::MultiTinGeometryContext::getRuleIndex() const {
  return wktParser::RuleMultiTinGeometry;
}

void wktParser::MultiTinGeometryContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiTinGeometry(this);
}

void wktParser::MultiTinGeometryContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<wktListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiTinGeometry(this);
}

wktParser::MultiTinGeometryContext* wktParser::multiTinGeometry() {
  MultiTinGeometryContext *_localctx = _tracker.createInstance<MultiTinGeometryContext>(_ctx, getState());
  enterRule(_localctx, 26, wktParser::RuleMultiTinGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(236);
    match(wktParser::TIN);
    setState(249);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        setState(237);
        match(wktParser::LPAR);
        setState(238);
        polygon();
        setState(243);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(239);
          match(wktParser::COMMA);
          setState(240);
          polygon();
          setState(245);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(246);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        setState(248);
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
  enterRule(_localctx, 28, wktParser::RuleCircularStringGeometry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    match(wktParser::CIRCULARSTRING);
    setState(252);
    match(wktParser::LPAR);
    setState(253);
    point();
    setState(258);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == wktParser::COMMA) {
      setState(254);
      match(wktParser::COMMA);
      setState(255);
      point();
      setState(260);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(261);
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
  enterRule(_localctx, 30, wktParser::RulePointOrClosedPoint);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(268);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::DECIMAL: {
        enterOuterAlt(_localctx, 1);
        setState(263);
        point();
        break;
      }

      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 2);
        setState(264);
        match(wktParser::LPAR);
        setState(265);
        point();
        setState(266);
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

tree::TerminalNode* wktParser::PolygonContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
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
  enterRule(_localctx, 32, wktParser::RulePolygon);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(282);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 1);
        setState(270);
        match(wktParser::LPAR);
        setState(271);
        lineString();
        setState(276);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(272);
          match(wktParser::COMMA);
          setState(273);
          lineString();
          setState(278);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(279);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        enterOuterAlt(_localctx, 2);
        setState(281);
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

tree::TerminalNode* wktParser::LineStringContext::EMPTY() {
  return getToken(wktParser::EMPTY, 0);
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
  enterRule(_localctx, 34, wktParser::RuleLineString);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(296);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case wktParser::LPAR: {
        enterOuterAlt(_localctx, 1);
        setState(284);
        match(wktParser::LPAR);
        setState(285);
        point();
        setState(290);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == wktParser::COMMA) {
          setState(286);
          match(wktParser::COMMA);
          setState(287);
          point();
          setState(292);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(293);
        match(wktParser::RPAR);
        break;
      }

      case wktParser::EMPTY: {
        enterOuterAlt(_localctx, 2);
        setState(295);
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
  enterRule(_localctx, 36, wktParser::RulePoint);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(299); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(298);
      match(wktParser::DECIMAL);
      setState(301); 
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
  enterRule(_localctx, 38, wktParser::RuleName);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(303);
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
  "geometryCollection", "geometry", "pointGeometry", "lineStringGeometry", 
  "polygonGeometry", "multiCurveGeometry", "multiSurfaceGeometry", "curvePolygonGeometry", 
  "compoundCurveGeometry", "multiPointGeometry", "multiLineStringGeometry", 
  "multiPolygonGeometry", "multiPolyhedralSurfaceGeometry", "multiTinGeometry", 
  "circularStringGeometry", "pointOrClosedPoint", "polygon", "lineString", 
  "point", "name"
};

std::vector<std::string> wktParser::_literalNames = {
  "", "", "", "", "','", "'('", "')'"
};

std::vector<std::string> wktParser::_symbolicNames = {
  "", "DECIMAL", "INTEGERPART", "DECIMALPART", "COMMA", "LPAR", "RPAR", 
  "POINT", "LINESTRING", "POLYGON", "MULTIPOINT", "MULTILINESTRING", "MULTIPOLYGON", 
  "GEOMETRYCOLLECTION", "EMPTY", "CIRCULARSTRING", "COMPOUNDCURVE", "MULTISURFACE", 
  "CURVEPOLYGON", "MULTICURVE", "TRIANGLE", "TIN", "POLYHEDRALSURFACE", 
  "STRING", "WS"
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
    0x3, 0x1a, 0x134, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x7, 0x2, 
    0x30, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x33, 0xb, 0x2, 0x3, 0x2, 0x3, 0x2, 
    0x3, 0x2, 0x5, 0x2, 0x38, 0xa, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x48, 0xa, 0x3, 0x3, 0x4, 
    0x3, 0x4, 0x5, 0x4, 0x4c, 0xa, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x5, 0x4, 0x53, 0xa, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x5, 0x7, 0x60, 0xa, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x7, 0x5, 0x7, 0x66, 0xa, 0x7, 0x7, 0x7, 0x68, 0xa, 0x7, 0xc, 0x7, 0xe, 
    0x7, 0x6b, 0xb, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x70, 0xa, 
    0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x76, 0xa, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x7b, 0xa, 0x8, 0x7, 0x8, 0x7d, 
    0xa, 0x8, 0xc, 0x8, 0xe, 0x8, 0x80, 0xb, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x5, 0x8, 0x85, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x5, 0x9, 0x8c, 0xa, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x5, 0x9, 0x92, 0xa, 0x9, 0x7, 0x9, 0x94, 0xa, 0x9, 0xc, 0x9, 0xe, 
    0x9, 0x97, 0xb, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0x9c, 0xa, 
    0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xa2, 0xa, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xa7, 0xa, 0xa, 0x7, 0xa, 0xa9, 
    0xa, 0xa, 0xc, 0xa, 0xe, 0xa, 0xac, 0xb, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 
    0xa, 0x5, 0xa, 0xb1, 0xa, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x7, 0xb, 0xb8, 0xa, 0xb, 0xc, 0xb, 0xe, 0xb, 0xbb, 0xb, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0xc0, 0xa, 0xb, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x7, 0xc, 0xc7, 0xa, 0xc, 0xc, 0xc, 
    0xe, 0xc, 0xca, 0xb, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0xcf, 
    0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x7, 0xd, 
    0xd6, 0xa, 0xd, 0xc, 0xd, 0xe, 0xd, 0xd9, 0xb, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x5, 0xd, 0xde, 0xa, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x7, 0xe, 0xe5, 0xa, 0xe, 0xc, 0xe, 0xe, 0xe, 0xe8, 0xb, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0xed, 0xa, 0xe, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x7, 0xf, 0xf4, 0xa, 0xf, 0xc, 
    0xf, 0xe, 0xf, 0xf7, 0xb, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 
    0xfc, 0xa, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x7, 0x10, 0x103, 0xa, 0x10, 0xc, 0x10, 0xe, 0x10, 0x106, 0xb, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x5, 0x11, 0x10f, 0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x7, 0x12, 0x115, 0xa, 0x12, 0xc, 0x12, 0xe, 0x12, 0x118, 
    0xb, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x5, 0x12, 0x11d, 0xa, 0x12, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0x123, 0xa, 0x13, 
    0xc, 0x13, 0xe, 0x13, 0x126, 0xb, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x5, 0x13, 0x12b, 0xa, 0x13, 0x3, 0x14, 0x6, 0x14, 0x12e, 0xa, 0x14, 
    0xd, 0x14, 0xe, 0x14, 0x12f, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x2, 0x2, 
    0x16, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 
    0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2, 0x2, 0x2, 0x155, 
    0x2, 0x2a, 0x3, 0x2, 0x2, 0x2, 0x4, 0x47, 0x3, 0x2, 0x2, 0x2, 0x6, 0x49, 
    0x3, 0x2, 0x2, 0x2, 0x8, 0x54, 0x3, 0x2, 0x2, 0x2, 0xa, 0x57, 0x3, 0x2, 
    0x2, 0x2, 0xc, 0x5a, 0x3, 0x2, 0x2, 0x2, 0xe, 0x71, 0x3, 0x2, 0x2, 0x2, 
    0x10, 0x86, 0x3, 0x2, 0x2, 0x2, 0x12, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x14, 
    0xb2, 0x3, 0x2, 0x2, 0x2, 0x16, 0xc1, 0x3, 0x2, 0x2, 0x2, 0x18, 0xd0, 
    0x3, 0x2, 0x2, 0x2, 0x1a, 0xdf, 0x3, 0x2, 0x2, 0x2, 0x1c, 0xee, 0x3, 
    0x2, 0x2, 0x2, 0x1e, 0xfd, 0x3, 0x2, 0x2, 0x2, 0x20, 0x10e, 0x3, 0x2, 
    0x2, 0x2, 0x22, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x24, 0x12a, 0x3, 0x2, 0x2, 
    0x2, 0x26, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x28, 0x131, 0x3, 0x2, 0x2, 0x2, 
    0x2a, 0x37, 0x7, 0xf, 0x2, 0x2, 0x2b, 0x2c, 0x7, 0x7, 0x2, 0x2, 0x2c, 
    0x31, 0x5, 0x4, 0x3, 0x2, 0x2d, 0x2e, 0x7, 0x6, 0x2, 0x2, 0x2e, 0x30, 
    0x5, 0x4, 0x3, 0x2, 0x2f, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x30, 0x33, 0x3, 
    0x2, 0x2, 0x2, 0x31, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x31, 0x32, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0x34, 0x3, 0x2, 0x2, 0x2, 0x33, 0x31, 0x3, 0x2, 0x2, 
    0x2, 0x34, 0x35, 0x7, 0x8, 0x2, 0x2, 0x35, 0x38, 0x3, 0x2, 0x2, 0x2, 
    0x36, 0x38, 0x7, 0x10, 0x2, 0x2, 0x37, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x37, 
    0x36, 0x3, 0x2, 0x2, 0x2, 0x38, 0x3, 0x3, 0x2, 0x2, 0x2, 0x39, 0x48, 
    0x5, 0xa, 0x6, 0x2, 0x3a, 0x48, 0x5, 0x8, 0x5, 0x2, 0x3b, 0x48, 0x5, 
    0x6, 0x4, 0x2, 0x3c, 0x48, 0x5, 0x12, 0xa, 0x2, 0x3d, 0x48, 0x5, 0x10, 
    0x9, 0x2, 0x3e, 0x48, 0x5, 0xe, 0x8, 0x2, 0x3f, 0x48, 0x5, 0xc, 0x7, 
    0x2, 0x40, 0x48, 0x5, 0x14, 0xb, 0x2, 0x41, 0x48, 0x5, 0x16, 0xc, 0x2, 
    0x42, 0x48, 0x5, 0x18, 0xd, 0x2, 0x43, 0x48, 0x5, 0x1e, 0x10, 0x2, 0x44, 
    0x48, 0x5, 0x1a, 0xe, 0x2, 0x45, 0x48, 0x5, 0x1c, 0xf, 0x2, 0x46, 0x48, 
    0x5, 0x2, 0x2, 0x2, 0x47, 0x39, 0x3, 0x2, 0x2, 0x2, 0x47, 0x3a, 0x3, 
    0x2, 0x2, 0x2, 0x47, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x47, 0x3c, 0x3, 0x2, 
    0x2, 0x2, 0x47, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x47, 0x3e, 0x3, 0x2, 0x2, 
    0x2, 0x47, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x47, 0x40, 0x3, 0x2, 0x2, 0x2, 
    0x47, 0x41, 0x3, 0x2, 0x2, 0x2, 0x47, 0x42, 0x3, 0x2, 0x2, 0x2, 0x47, 
    0x43, 0x3, 0x2, 0x2, 0x2, 0x47, 0x44, 0x3, 0x2, 0x2, 0x2, 0x47, 0x45, 
    0x3, 0x2, 0x2, 0x2, 0x47, 0x46, 0x3, 0x2, 0x2, 0x2, 0x48, 0x5, 0x3, 
    0x2, 0x2, 0x2, 0x49, 0x52, 0x7, 0x9, 0x2, 0x2, 0x4a, 0x4c, 0x5, 0x28, 
    0x15, 0x2, 0x4b, 0x4a, 0x3, 0x2, 0x2, 0x2, 0x4b, 0x4c, 0x3, 0x2, 0x2, 
    0x2, 0x4c, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x4d, 0x4e, 0x7, 0x7, 0x2, 0x2, 
    0x4e, 0x4f, 0x5, 0x26, 0x14, 0x2, 0x4f, 0x50, 0x7, 0x8, 0x2, 0x2, 0x50, 
    0x53, 0x3, 0x2, 0x2, 0x2, 0x51, 0x53, 0x7, 0x10, 0x2, 0x2, 0x52, 0x4b, 
    0x3, 0x2, 0x2, 0x2, 0x52, 0x51, 0x3, 0x2, 0x2, 0x2, 0x53, 0x7, 0x3, 
    0x2, 0x2, 0x2, 0x54, 0x55, 0x7, 0xa, 0x2, 0x2, 0x55, 0x56, 0x5, 0x24, 
    0x13, 0x2, 0x56, 0x9, 0x3, 0x2, 0x2, 0x2, 0x57, 0x58, 0x7, 0xb, 0x2, 
    0x2, 0x58, 0x59, 0x5, 0x22, 0x12, 0x2, 0x59, 0xb, 0x3, 0x2, 0x2, 0x2, 
    0x5a, 0x6f, 0x7, 0x15, 0x2, 0x2, 0x5b, 0x5f, 0x7, 0x7, 0x2, 0x2, 0x5c, 
    0x60, 0x5, 0x24, 0x13, 0x2, 0x5d, 0x60, 0x5, 0x1e, 0x10, 0x2, 0x5e, 
    0x60, 0x5, 0x12, 0xa, 0x2, 0x5f, 0x5c, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5d, 
    0x3, 0x2, 0x2, 0x2, 0x5f, 0x5e, 0x3, 0x2, 0x2, 0x2, 0x60, 0x69, 0x3, 
    0x2, 0x2, 0x2, 0x61, 0x65, 0x7, 0x6, 0x2, 0x2, 0x62, 0x66, 0x5, 0x1e, 
    0x10, 0x2, 0x63, 0x66, 0x5, 0x24, 0x13, 0x2, 0x64, 0x66, 0x5, 0x12, 
    0xa, 0x2, 0x65, 0x62, 0x3, 0x2, 0x2, 0x2, 0x65, 0x63, 0x3, 0x2, 0x2, 
    0x2, 0x65, 0x64, 0x3, 0x2, 0x2, 0x2, 0x66, 0x68, 0x3, 0x2, 0x2, 0x2, 
    0x67, 0x61, 0x3, 0x2, 0x2, 0x2, 0x68, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x69, 
    0x67, 0x3, 0x2, 0x2, 0x2, 0x69, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x6c, 
    0x3, 0x2, 0x2, 0x2, 0x6b, 0x69, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x6d, 0x7, 
    0x8, 0x2, 0x2, 0x6d, 0x70, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x70, 0x7, 0x10, 
    0x2, 0x2, 0x6f, 0x5b, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x6e, 0x3, 0x2, 0x2, 
    0x2, 0x70, 0xd, 0x3, 0x2, 0x2, 0x2, 0x71, 0x84, 0x7, 0x13, 0x2, 0x2, 
    0x72, 0x75, 0x7, 0x7, 0x2, 0x2, 0x73, 0x76, 0x5, 0x22, 0x12, 0x2, 0x74, 
    0x76, 0x5, 0x10, 0x9, 0x2, 0x75, 0x73, 0x3, 0x2, 0x2, 0x2, 0x75, 0x74, 
    0x3, 0x2, 0x2, 0x2, 0x76, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x77, 0x7a, 0x7, 
    0x6, 0x2, 0x2, 0x78, 0x7b, 0x5, 0x22, 0x12, 0x2, 0x79, 0x7b, 0x5, 0x10, 
    0x9, 0x2, 0x7a, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x79, 0x3, 0x2, 0x2, 
    0x2, 0x7b, 0x7d, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x77, 0x3, 0x2, 0x2, 0x2, 
    0x7d, 0x80, 0x3, 0x2, 0x2, 0x2, 0x7e, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x7e, 
    0x7f, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x81, 0x3, 0x2, 0x2, 0x2, 0x80, 0x7e, 
    0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 0x7, 0x8, 0x2, 0x2, 0x82, 0x85, 0x3, 
    0x2, 0x2, 0x2, 0x83, 0x85, 0x7, 0x10, 0x2, 0x2, 0x84, 0x72, 0x3, 0x2, 
    0x2, 0x2, 0x84, 0x83, 0x3, 0x2, 0x2, 0x2, 0x85, 0xf, 0x3, 0x2, 0x2, 
    0x2, 0x86, 0x9b, 0x7, 0x14, 0x2, 0x2, 0x87, 0x8b, 0x7, 0x7, 0x2, 0x2, 
    0x88, 0x8c, 0x5, 0x24, 0x13, 0x2, 0x89, 0x8c, 0x5, 0x1e, 0x10, 0x2, 
    0x8a, 0x8c, 0x5, 0x12, 0xa, 0x2, 0x8b, 0x88, 0x3, 0x2, 0x2, 0x2, 0x8b, 
    0x89, 0x3, 0x2, 0x2, 0x2, 0x8b, 0x8a, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x95, 
    0x3, 0x2, 0x2, 0x2, 0x8d, 0x91, 0x7, 0x6, 0x2, 0x2, 0x8e, 0x92, 0x5, 
    0x1e, 0x10, 0x2, 0x8f, 0x92, 0x5, 0x24, 0x13, 0x2, 0x90, 0x92, 0x5, 
    0x12, 0xa, 0x2, 0x91, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x91, 0x8f, 0x3, 0x2, 
    0x2, 0x2, 0x91, 0x90, 0x3, 0x2, 0x2, 0x2, 0x92, 0x94, 0x3, 0x2, 0x2, 
    0x2, 0x93, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x94, 0x97, 0x3, 0x2, 0x2, 0x2, 
    0x95, 0x93, 0x3, 0x2, 0x2, 0x2, 0x95, 0x96, 0x3, 0x2, 0x2, 0x2, 0x96, 
    0x98, 0x3, 0x2, 0x2, 0x2, 0x97, 0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x99, 
    0x7, 0x8, 0x2, 0x2, 0x99, 0x9c, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x9c, 0x7, 
    0x10, 0x2, 0x2, 0x9b, 0x87, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x9a, 0x3, 0x2, 
    0x2, 0x2, 0x9c, 0x11, 0x3, 0x2, 0x2, 0x2, 0x9d, 0xb0, 0x7, 0x12, 0x2, 
    0x2, 0x9e, 0xa1, 0x7, 0x7, 0x2, 0x2, 0x9f, 0xa2, 0x5, 0x24, 0x13, 0x2, 
    0xa0, 0xa2, 0x5, 0x1e, 0x10, 0x2, 0xa1, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xa1, 
    0xa0, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xaa, 0x3, 0x2, 0x2, 0x2, 0xa3, 0xa6, 
    0x7, 0x6, 0x2, 0x2, 0xa4, 0xa7, 0x5, 0x1e, 0x10, 0x2, 0xa5, 0xa7, 0x5, 
    0x24, 0x13, 0x2, 0xa6, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xa6, 0xa5, 0x3, 0x2, 
    0x2, 0x2, 0xa7, 0xa9, 0x3, 0x2, 0x2, 0x2, 0xa8, 0xa3, 0x3, 0x2, 0x2, 
    0x2, 0xa9, 0xac, 0x3, 0x2, 0x2, 0x2, 0xaa, 0xa8, 0x3, 0x2, 0x2, 0x2, 
    0xaa, 0xab, 0x3, 0x2, 0x2, 0x2, 0xab, 0xad, 0x3, 0x2, 0x2, 0x2, 0xac, 
    0xaa, 0x3, 0x2, 0x2, 0x2, 0xad, 0xae, 0x7, 0x8, 0x2, 0x2, 0xae, 0xb1, 
    0x3, 0x2, 0x2, 0x2, 0xaf, 0xb1, 0x7, 0x10, 0x2, 0x2, 0xb0, 0x9e, 0x3, 
    0x2, 0x2, 0x2, 0xb0, 0xaf, 0x3, 0x2, 0x2, 0x2, 0xb1, 0x13, 0x3, 0x2, 
    0x2, 0x2, 0xb2, 0xbf, 0x7, 0xc, 0x2, 0x2, 0xb3, 0xb4, 0x7, 0x7, 0x2, 
    0x2, 0xb4, 0xb9, 0x5, 0x20, 0x11, 0x2, 0xb5, 0xb6, 0x7, 0x6, 0x2, 0x2, 
    0xb6, 0xb8, 0x5, 0x20, 0x11, 0x2, 0xb7, 0xb5, 0x3, 0x2, 0x2, 0x2, 0xb8, 
    0xbb, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xba, 
    0x3, 0x2, 0x2, 0x2, 0xba, 0xbc, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xb9, 0x3, 
    0x2, 0x2, 0x2, 0xbc, 0xbd, 0x7, 0x8, 0x2, 0x2, 0xbd, 0xc0, 0x3, 0x2, 
    0x2, 0x2, 0xbe, 0xc0, 0x7, 0x10, 0x2, 0x2, 0xbf, 0xb3, 0x3, 0x2, 0x2, 
    0x2, 0xbf, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xc0, 0x15, 0x3, 0x2, 0x2, 0x2, 
    0xc1, 0xce, 0x7, 0xd, 0x2, 0x2, 0xc2, 0xc3, 0x7, 0x7, 0x2, 0x2, 0xc3, 
    0xc8, 0x5, 0x24, 0x13, 0x2, 0xc4, 0xc5, 0x7, 0x6, 0x2, 0x2, 0xc5, 0xc7, 
    0x5, 0x24, 0x13, 0x2, 0xc6, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xc7, 0xca, 0x3, 
    0x2, 0x2, 0x2, 0xc8, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x3, 0x2, 
    0x2, 0x2, 0xc9, 0xcb, 0x3, 0x2, 0x2, 0x2, 0xca, 0xc8, 0x3, 0x2, 0x2, 
    0x2, 0xcb, 0xcc, 0x7, 0x8, 0x2, 0x2, 0xcc, 0xcf, 0x3, 0x2, 0x2, 0x2, 
    0xcd, 0xcf, 0x7, 0x10, 0x2, 0x2, 0xce, 0xc2, 0x3, 0x2, 0x2, 0x2, 0xce, 
    0xcd, 0x3, 0x2, 0x2, 0x2, 0xcf, 0x17, 0x3, 0x2, 0x2, 0x2, 0xd0, 0xdd, 
    0x7, 0xe, 0x2, 0x2, 0xd1, 0xd2, 0x7, 0x7, 0x2, 0x2, 0xd2, 0xd7, 0x5, 
    0x22, 0x12, 0x2, 0xd3, 0xd4, 0x7, 0x6, 0x2, 0x2, 0xd4, 0xd6, 0x5, 0x22, 
    0x12, 0x2, 0xd5, 0xd3, 0x3, 0x2, 0x2, 0x2, 0xd6, 0xd9, 0x3, 0x2, 0x2, 
    0x2, 0xd7, 0xd5, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd8, 0x3, 0x2, 0x2, 0x2, 
    0xd8, 0xda, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xd7, 0x3, 0x2, 0x2, 0x2, 0xda, 
    0xdb, 0x7, 0x8, 0x2, 0x2, 0xdb, 0xde, 0x3, 0x2, 0x2, 0x2, 0xdc, 0xde, 
    0x7, 0x10, 0x2, 0x2, 0xdd, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xdc, 0x3, 
    0x2, 0x2, 0x2, 0xde, 0x19, 0x3, 0x2, 0x2, 0x2, 0xdf, 0xec, 0x7, 0x18, 
    0x2, 0x2, 0xe0, 0xe1, 0x7, 0x7, 0x2, 0x2, 0xe1, 0xe6, 0x5, 0x22, 0x12, 
    0x2, 0xe2, 0xe3, 0x7, 0x6, 0x2, 0x2, 0xe3, 0xe5, 0x5, 0x22, 0x12, 0x2, 
    0xe4, 0xe2, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xe6, 
    0xe4, 0x3, 0x2, 0x2, 0x2, 0xe6, 0xe7, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe9, 
    0x3, 0x2, 0x2, 0x2, 0xe8, 0xe6, 0x3, 0x2, 0x2, 0x2, 0xe9, 0xea, 0x7, 
    0x8, 0x2, 0x2, 0xea, 0xed, 0x3, 0x2, 0x2, 0x2, 0xeb, 0xed, 0x7, 0x10, 
    0x2, 0x2, 0xec, 0xe0, 0x3, 0x2, 0x2, 0x2, 0xec, 0xeb, 0x3, 0x2, 0x2, 
    0x2, 0xed, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xee, 0xfb, 0x7, 0x17, 0x2, 0x2, 
    0xef, 0xf0, 0x7, 0x7, 0x2, 0x2, 0xf0, 0xf5, 0x5, 0x22, 0x12, 0x2, 0xf1, 
    0xf2, 0x7, 0x6, 0x2, 0x2, 0xf2, 0xf4, 0x5, 0x22, 0x12, 0x2, 0xf3, 0xf1, 
    0x3, 0x2, 0x2, 0x2, 0xf4, 0xf7, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xf3, 0x3, 
    0x2, 0x2, 0x2, 0xf5, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xf6, 0xf8, 0x3, 0x2, 
    0x2, 0x2, 0xf7, 0xf5, 0x3, 0x2, 0x2, 0x2, 0xf8, 0xf9, 0x7, 0x8, 0x2, 
    0x2, 0xf9, 0xfc, 0x3, 0x2, 0x2, 0x2, 0xfa, 0xfc, 0x7, 0x10, 0x2, 0x2, 
    0xfb, 0xef, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfa, 0x3, 0x2, 0x2, 0x2, 0xfc, 
    0x1d, 0x3, 0x2, 0x2, 0x2, 0xfd, 0xfe, 0x7, 0x11, 0x2, 0x2, 0xfe, 0xff, 
    0x7, 0x7, 0x2, 0x2, 0xff, 0x104, 0x5, 0x26, 0x14, 0x2, 0x100, 0x101, 
    0x7, 0x6, 0x2, 0x2, 0x101, 0x103, 0x5, 0x26, 0x14, 0x2, 0x102, 0x100, 
    0x3, 0x2, 0x2, 0x2, 0x103, 0x106, 0x3, 0x2, 0x2, 0x2, 0x104, 0x102, 
    0x3, 0x2, 0x2, 0x2, 0x104, 0x105, 0x3, 0x2, 0x2, 0x2, 0x105, 0x107, 
    0x3, 0x2, 0x2, 0x2, 0x106, 0x104, 0x3, 0x2, 0x2, 0x2, 0x107, 0x108, 
    0x7, 0x8, 0x2, 0x2, 0x108, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10f, 0x5, 
    0x26, 0x14, 0x2, 0x10a, 0x10b, 0x7, 0x7, 0x2, 0x2, 0x10b, 0x10c, 0x5, 
    0x26, 0x14, 0x2, 0x10c, 0x10d, 0x7, 0x8, 0x2, 0x2, 0x10d, 0x10f, 0x3, 
    0x2, 0x2, 0x2, 0x10e, 0x109, 0x3, 0x2, 0x2, 0x2, 0x10e, 0x10a, 0x3, 
    0x2, 0x2, 0x2, 0x10f, 0x21, 0x3, 0x2, 0x2, 0x2, 0x110, 0x111, 0x7, 0x7, 
    0x2, 0x2, 0x111, 0x116, 0x5, 0x24, 0x13, 0x2, 0x112, 0x113, 0x7, 0x6, 
    0x2, 0x2, 0x113, 0x115, 0x5, 0x24, 0x13, 0x2, 0x114, 0x112, 0x3, 0x2, 
    0x2, 0x2, 0x115, 0x118, 0x3, 0x2, 0x2, 0x2, 0x116, 0x114, 0x3, 0x2, 
    0x2, 0x2, 0x116, 0x117, 0x3, 0x2, 0x2, 0x2, 0x117, 0x119, 0x3, 0x2, 
    0x2, 0x2, 0x118, 0x116, 0x3, 0x2, 0x2, 0x2, 0x119, 0x11a, 0x7, 0x8, 
    0x2, 0x2, 0x11a, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x11d, 0x7, 0x10, 
    0x2, 0x2, 0x11c, 0x110, 0x3, 0x2, 0x2, 0x2, 0x11c, 0x11b, 0x3, 0x2, 
    0x2, 0x2, 0x11d, 0x23, 0x3, 0x2, 0x2, 0x2, 0x11e, 0x11f, 0x7, 0x7, 0x2, 
    0x2, 0x11f, 0x124, 0x5, 0x26, 0x14, 0x2, 0x120, 0x121, 0x7, 0x6, 0x2, 
    0x2, 0x121, 0x123, 0x5, 0x26, 0x14, 0x2, 0x122, 0x120, 0x3, 0x2, 0x2, 
    0x2, 0x123, 0x126, 0x3, 0x2, 0x2, 0x2, 0x124, 0x122, 0x3, 0x2, 0x2, 
    0x2, 0x124, 0x125, 0x3, 0x2, 0x2, 0x2, 0x125, 0x127, 0x3, 0x2, 0x2, 
    0x2, 0x126, 0x124, 0x3, 0x2, 0x2, 0x2, 0x127, 0x128, 0x7, 0x8, 0x2, 
    0x2, 0x128, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12b, 0x7, 0x10, 0x2, 
    0x2, 0x12a, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x12a, 0x129, 0x3, 0x2, 0x2, 
    0x2, 0x12b, 0x25, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x12e, 0x7, 0x3, 0x2, 0x2, 
    0x12d, 0x12c, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x12f, 0x3, 0x2, 0x2, 0x2, 
    0x12f, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x12f, 0x130, 0x3, 0x2, 0x2, 0x2, 
    0x130, 0x27, 0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x7, 0x19, 0x2, 0x2, 
    0x132, 0x29, 0x3, 0x2, 0x2, 0x2, 0x28, 0x31, 0x37, 0x47, 0x4b, 0x52, 
    0x5f, 0x65, 0x69, 0x6f, 0x75, 0x7a, 0x7e, 0x84, 0x8b, 0x91, 0x95, 0x9b, 
    0xa1, 0xa6, 0xaa, 0xb0, 0xb9, 0xbf, 0xc8, 0xce, 0xd7, 0xdd, 0xe6, 0xec, 
    0xf5, 0xfb, 0x104, 0x10e, 0x116, 0x11c, 0x124, 0x12a, 0x12f, 
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
