
// Generated from wkt.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime/antlr4-runtime.h"
#include "wktListener.h"


/**
 * This class provides an empty implementation of wktListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  wktBaseListener : public wktListener {
public:

  virtual void enterGeometry(wktParser::GeometryContext * /*ctx*/) override { }
  virtual void exitGeometry(wktParser::GeometryContext * /*ctx*/) override { }

  virtual void enterPointGeometry(wktParser::PointGeometryContext * /*ctx*/) override { }
  virtual void exitPointGeometry(wktParser::PointGeometryContext * /*ctx*/) override { }

  virtual void enterLineStringGeometry(wktParser::LineStringGeometryContext * /*ctx*/) override { }
  virtual void exitLineStringGeometry(wktParser::LineStringGeometryContext * /*ctx*/) override { }

  virtual void enterPolygonGeometry(wktParser::PolygonGeometryContext * /*ctx*/) override { }
  virtual void exitPolygonGeometry(wktParser::PolygonGeometryContext * /*ctx*/) override { }

  virtual void enterMultiPointGeometry(wktParser::MultiPointGeometryContext * /*ctx*/) override { }
  virtual void exitMultiPointGeometry(wktParser::MultiPointGeometryContext * /*ctx*/) override { }

  virtual void enterMultiLineStringGeometry(wktParser::MultiLineStringGeometryContext * /*ctx*/) override { }
  virtual void exitMultiLineStringGeometry(wktParser::MultiLineStringGeometryContext * /*ctx*/) override { }

  virtual void enterMultiPolygonGeometry(wktParser::MultiPolygonGeometryContext * /*ctx*/) override { }
  virtual void exitMultiPolygonGeometry(wktParser::MultiPolygonGeometryContext * /*ctx*/) override { }

  virtual void enterCircularStringGeometry(wktParser::CircularStringGeometryContext * /*ctx*/) override { }
  virtual void exitCircularStringGeometry(wktParser::CircularStringGeometryContext * /*ctx*/) override { }

  virtual void enterPointOrClosedPoint(wktParser::PointOrClosedPointContext * /*ctx*/) override { }
  virtual void exitPointOrClosedPoint(wktParser::PointOrClosedPointContext * /*ctx*/) override { }

  virtual void enterPolygon(wktParser::PolygonContext * /*ctx*/) override { }
  virtual void exitPolygon(wktParser::PolygonContext * /*ctx*/) override { }

  virtual void enterLineString(wktParser::LineStringContext * /*ctx*/) override { }
  virtual void exitLineString(wktParser::LineStringContext * /*ctx*/) override { }

  virtual void enterPoint(wktParser::PointContext * /*ctx*/) override { }
  virtual void exitPoint(wktParser::PointContext * /*ctx*/) override { }

  virtual void enterName(wktParser::NameContext * /*ctx*/) override { }
  virtual void exitName(wktParser::NameContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

