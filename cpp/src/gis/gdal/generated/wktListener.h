
// Generated from wkt.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime/antlr4-runtime.h"
#include "wktParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by wktParser.
 */
class  wktListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterGeometry(wktParser::GeometryContext *ctx) = 0;
  virtual void exitGeometry(wktParser::GeometryContext *ctx) = 0;

  virtual void enterPointGeometry(wktParser::PointGeometryContext *ctx) = 0;
  virtual void exitPointGeometry(wktParser::PointGeometryContext *ctx) = 0;

  virtual void enterLineStringGeometry(wktParser::LineStringGeometryContext *ctx) = 0;
  virtual void exitLineStringGeometry(wktParser::LineStringGeometryContext *ctx) = 0;

  virtual void enterPolygonGeometry(wktParser::PolygonGeometryContext *ctx) = 0;
  virtual void exitPolygonGeometry(wktParser::PolygonGeometryContext *ctx) = 0;

  virtual void enterMultiPointGeometry(wktParser::MultiPointGeometryContext *ctx) = 0;
  virtual void exitMultiPointGeometry(wktParser::MultiPointGeometryContext *ctx) = 0;

  virtual void enterMultiLineStringGeometry(wktParser::MultiLineStringGeometryContext *ctx) = 0;
  virtual void exitMultiLineStringGeometry(wktParser::MultiLineStringGeometryContext *ctx) = 0;

  virtual void enterMultiPolygonGeometry(wktParser::MultiPolygonGeometryContext *ctx) = 0;
  virtual void exitMultiPolygonGeometry(wktParser::MultiPolygonGeometryContext *ctx) = 0;

  virtual void enterCircularStringGeometry(wktParser::CircularStringGeometryContext *ctx) = 0;
  virtual void exitCircularStringGeometry(wktParser::CircularStringGeometryContext *ctx) = 0;

  virtual void enterPointOrClosedPoint(wktParser::PointOrClosedPointContext *ctx) = 0;
  virtual void exitPointOrClosedPoint(wktParser::PointOrClosedPointContext *ctx) = 0;

  virtual void enterPolygon(wktParser::PolygonContext *ctx) = 0;
  virtual void exitPolygon(wktParser::PolygonContext *ctx) = 0;

  virtual void enterLineString(wktParser::LineStringContext *ctx) = 0;
  virtual void exitLineString(wktParser::LineStringContext *ctx) = 0;

  virtual void enterPoint(wktParser::PointContext *ctx) = 0;
  virtual void exitPoint(wktParser::PointContext *ctx) = 0;

  virtual void enterName(wktParser::NameContext *ctx) = 0;
  virtual void exitName(wktParser::NameContext *ctx) = 0;


};

