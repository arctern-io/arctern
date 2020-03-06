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
#include "gis/gdal/generated/wktListener.h"

/**
 * This class provides an empty implementation of wktListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class wktBaseListener : public wktListener {
 public:
  void enterGeometryCollection(wktParser::GeometryCollectionContext* /*ctx*/) override {}
  void exitGeometryCollection(wktParser::GeometryCollectionContext* /*ctx*/) override {}

  void enterGeometry(wktParser::GeometryContext* /*ctx*/) override {}
  void exitGeometry(wktParser::GeometryContext* /*ctx*/) override {}

  void enterPointGeometry(wktParser::PointGeometryContext* /*ctx*/) override {}
  void exitPointGeometry(wktParser::PointGeometryContext* /*ctx*/) override {}

  void enterLineStringGeometry(wktParser::LineStringGeometryContext* /*ctx*/) override {}
  void exitLineStringGeometry(wktParser::LineStringGeometryContext* /*ctx*/) override {}

  void enterPolygonGeometry(wktParser::PolygonGeometryContext* /*ctx*/) override {}
  void exitPolygonGeometry(wktParser::PolygonGeometryContext* /*ctx*/) override {}

  void enterMultiPointGeometry(wktParser::MultiPointGeometryContext* /*ctx*/) override {}
  void exitMultiPointGeometry(wktParser::MultiPointGeometryContext* /*ctx*/) override {}

  void enterMultiLineStringGeometry(
      wktParser::MultiLineStringGeometryContext* /*ctx*/) override {}
  void exitMultiLineStringGeometry(
      wktParser::MultiLineStringGeometryContext* /*ctx*/) override {}

  void enterMultiPolygonGeometry(
      wktParser::MultiPolygonGeometryContext* /*ctx*/) override {}
  void exitMultiPolygonGeometry(
      wktParser::MultiPolygonGeometryContext* /*ctx*/) override {}

  void enterCircularStringGeometry(
      wktParser::CircularStringGeometryContext* /*ctx*/) override {}
  void exitCircularStringGeometry(
      wktParser::CircularStringGeometryContext* /*ctx*/) override {}

  void enterPointOrClosedPoint(wktParser::PointOrClosedPointContext* /*ctx*/) override {}
  void exitPointOrClosedPoint(wktParser::PointOrClosedPointContext* /*ctx*/) override {}

  void enterPolygon(wktParser::PolygonContext* /*ctx*/) override {}
  void exitPolygon(wktParser::PolygonContext* /*ctx*/) override {}

  void enterLineString(wktParser::LineStringContext* /*ctx*/) override {}
  void exitLineString(wktParser::LineStringContext* /*ctx*/) override {}

  void enterPoint(wktParser::PointContext* /*ctx*/) override {}
  void exitPoint(wktParser::PointContext* /*ctx*/) override {}

  void enterName(wktParser::NameContext* /*ctx*/) override {}
  void exitName(wktParser::NameContext* /*ctx*/) override {}

  void enterEveryRule(antlr4::ParserRuleContext* /*ctx*/) override {}
  void exitEveryRule(antlr4::ParserRuleContext* /*ctx*/) override {}
  void visitTerminal(antlr4::tree::TerminalNode* /*node*/) override {}
  void visitErrorNode(antlr4::tree::ErrorNode* /*node*/) override {}
};
