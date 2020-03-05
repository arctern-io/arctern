
// Generated from wkt.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime/antlr4-runtime.h"




class  wktParser : public antlr4::Parser {
public:
  enum {
    DECIMAL = 1, INTEGERPART = 2, DECIMALPART = 3, COMMA = 4, LPAR = 5, 
    RPAR = 6, POINT = 7, LINESTRING = 8, POLYGON = 9, MULTIPOINT = 10, MULTILINESTRING = 11, 
    MULTIPOLYGON = 12, GEOMETRYCOLLECTION = 13, EMPTY = 14, CIRCULARSTRING = 15, 
    COMPOUNDCURVE = 16, CURVEPOLYGON = 17, MULTICURVE = 18, TRIANGLE = 19, 
    TIN = 20, POLYHEDRALSURFACE = 21, STRING = 22, WS = 23
  };

  enum {
    RuleGeometry = 0, RulePointGeometry = 1, RuleLineStringGeometry = 2, 
    RulePolygonGeometry = 3, RuleMultiPointGeometry = 4, RuleMultiLineStringGeometry = 5, 
    RuleMultiPolygonGeometry = 6, RuleCircularStringGeometry = 7, RulePointOrClosedPoint = 8, 
    RulePolygon = 9, RuleLineString = 10, RulePoint = 11, RuleName = 12
  };

  wktParser(antlr4::TokenStream *input);
  ~wktParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


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

  class  GeometryContext : public antlr4::ParserRuleContext {
  public:
    GeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<PolygonGeometryContext *> polygonGeometry();
    PolygonGeometryContext* polygonGeometry(size_t i);
    std::vector<LineStringGeometryContext *> lineStringGeometry();
    LineStringGeometryContext* lineStringGeometry(size_t i);
    std::vector<PointGeometryContext *> pointGeometry();
    PointGeometryContext* pointGeometry(size_t i);
    std::vector<MultiPointGeometryContext *> multiPointGeometry();
    MultiPointGeometryContext* multiPointGeometry(size_t i);
    std::vector<MultiLineStringGeometryContext *> multiLineStringGeometry();
    MultiLineStringGeometryContext* multiLineStringGeometry(size_t i);
    std::vector<MultiPolygonGeometryContext *> multiPolygonGeometry();
    MultiPolygonGeometryContext* multiPolygonGeometry(size_t i);
    std::vector<CircularStringGeometryContext *> circularStringGeometry();
    CircularStringGeometryContext* circularStringGeometry(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GeometryContext* geometry();

  class  PointGeometryContext : public antlr4::ParserRuleContext {
  public:
    PointGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *POINT();
    antlr4::tree::TerminalNode *EMPTY();
    antlr4::tree::TerminalNode *LPAR();
    PointContext *point();
    antlr4::tree::TerminalNode *RPAR();
    NameContext *name();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PointGeometryContext* pointGeometry();

  class  LineStringGeometryContext : public antlr4::ParserRuleContext {
  public:
    LineStringGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LINESTRING();
    LineStringContext *lineString();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LineStringGeometryContext* lineStringGeometry();

  class  PolygonGeometryContext : public antlr4::ParserRuleContext {
  public:
    PolygonGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *POLYGON();
    PolygonContext *polygon();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PolygonGeometryContext* polygonGeometry();

  class  MultiPointGeometryContext : public antlr4::ParserRuleContext {
  public:
    MultiPointGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MULTIPOINT();
    antlr4::tree::TerminalNode *LPAR();
    std::vector<PointOrClosedPointContext *> pointOrClosedPoint();
    PointOrClosedPointContext* pointOrClosedPoint(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MultiPointGeometryContext* multiPointGeometry();

  class  MultiLineStringGeometryContext : public antlr4::ParserRuleContext {
  public:
    MultiLineStringGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MULTILINESTRING();
    antlr4::tree::TerminalNode *LPAR();
    std::vector<LineStringContext *> lineString();
    LineStringContext* lineString(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MultiLineStringGeometryContext* multiLineStringGeometry();

  class  MultiPolygonGeometryContext : public antlr4::ParserRuleContext {
  public:
    MultiPolygonGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MULTIPOLYGON();
    antlr4::tree::TerminalNode *EMPTY();
    antlr4::tree::TerminalNode *LPAR();
    std::vector<PolygonContext *> polygon();
    PolygonContext* polygon(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MultiPolygonGeometryContext* multiPolygonGeometry();

  class  CircularStringGeometryContext : public antlr4::ParserRuleContext {
  public:
    CircularStringGeometryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CIRCULARSTRING();
    antlr4::tree::TerminalNode *LPAR();
    std::vector<PointContext *> point();
    PointContext* point(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CircularStringGeometryContext* circularStringGeometry();

  class  PointOrClosedPointContext : public antlr4::ParserRuleContext {
  public:
    PointOrClosedPointContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    PointContext *point();
    antlr4::tree::TerminalNode *LPAR();
    antlr4::tree::TerminalNode *RPAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PointOrClosedPointContext* pointOrClosedPoint();

  class  PolygonContext : public antlr4::ParserRuleContext {
  public:
    PolygonContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAR();
    std::vector<LineStringContext *> lineString();
    LineStringContext* lineString(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PolygonContext* polygon();

  class  LineStringContext : public antlr4::ParserRuleContext {
  public:
    LineStringContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAR();
    std::vector<PointContext *> point();
    PointContext* point(size_t i);
    antlr4::tree::TerminalNode *RPAR();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  LineStringContext* lineString();

  class  PointContext : public antlr4::ParserRuleContext {
  public:
    PointContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> DECIMAL();
    antlr4::tree::TerminalNode* DECIMAL(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  PointContext* point();

  class  NameContext : public antlr4::ParserRuleContext {
  public:
    NameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
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

