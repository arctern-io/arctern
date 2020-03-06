
// Generated from wkt.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  wktLexer : public antlr4::Lexer {
public:
  enum {
    DECIMAL = 1, INTEGERPART = 2, DECIMALPART = 3, COMMA = 4, LPAR = 5, 
    RPAR = 6, POINT = 7, LINESTRING = 8, POLYGON = 9, MULTIPOINT = 10, MULTILINESTRING = 11, 
    MULTIPOLYGON = 12, GEOMETRYCOLLECTION = 13, EMPTY = 14, CIRCULARSTRING = 15, 
    COMPOUNDCURVE = 16, MULTISURFACE = 17, CURVEPOLYGON = 18, MULTICURVE = 19, 
    TRIANGLE = 20, TIN = 21, POLYHEDRALSURFACE = 22, STRING = 23, WS = 24
  };

  wktLexer(antlr4::CharStream *input);
  ~wktLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

