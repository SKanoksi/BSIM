/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  parser.hpp (header):
    Setup file parser and parameter file parser

\*****************************************************************************/

#ifndef BSIM_PARSER_HPP
#define BSIM_PARSER_HPP

#include <fstream>
#include <sstream>
#include <vector>

#include <root.hpp>

namespace BSIM
{

class Parser : public BSIM_root
{
public:
  Parser();
  ~Parser();

  bool parse_setup(const char *path);
  void print_setting();
  bool get_domain_id(Int rank, Vec2<lpInt> &id);

private:
  bool parseSetupLine(const std::string opt, const std::string val);

  std::vector<std::string> splitString(const std::string line, const char delimiter);
  bool isInList(const std::string val, const std::vector<std::string> list);

  template <typename T>
  Vec2<T> parseVec2(const std::string val, T toNum(std::string)){
    size_t start = val.find('(');
    size_t sep   = val.find(',', start+1);
    size_t end   = val.find(')', sep+1);
    return Vec2<T>( toNum(val.substr(start+1,sep)), toNum(val.substr(sep+1,end)) );
  }

  template <typename T>
  bool parseArray(const std::string val, T toNum(std::string), std::vector<T> *array){
    size_t start = val.find('[');
    size_t end   = val.find(']', start+1);
    if( start == std::string::npos || end == std::string::npos || start >= end ){
      std::cerr
      << "Error while running BSIM::parser.hpp \n"
      << "--> Bad array input without [ or ] was encountered while parsing the Setup file.\n" ;
      return false ;
    }

    size_t lsep = start ;
    size_t rsep = val.find(',', lsep+1) ;
    for(; rsep != std::string::npos ; rsep = val.find(',', lsep+1))
    {
      array->push_back( toNum(val.substr(lsep+1,rsep)) ) ;
      lsep = rsep ;
    }
    array->push_back( toNum(val.substr(lsep+1,end)) ) ;

    return true ;
  }

};

} // NAMESPACE BSIM

#endif // BSIM_PARSER_HPP
