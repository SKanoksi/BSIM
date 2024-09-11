/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  datetime.hpp (header):
    date time in YYYY-MM-DD_HH-MM-SS and related functions

\*****************************************************************************/

#ifndef BSIM_DATETIME_HPP
#define BSIM_DATETIME_HPP

#include <string>
#include <iostream>

#include <datatype.hpp>

namespace BSIM
{

class DateTime
{
public:
  DateTime(const bool is_date);
  DateTime(const Uint yy, const Uint mm, const Uint dd, const Uint hr, const Uint min, const Uint sec,
           const bool is_date);
  ~DateTime();

  bool operator== (const DateTime &other);
  bool operator<= (const DateTime &other);
  void operator+= (const DateTime &other);
  void operator+= (const Uint second);

  bool from_string(const std::string &str_in);
  const std::string to_str();
  hpUint to_second();

private:
  Uint year   ;
  Uint month  ;
  Uint day    ;
  Uint hour   ;
  Uint minute ;
  Uint second ;
  const bool isDate ; // true = date (time point), false = days (amount)

  void recheck();

};

} // NAMESPACE BSIM

#endif // BSIM_DATETIME_HPP
