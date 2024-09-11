/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  datatype.cpp (source):
    standard internal datatypes

\*****************************************************************************/

#include "datatype.hpp"

Float str2Float(const std::string str){ return std::stod(str); }
lpFloat str2lpFloat(const std::string str){ return std::stof(str); }
hpFloat str2hpFloat(const std::string str){ return std::stold(str); }

Int str2Int(const std::string str){ return std::stoi(str); }
lpInt str2lpInt(const std::string str){ return static_cast<lpInt>( std::stoi(str) ); }
hpInt str2hpInt(const std::string str){ return std::stoll(str); }

Uint str2Uint(const std::string str){ return static_cast<Uint>( std::stoul(str) ); }
lpUint str2lpUint(const std::string str){ return static_cast<lpUint>( std::stoul(str) ); }
hpUint str2hpUint(const std::string str){ return std::stoull(str); }
