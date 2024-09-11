/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  variable.hpp (header):
    BSIM variables

\*****************************************************************************/

#ifndef BSIM_VARIABLE_HPP
#define BSIM_VARIABLE_HPP

#include <datatype.hpp>

// Units:
// density   = kg/m^3
// velocity  = m/s
// pressure  = Pa = kg/m/s^2

#define BSIM_TOTAL_NUM_VAR 3
#define BSIM_OUTPUT_NUM_VAR 3

namespace BSIM_VAR
{
  enum class tag
  {
    u   = 0,
    v   = 1,
    T   = 2
  };

  extern const char *name[] ;
}

namespace BSIM_OUTPUT
{
  extern const lpUint index[] ;
}



#endif // BSIM_VARIABLE_HPP
