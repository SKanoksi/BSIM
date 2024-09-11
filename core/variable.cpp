/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  variable.cpp (header):
    BSIM variables

\*****************************************************************************/

#include "variable.hpp"

namespace BSIM_VAR
{
  const char *name[] = {
    "u_velocity",
    "v_velocity",
    "T_temperature"
  };

}

namespace BSIM_OUTPUT
{
  const lpUint index[] = {0,1,2};
}
