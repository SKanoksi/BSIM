/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  IO_ADIOS2.hpp (header):
    file input-output using ADIOS2 library

\*****************************************************************************/

#ifndef BSIM_IO_ADIOS2_HPP
#define BSIM_IO_ADIOS2_HPP

#include <adios2.h>

#include <mpi.h>
#include <vector>
#include <iostream>

#include <root.hpp>
#include <variable.hpp>

namespace BSIM
{

class IO_ADIOS2
{
public:
  IO_ADIOS2(std::string adios2file, Vec2<lpUint> number_of_grid_point, Vec2<lpUint> shared_edge_size);
  ~IO_ADIOS2();

  adios2::IO write_checkpoint, read_checkpoint  ;
  adios2::IO write_output, read_metinfo ;

  std::vector<adios2::Variable<Float>> checkpoint_var, checkpoint_grad_x, checkpoint_grad_y ;
  std::vector<adios2::Variable<Float>>     output_var,     output_grad_x,     output_grad_y ;

private:
  adios2::ADIOS *ptr_adios2 = nullptr ;

  bool init(std::string adios2file, Vec2<lpUint> number_of_grid_point, Vec2<lpUint> shared_edge_size);
  void finalize();

};

} // NAMESPACE BSIM

#endif // BSIM_IO_ADIOS2_HPP
