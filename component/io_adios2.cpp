/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  IO_ADIOS2.cpp (main):
    file input-output using ADIOS2 library

\*****************************************************************************/

#include "io_adios2.hpp"

namespace BSIM
{

IO_ADIOS2::IO_ADIOS2(std::string adios2file, Vec2<lpUint> number_of_grid_point, Vec2<lpUint> shared_edge_size)
{
	if( !this->init(adios2file, number_of_grid_point, shared_edge_size) ){
		std::cerr
		<< "\nERROR while running BSIM::io_adios2.cpp \n"
		<< "--> Cannot initialize ADIOS2 IO " << std::endl;
	}
}

IO_ADIOS2::~IO_ADIOS2()
{
	this->finalize();
}


bool IO_ADIOS2::init(std::string adios2file, Vec2<lpUint> number_of_grid_point, Vec2<lpUint> shared_edge_size)
{
  try
  {
    this->ptr_adios2 = new adios2::ADIOS(adios2file, MPI_COMM_WORLD);

    this->write_checkpoint = this->ptr_adios2->DeclareIO("BP_Write_CheckPoint");
    if( !this->write_checkpoint.InConfigFile() ){
      std::cout
      << "\nWARNING while running BSIM::io_adios2.cpp \n"
      << "--> BP_Write_CheckPoint IO is NOT defined in " << adios2file << "\n"
      << "--> BP4 format and ADIOS2 default parameters will be used." << std::endl;
      this->write_checkpoint.SetEngine("BP4");
    }

    this->write_output = this->ptr_adios2->DeclareIO("BP_Write_Output");
    if( !this->write_output.InConfigFile() ){
      std::cout
      << "\nWARNING while running BSIM::io_adios2.cpp \n"
      << "--> BP_Write_Output IO is NOT defined in " << adios2file << "\n"
      << "--> BP4 format and ADIOS2 default parameters will be used." << std::endl;
      this->write_output.SetEngine("BP4");
    }

    this->read_checkpoint = this->ptr_adios2->DeclareIO("BP_Read_CheckPoint");
    if( !this->read_checkpoint.InConfigFile() ){
      std::cout
      << "\nWARNING while running BSIM::io_adios2.cpp \n"
      << "--> BP_Read_CheckPoint IO is NOT defined in " << adios2file << "\n"
      << "--> BP4 format and ADIOS2 default parameters will be used." << std::endl;
      this->read_checkpoint.SetEngine("BP4");
    }

    this->read_metinfo = this->ptr_adios2->DeclareIO("BP_Read_Metinfo");
    if( !this->read_metinfo.InConfigFile() ){
      std::cout
      << "\nWARNING while running BSIM::io_adios2.cpp \n"
      << "--> BP_Read_Metinfo IO is NOT defined in " << adios2file << "\n"
      << "--> BP4 format and ADIOS2 default parameters will be used." << std::endl;
      this->read_metinfo.SetEngine("BP4");
    }

  }
  catch (std::invalid_argument &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> Initializing ADIOS2 IO \n"
    << "--> Invalid argument exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::ios_base::failure &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> Initializing ADIOS2 IO \n"
    << "--> IO System base failure exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::exception &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> Initializing ADIOS2 IO \n"
    << "--> Exception: " << e.what() << std::endl ;
    return false;
  }

  try
  {

    // Checkpoint
    Vec2<lpUint> N = number_of_grid_point + 2 * shared_edge_size ;

    for(lpInt i=0 ; i!=BSIM_TOTAL_NUM_VAR ; ++i)
    {
      std::string name = BSIM_VAR::name[i] ;

      this->checkpoint_var.push_back( this->write_checkpoint.DefineVariable<Float>(name,
                                     {N.y, N.x},
                                     {},
                                     {},
                                     !adios2::ConstantDims) );
      this->checkpoint_grad_x.push_back( this->write_checkpoint.DefineVariable<Float>(name + "_dx",
                                        {N.y, N.x},
                                        {},
                                        {},
                                        !adios2::ConstantDims) );
      this->checkpoint_grad_y.push_back( this->write_checkpoint.DefineVariable<Float>(name + "_dy",
                                        {N.y, N.x},
                                        {},
                                        {},
                                        !adios2::ConstantDims) );
    }
    //filename.append(domain->checkpoint_prefix);

    // Output
    for(lpInt i=0 ; i!=BSIM_OUTPUT_NUM_VAR ; ++i)
    {
      std::string name = BSIM_VAR::name[BSIM_OUTPUT::index[i]] ;

      this->output_var.push_back( this->write_output.DefineVariable<Float>(name,
                                     {number_of_grid_point.y, number_of_grid_point.x},
                                     {},
                                     {},
                                     !adios2::ConstantDims) );
      this->output_grad_x.push_back( this->write_output.DefineVariable<Float>(name + "_dx",
                                        {number_of_grid_point.y, number_of_grid_point.x},
                                        {},
                                        {},
                                        !adios2::ConstantDims) );
      this->output_grad_y.push_back( this->write_output.DefineVariable<Float>(name + "_dy",
                                        {number_of_grid_point.y, number_of_grid_point.x},
                                        {},
                                        {},
                                        !adios2::ConstantDims) );
    }

  }
  catch (std::invalid_argument &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> declaring variables \n"
    << "--> Invalid argument exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::ios_base::failure &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> declaring variables \n"
    << "--> IO System base failure exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::exception &e)
  {
    std::cerr
    << "\nERROR while running BSIM::io_adios2.cpp \n"
    << "--> declaring variables \n"
    << "--> Exception: " << e.what() << std::endl ;
    return false;
  }

return true; }


void IO_ADIOS2::finalize()
{
  this->ptr_adios2->FlushAll();
  delete this->ptr_adios2 ;
}

} // NAMESPACE BSIM
