/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  parser.cpp (main):
    Setup file parser and parameter file parser

\*****************************************************************************/

#include "parser.hpp"

namespace BSIM
{

Parser::Parser()
{

}

Parser::~Parser()
{

}


bool Parser::parse_setup(const char *path)
{
  // Read file
  std::ifstream file(path, std::ifstream::in);
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  // Parse each line
  std::istringstream iss(buffer.str());
  std::string line;
  while( std::getline(iss, line) )
  {
    // Remove comments (after #)
    size_t end = line.find('#');
    if( end == std::string::npos ){ end = line.length() ; }
    line = line.substr(0, end) ;

    // Trim trailing spaces
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    if( line.length() == 0 ){ continue; }

    // Split by =
    std::vector<std::string> v = this->splitString(line, '=');
    if( v.size() == 2 ){
      if( this->parseSetupLine(v[0],v[1]) == false ){
        return false ;
      }
    }else{
      std::cerr
      << "Error while running BSIM::parser.cpp \n"
      << "--> Incorrect setup line :: " << line << std::endl;
      return false ;
    }
  }

  // Additional checks for correctness
  if( this->mpi_map.size() == 0 )
  {
    for(Int i = 0 ; i < this->num_process.x*this->num_process.y ; ++i)
    {
      this->mpi_map.push_back( i );
    }
  }else{
    if( static_cast<lpUint>(this->mpi_map.size()) != this->num_process.x * this->num_process.y )
    {
      std::cerr
      << "\nERROR while running BSIM::parser.cpp \n"
      << "--> The size of \"mpi_map\" is not equal to num_process.x * num_process.y. "
      << "Recheck the runtime.setup.BSIM file" << std::endl ;
      return false;
    }
  }

  if( this->timestep_divide_second != 1 && this->timestep_second != 1 )
  {
    if( this->timestep_divide_second != 0 )
    {
      std::cerr
      << "\nERROR while running BSIM::parser.cpp \n"
      << "--> timestep_second must be 1 when timestep_divide_second is >1."
      << "Recheck the runtime.setup.BSIM file" << std::endl ;
    }else{
      std::cerr
      << "\nERROR while running BSIM::parser.cpp \n"
      << "--> timestep_divide_second cannot be zero !!"
      << "Recheck the runtime.setup.BSIM file" << std::endl ;
    }
    return false;
  }

return true; }


void Parser::print_setting()
{
  DateTime endtime = this->starttime ;
  endtime += this->total_simulationtime ;

  std::cout << "\n @ -> @ -> @   BSIM   @ <- @ <- @ \n"
  << "\n"
  << "Start from " << this->starttime.to_str() << "\n"
  << "       to  " << endtime.to_str()   << "\n"
  << "Using timestep size = " << this->timestep_second << " seconds\n"
  << "\n"
  << "Using grid size = " << this->number_of_grid_point << " with grid spacing = " << this->grid_spacing << "\n"
  << "Utilizing " << this->num_process << " MPI processes, each has " << this->num_thread << " OpenMP threads \n"
  << "\n"
  << "Input from " << this->input_dir  << " every " << this->input_interval.to_str()  << "\n"
  << "Output to  " << this->output_dir << " every " << this->output_interval.to_str() << "\n"
  << "Checkpoints created every " << this->checkpoint_interval.to_str()
  << "\n" << std::endl;
}


bool Parser::parseSetupLine(const std::string opt, const std::string val)
{
  // time section
  if( opt.compare("start_time")==0 ){
    this->starttime.from_string(val);
    this->currenttime.from_string(val);

  }else if( opt.compare("total_simulation_time")==0 ){
    this->total_simulationtime.from_string(val);

  }else if( opt.compare("timestep_second")==0 ){
    this->timestep_second = str2Uint(val);

  }else if( opt.compare("timestep_divide_second")==0 ){
    this->timestep_divide_second = str2Uint(val);

  // Input section
  }else if( opt.compare("input_dir")==0 ){
    this->input_dir = val ;

  }else if( opt.compare("metinfo_prefix")==0 ){
    this->metinfo_prefix = val ;

  }else if( opt.compare("geoinfo_prefix")==0 ){
    this->geoinfo_prefix = val ;

  }else if( opt.compare("input_interval")==0 ){
    this->input_interval.from_string(val);

  // Output section
  }else if( opt.compare("output_dir")==0 ){
    this->output_dir = val ;

  }else if( opt.compare("output_prefix")==0 ){
    this->output_prefix = val ;

  }else if( opt.compare("output_interval")==0 ){
    this->output_interval.from_string(val);

  }else if( opt.compare("checkpoint_prefix")==0 ){
    this->checkpoint_prefix = val ;

  }else if( opt.compare("checkpoint_interval")==0 ){
    this->checkpoint_interval.from_string(val);

  // Domain section
  }else if( opt.compare("number_of_grid_point")==0 ){
    this->number_of_grid_point = this->parseVec2<lpUint>(val, str2lpUint);

  }else if( opt.compare("shared_edge_size")==0 ){
    this->shared_edge_size = this->parseVec2<lpUint>(val, str2lpUint);

  }else if( opt.compare("grid_spacing")==0 ){
    this->grid_spacing = this->parseVec2<Float>(val, str2Float);

  }else if( opt.compare("global_pos0")==0 ){
    this->global_pos0 = this->parseVec2<Float>(val, str2Float);

  // BC section
  }else if( opt.compare("boundary_type")==0 ){
    this->boundary_type = str2lpUint(val);

  // Optimization
  }else if( opt.compare("num_process")==0 ){
    this->num_process = this->parseVec2<lpUint>(val, str2lpUint);

  }else if( opt.compare("num_thread")==0 ){
    this->num_thread = str2lpUint(val);

  }else if( opt.compare("mpi_map")==0 ){
    if( !this->parseArray<Int>(val, str2Int, &(this->mpi_map)) )
    {
      std::cerr
      << "Error while running BSIM::parser.hpp \n"
      << "--> Parsing mpi_map in the input Setup file.\n" ;
      return false ;
    }

  // Others
  }else if( opt.compare("iter_per_stdout")==0 ){
    this->iter_per_stdout = str2Uint(val);

  }else if( opt.compare("adios2_config_file")==0 ){
    this->adios2file = val ;

  // Unknown option
  }else{
    std::cerr
    << "Error while running BSIM::parser.cpp \n"
    << "--> Unknown option, " << opt << ", was found in the Setup file.\n" ;
    return false;
  }

return true; }


std::vector<std::string> Parser::splitString(const std::string line, const char delimiter)
{
  std::string temp = line ;
  std::vector<std::string> words ;
  size_t mid = temp.find(delimiter) ;
  while( 0 < mid && mid < temp.length() )
  {
    words.push_back( temp.substr(0, mid) );
    temp = temp.substr(mid+1);
    mid  = temp.find(delimiter);
  }
  if( temp.length() != 0 ){
    words.push_back( temp );
  }

return words ; }


bool Parser::isInList(const std::string val, const std::vector<std::string> list)
{
  for(lpUint i=0 ; i != list.size() ; ++i)
  {
    if( val.compare(list[i])==0 )
    {
      return true;
    }
  }

return false; }


bool Parser::get_domain_id(Int rank, Vec2<lpInt> &id)
{
  bool found = false ;
  for(lpUint j=0 ; j != this->num_process.y ; ++j)
  {
    for(lpUint i=0 ; i != this->num_process.x ; ++i)
    {
      if( rank == this->mpi_map[j*this->num_process.x + i] )
      {
        id.x = i ;
        id.y = j ;
        found = true ;
        break;
      }
    }
  }

return found; }


} // NAMESPACE BSIM
