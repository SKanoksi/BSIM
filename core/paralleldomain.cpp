/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  paralleldomain.cpp (main):
    Variables, memory allocation and domain decomposition

\*****************************************************************************/

#include "paralleldomain.hpp"

namespace BSIM
{

Parallel_Domain::Parallel_Domain(const Vec2<lpInt> domain_id) : id(domain_id)
{
  if( this->boundary_type==0 ){
    this->rank_sw = this->rank_from_id_externalBC(this->id.x-1, this->id.y-1);
    this->rank_ss = this->rank_from_id_externalBC(this->id.x  , this->id.y-1);
    this->rank_se = this->rank_from_id_externalBC(this->id.x+1, this->id.y-1);
    this->rank_ee = this->rank_from_id_externalBC(this->id.x+1, this->id.y  );
    this->rank_ne = this->rank_from_id_externalBC(this->id.x+1, this->id.y+1);
    this->rank_nn = this->rank_from_id_externalBC(this->id.x  , this->id.y+1);
    this->rank_nw = this->rank_from_id_externalBC(this->id.x-1, this->id.y+1);
    this->rank_ww = this->rank_from_id_externalBC(this->id.x-1, this->id.y  );
  }else{
    this->rank_sw = this->rank_from_id_periodicBC(this->id.x-1, this->id.y-1);
    this->rank_ss = this->rank_from_id_periodicBC(this->id.x  , this->id.y-1);
    this->rank_se = this->rank_from_id_periodicBC(this->id.x+1, this->id.y-1);
    this->rank_ee = this->rank_from_id_periodicBC(this->id.x+1, this->id.y  );
    this->rank_ne = this->rank_from_id_periodicBC(this->id.x+1, this->id.y+1);
    this->rank_nn = this->rank_from_id_periodicBC(this->id.x  , this->id.y+1);
    this->rank_nw = this->rank_from_id_periodicBC(this->id.x-1, this->id.y+1);
    this->rank_ww = this->rank_from_id_periodicBC(this->id.x-1, this->id.y  );
  }



  Vec2<lpInt> patch_size, temp ;
  Vec2<lpInt> stride = BSIM::func::vec2::floor(Vec2<lpInt>(this->number_of_grid_point/this->num_process)) ;

  // X
  temp = this->compute_range_local_domain(this->id.x, this->num_process.x, stride.x, this->number_of_grid_point.x);
  this->origin_on_global_grid.x = temp.x ;
  patch_size.x = temp.y - temp.x ;

  // Y
  temp = this->compute_range_local_domain(this->id.y, this->num_process.y, stride.y, this->number_of_grid_point.y);
  this->origin_on_global_grid.y = temp.x ;
  patch_size.y = temp.y - temp.x ;

  this->allocate_variable(patch_size,
                          this->shared_edge_size,
                          this->global_pos0 + Vec2<Float>(this->origin_on_global_grid)*this->grid_spacing,
                          this->grid_spacing);

  this->xx_size = patch_size.x             * this->shared_edge_size.y ;
  this->xy_size = this->shared_edge_size.x * this->shared_edge_size.y ;
  this->yy_size = this->shared_edge_size.x *             patch_size.y ;

  // ---

  this->IO = new IO_ADIOS2(this->adios2file, this->number_of_grid_point, this->shared_edge_size);

}


Parallel_Domain::~Parallel_Domain()
{
  delete this->IO ;
  this->deallocate_variable();
}


bool Parallel_Domain::adios2_read_checkpoint(const std::string filename)
{
  bool normal_exit = true ;

  try
  {
    adios2::Engine reader = this->IO->read_checkpoint.Open(filename, adios2::Mode::Read);

    bool has_grad_x[BSIM_TOTAL_NUM_VAR] ;
    bool has_grad_y[BSIM_TOTAL_NUM_VAR] ;

    Vec2<lpUint> start = Vec2<lpUint>(this->origin_on_global_grid);
    Vec2<lpUint> N     = Vec2<lpUint>(this->BSIM_var[0]->val->full_size);

    for(lpInt i=0 ; i != BSIM_TOTAL_NUM_VAR ; ++i)
    {
      std::string name = BSIM_VAR::name[i] ;

      // value
      adios2::Variable<Float> adios2_var = this->IO->read_checkpoint.InquireVariable<Float>(name);
      if( adios2_var )
      {
        adios2_var.SetSelection({ {start.y, start.x}, {N.y, N.x} });
        reader.Get(adios2_var, this->BSIM_var[i]->val->data, adios2::Mode::Deferred);

        // grad_x
        adios2::Variable<Float> adios2_grad_x = this->IO->read_checkpoint.InquireVariable<Float>(name + "_dx");
        if( adios2_grad_x )
        {
          adios2_grad_x.SetSelection({ {start.y, start.x}, {N.y, N.x} });
          reader.Get(adios2_grad_x, this->BSIM_var[i]->grad_x->data, adios2::Mode::Deferred);
          has_grad_x[i] = true ;
        }else{
          has_grad_x[i] = false ;
        }

        // grad_y
        adios2::Variable<Float> adios2_grad_y = this->IO->read_checkpoint.InquireVariable<Float>(name + "_dy");
        if( adios2_grad_y )
        {
          adios2_grad_y.SetSelection({ {start.y, start.x}, {N.y, N.x} });
          reader.Get(adios2_grad_y, this->BSIM_var[i]->grad_y->data, adios2::Mode::Deferred);
          has_grad_y[i] = true ;
        }else{
          has_grad_y[i] = false ;
        }

      }else{
        std::cerr
        << "\nERROR while running BSIM::paralleldomain.cpp"
        << "\n--> read_checkpoint"
        << "\n--> Cannot find variable " << name << " in "<< filename << std::endl ;
        normal_exit = false ;
      }

    }

    reader.PerformGets(); // Actual IO threading is managed by adios2_config_file.
    reader.Close();

    for(lpInt i=0 ; i != BSIM_TOTAL_NUM_VAR ; ++i)
    {
      if( !has_grad_x[i] ){
        this->BSIM_var[i]->cal_grad_x();
      }
      if( !has_grad_y[i] ){
        this->BSIM_var[i]->cal_grad_y();
      }
    }

  }
  catch (std::invalid_argument &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> read_checkpoint"
    << "\n--> Invalid argument exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::ios_base::failure &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> read_checkpoint"
    << "\n--> IO System base failure exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::exception &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> read_checkpoint"
    << "\n--> Exception: " << e.what() << std::endl ;
    return false;
  }

return normal_exit; }


bool Parallel_Domain::adios2_write_checkpoint(const std::string filename)
{
  try
  {
    adios2::Engine writer = this->IO->write_checkpoint.Open(filename, adios2::Mode::Write);

    Vec2<lpUint> start = Vec2<lpUint>(this->origin_on_global_grid)  ;
    Vec2<lpUint> N     = Vec2<lpUint>(this->BSIM_var[0]->val->size) ;
    Vec2<lpUint> ptr   = Vec2<lpUint>(0,0) ;

    // Checkpoint also write edges
    if( this->id.x == 0 ){
      if( this->id.x == this->num_process.x-1 ){
        N.x += static_cast<lpUint>(2*this->shared_edge_size.x) ;
      }else{
        N.x += this->shared_edge_size.x ;
      }
    }else{
      start.x += this->shared_edge_size.x ;
      ptr.x   += this->shared_edge_size.x ;
      if( this->id.x == this->num_process.x-1 ){
        N.x   += this->shared_edge_size.x ;
      }
    }
    if( this->id.y == 0 ){
      if( this->id.y == this->num_process.y-1 ){
        N.y += static_cast<lpUint>(2*this->shared_edge_size.y) ;
      }else{
        N.y += this->shared_edge_size.y ;
      }
    }else{
      start.y += this->shared_edge_size.y ;
      ptr.y   += this->shared_edge_size.y ;
      if( this->id.y == this->num_process.y-1 ){
        N.y   += this->shared_edge_size.y ;
      }
    }

    // value
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->checkpoint_var.size()) ; ++i)
    {
      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->checkpoint_var[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->checkpoint_var[i],
                   &this->BSIM_var[i]->val->at_sw(ptr.x, static_cast<lpUint>(ptr.y+j)),
                   adios2::Mode::Deferred);
      }
    }

    // grad_x
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->checkpoint_grad_x.size()) ; ++i)
    {
      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->checkpoint_grad_x[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->checkpoint_grad_x[i],
                   &this->BSIM_var[i]->grad_x->at_sw(ptr.x, static_cast<lpUint>(ptr.y+j)),
                   adios2::Mode::Deferred);
      }
    }

    // grad_y
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->checkpoint_grad_y.size()) ; ++i)
    {
      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->checkpoint_grad_y[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->checkpoint_grad_y[i],
                   &this->BSIM_var[i]->grad_y->at_sw(ptr.x, static_cast<lpUint>(ptr.y+j)),
                   adios2::Mode::Deferred);
      }
    }

    writer.PerformPuts(); // Actual IO threading is managed by adios2_config_file.
    writer.Close();

  }
  catch (std::invalid_argument &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_checkpoint"
    << "\n--> Invalid argument exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::ios_base::failure &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_checkpoint"
    << "\n--> IO System base failure exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::exception &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_checkpoint"
    << "\n--> Exception: " << e.what() << std::endl ;
    return false;
  }

return true; }


bool Parallel_Domain::adios2_write_output(const std::string filename)
{

  try
  {
    adios2::Engine writer = this->IO->write_output.Open(filename, adios2::Mode::Write);

    Vec2<lpUint> start = Vec2<lpUint>(this->origin_on_global_grid);
    Vec2<lpUint> N = Vec2<lpUint>(this->BSIM_var[0]->val->size);

    // value
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->output_var.size()) ; ++i)
    {
      lpUint k = BSIM_OUTPUT::index[i] ;

      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->output_var[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->output_var[i],
                   &this->BSIM_var[k]->val->at(0,j),
                   adios2::Mode::Deferred);
      }
    }

    // grad_x
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->output_grad_x.size()) ; ++i)
    {
      lpUint k = BSIM_OUTPUT::index[i] ;

      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->output_grad_x[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->output_grad_x[i],
                   &this->BSIM_var[k]->grad_x->at(0,j),
                   adios2::Mode::Deferred);
      }
    }

    // grad_y
    for(lpUint i=0 ; i != static_cast<lpUint>(this->IO->output_grad_y.size()) ; ++i)
    {
      lpUint k = BSIM_OUTPUT::index[i] ;

      for(lpUint j=0 ; j != N.y ; ++j)
      {
        this->IO->output_grad_y[i].SetSelection({ {static_cast<lpUint>(j + start.y), start.x}, {1, N.x} });
        writer.Put(this->IO->output_grad_y[i],
                   &this->BSIM_var[k]->grad_y->at(0,j),
                   adios2::Mode::Deferred);
      }
    }

    writer.PerformPuts(); // Actual IO threading is managed by adios2_config_file.
    writer.Close();

  }
  catch (std::invalid_argument &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_output"
    << "\n--> Invalid argument exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::ios_base::failure &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_output"
    << "\n--> IO System base failure exception: " << e.what() << std::endl ;
    return false;
  }
  catch (std::exception &e)
  {
    std::cerr
    << "\nERROR while running BSIM::paralleldomain.cpp"
    << "\n--> write_output"
    << "\n--> Exception: " << e.what() << std::endl ;
    return false;
  }

return true; }


void Parallel_Domain::exchange_all_bc_periodic(const lpInt varid)
{
  // Array send
  this->BSIM_var[varid]->val->sendtobe_edge(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->grad_x->sendtobe_edge(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_y->sendtobe_edge(*this->sendtobe_edge[2]);

  // Communication
  // Send value
  MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[0]);
  MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[1]);
  MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[2]);
  MPI_Irecv( this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[3]);
  MPI_Irecv( this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[4]);
  MPI_Irecv( this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[5]);

  MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[6]);
  MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[7]);
  MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[8]);
  MPI_Irecv( this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[9]);
  MPI_Irecv( this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[10]);
  MPI_Irecv( this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[11]);

  MPI_Isend(this->sendtobe_edge[0]->nn, this->xx_size, MPI_Float, this->rank_ss, 6, MPI_COMM_WORLD, &mpi_request[12]);
  MPI_Isend(this->sendtobe_edge[1]->nn, this->xx_size, MPI_Float, this->rank_ss, 7, MPI_COMM_WORLD, &mpi_request[13]);
  MPI_Isend(this->sendtobe_edge[2]->nn, this->xx_size, MPI_Float, this->rank_ss, 8, MPI_COMM_WORLD, &mpi_request[14]);
  MPI_Irecv( this->receive_edge[0]->nn, this->xx_size, MPI_Float, this->rank_nn, 6, MPI_COMM_WORLD, &mpi_request[15]);
  MPI_Irecv( this->receive_edge[1]->nn, this->xx_size, MPI_Float, this->rank_nn, 7, MPI_COMM_WORLD, &mpi_request[16]);
  MPI_Irecv( this->receive_edge[2]->nn, this->xx_size, MPI_Float, this->rank_nn, 8, MPI_COMM_WORLD, &mpi_request[17]);

  MPI_Isend(this->sendtobe_edge[0]->ss, this->xx_size, MPI_Float, this->rank_nn,  9, MPI_COMM_WORLD, &mpi_request[18]);
  MPI_Isend(this->sendtobe_edge[1]->ss, this->xx_size, MPI_Float, this->rank_nn, 10, MPI_COMM_WORLD, &mpi_request[19]);
  MPI_Isend(this->sendtobe_edge[2]->ss, this->xx_size, MPI_Float, this->rank_nn, 11, MPI_COMM_WORLD, &mpi_request[20]);
  MPI_Irecv( this->receive_edge[0]->ss, this->xx_size, MPI_Float, this->rank_ss,  9, MPI_COMM_WORLD, &mpi_request[21]);
  MPI_Irecv( this->receive_edge[1]->ss, this->xx_size, MPI_Float, this->rank_ss, 10, MPI_COMM_WORLD, &mpi_request[22]);
  MPI_Irecv( this->receive_edge[2]->ss, this->xx_size, MPI_Float, this->rank_ss, 11, MPI_COMM_WORLD, &mpi_request[23]);

  MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[24]);
  MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[25]);
  MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[26]);
  MPI_Irecv( this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[27]);
  MPI_Irecv( this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[28]);
  MPI_Irecv( this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[29]);

  MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[30]);
  MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[31]);
  MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[32]);
  MPI_Irecv( this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[33]);
  MPI_Irecv( this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[34]);
  MPI_Irecv( this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[35]);

  MPI_Isend(this->sendtobe_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ww, 18, MPI_COMM_WORLD, &mpi_request[36]);
  MPI_Isend(this->sendtobe_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ww, 19, MPI_COMM_WORLD, &mpi_request[37]);
  MPI_Isend(this->sendtobe_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ww, 20, MPI_COMM_WORLD, &mpi_request[38]);
  MPI_Irecv( this->receive_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ee, 18, MPI_COMM_WORLD, &mpi_request[39]);
  MPI_Irecv( this->receive_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ee, 19, MPI_COMM_WORLD, &mpi_request[40]);
  MPI_Irecv( this->receive_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ee, 20, MPI_COMM_WORLD, &mpi_request[41]);

  MPI_Isend(this->sendtobe_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ee, 21, MPI_COMM_WORLD, &mpi_request[42]);
  MPI_Isend(this->sendtobe_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ee, 22, MPI_COMM_WORLD, &mpi_request[43]);
  MPI_Isend(this->sendtobe_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ee, 23, MPI_COMM_WORLD, &mpi_request[44]);
  MPI_Irecv( this->receive_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ww, 21, MPI_COMM_WORLD, &mpi_request[45]);
  MPI_Irecv( this->receive_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ww, 22, MPI_COMM_WORLD, &mpi_request[46]);
  MPI_Irecv( this->receive_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ww, 23, MPI_COMM_WORLD, &mpi_request[47]);
}


void Parallel_Domain::wait_all_bc_periodic(const lpInt varid)
{
  MPI_Waitall(48, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  this->BSIM_var[varid]->val->copy_edge_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->grad_x->copy_edge_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_y->copy_edge_from(*this->receive_edge[2]);

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


void Parallel_Domain::exchange_NS_bc_periodic(const lpInt varid)
{
  // Array send
  this->BSIM_var[varid]->val->sendtobe_sw(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_ss(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_se(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_nw(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_nn(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_ne(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->grad_x->sendtobe_sw(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_ss(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_se(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_nw(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_nn(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_ne(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_y->sendtobe_sw(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_ss(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_se(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_nw(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_nn(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_ne(*this->sendtobe_edge[2]);


  // Communication
  // Send value
  MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[0]);
  MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[1]);
  MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[2]);
  MPI_Irecv( this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[3]);
  MPI_Irecv( this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[4]);
  MPI_Irecv( this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[5]);

  MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[6]);
  MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[7]);
  MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[8]);
  MPI_Irecv( this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[9]);
  MPI_Irecv( this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[10]);
  MPI_Irecv( this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[11]);

  MPI_Isend(this->sendtobe_edge[0]->nn, this->xx_size, MPI_Float, this->rank_ss, 6, MPI_COMM_WORLD, &mpi_request[12]);
  MPI_Isend(this->sendtobe_edge[1]->nn, this->xx_size, MPI_Float, this->rank_ss, 7, MPI_COMM_WORLD, &mpi_request[13]);
  MPI_Isend(this->sendtobe_edge[2]->nn, this->xx_size, MPI_Float, this->rank_ss, 8, MPI_COMM_WORLD, &mpi_request[14]);
  MPI_Irecv( this->receive_edge[0]->nn, this->xx_size, MPI_Float, this->rank_nn, 6, MPI_COMM_WORLD, &mpi_request[15]);
  MPI_Irecv( this->receive_edge[1]->nn, this->xx_size, MPI_Float, this->rank_nn, 7, MPI_COMM_WORLD, &mpi_request[16]);
  MPI_Irecv( this->receive_edge[2]->nn, this->xx_size, MPI_Float, this->rank_nn, 8, MPI_COMM_WORLD, &mpi_request[17]);

  MPI_Isend(this->sendtobe_edge[0]->ss, this->xx_size, MPI_Float, this->rank_nn,  9, MPI_COMM_WORLD, &mpi_request[18]);
  MPI_Isend(this->sendtobe_edge[1]->ss, this->xx_size, MPI_Float, this->rank_nn, 10, MPI_COMM_WORLD, &mpi_request[19]);
  MPI_Isend(this->sendtobe_edge[2]->ss, this->xx_size, MPI_Float, this->rank_nn, 11, MPI_COMM_WORLD, &mpi_request[20]);
  MPI_Irecv( this->receive_edge[0]->ss, this->xx_size, MPI_Float, this->rank_ss,  9, MPI_COMM_WORLD, &mpi_request[21]);
  MPI_Irecv( this->receive_edge[1]->ss, this->xx_size, MPI_Float, this->rank_ss, 10, MPI_COMM_WORLD, &mpi_request[22]);
  MPI_Irecv( this->receive_edge[2]->ss, this->xx_size, MPI_Float, this->rank_ss, 11, MPI_COMM_WORLD, &mpi_request[23]);

  MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[24]);
  MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[25]);
  MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[26]);
  MPI_Irecv( this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[27]);
  MPI_Irecv( this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[28]);
  MPI_Irecv( this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[29]);

  MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[30]);
  MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[31]);
  MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[32]);
  MPI_Irecv( this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[33]);
  MPI_Irecv( this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[34]);
  MPI_Irecv( this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[35]);

  this->num_req = 36 ;

  if( this->id.x == 0 ){
    this->BSIM_var[varid]->val->sendtobe_ee(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ee(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ee(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ww, 18, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ww, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ww, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ww, 21, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ww, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ww, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }
  if( this->id.x == this->num_process.x-1 ){
    this->BSIM_var[varid]->val->sendtobe_ww(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ww(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ww(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ee, 21, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ee, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ee, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ee, 18, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ee, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ee, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }
}


void Parallel_Domain::wait_NS_bc_periodic(const lpInt varid)
{
  MPI_Waitall(this->num_req, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  this->BSIM_var[varid]->val->copy_sw_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_ss_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_se_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_nw_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_nn_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_ne_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->grad_x->copy_sw_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_ss_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_se_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_nw_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_nn_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_ne_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_y->copy_sw_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_ss_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_se_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_nw_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_nn_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_ne_from(*this->receive_edge[2]);

  if( this->id.x == 0 ){
    this->BSIM_var[varid]->val->copy_ww_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ww_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ww_from(*this->receive_edge[2]);
  }
  if( this->id.x == this->num_process.x-1 ){
    this->BSIM_var[varid]->val->copy_ee_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ee_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ee_from(*this->receive_edge[2]);
  }

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


void Parallel_Domain::exchange_EW_bc_periodic(const lpInt varid)
{
  // Array send
  this->BSIM_var[varid]->val->sendtobe_sw(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_se(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_ww(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_ee(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_nw(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->val->sendtobe_ne(*this->sendtobe_edge[0]);
  this->BSIM_var[varid]->grad_x->sendtobe_sw(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_se(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_ww(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_ee(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_nw(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_x->sendtobe_ne(*this->sendtobe_edge[1]);
  this->BSIM_var[varid]->grad_y->sendtobe_sw(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_se(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_ww(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_ee(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_nw(*this->sendtobe_edge[2]);
  this->BSIM_var[varid]->grad_y->sendtobe_ne(*this->sendtobe_edge[2]);


  // Communication
  // Send value
  MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[0]);
  MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[1]);
  MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[2]);
  MPI_Irecv( this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[3]);
  MPI_Irecv( this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[4]);
  MPI_Irecv( this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[5]);

  MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[6]);
  MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[7]);
  MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[8]);
  MPI_Irecv( this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[9]);
  MPI_Irecv( this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[10]);
  MPI_Irecv( this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[11]);

  MPI_Isend(this->sendtobe_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ww, 18, MPI_COMM_WORLD, &mpi_request[12]);
  MPI_Isend(this->sendtobe_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ww, 19, MPI_COMM_WORLD, &mpi_request[13]);
  MPI_Isend(this->sendtobe_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ww, 20, MPI_COMM_WORLD, &mpi_request[14]);
  MPI_Irecv( this->receive_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ee, 18, MPI_COMM_WORLD, &mpi_request[15]);
  MPI_Irecv( this->receive_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ee, 19, MPI_COMM_WORLD, &mpi_request[16]);
  MPI_Irecv( this->receive_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ee, 20, MPI_COMM_WORLD, &mpi_request[17]);

  MPI_Isend(this->sendtobe_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ee, 21, MPI_COMM_WORLD, &mpi_request[18]);
  MPI_Isend(this->sendtobe_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ee, 22, MPI_COMM_WORLD, &mpi_request[19]);
  MPI_Isend(this->sendtobe_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ee, 23, MPI_COMM_WORLD, &mpi_request[20]);
  MPI_Irecv( this->receive_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ww, 21, MPI_COMM_WORLD, &mpi_request[21]);
  MPI_Irecv( this->receive_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ww, 22, MPI_COMM_WORLD, &mpi_request[22]);
  MPI_Irecv( this->receive_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ww, 23, MPI_COMM_WORLD, &mpi_request[23]);

  MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[24]);
  MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[25]);
  MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[26]);
  MPI_Irecv( this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[27]);
  MPI_Irecv( this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[28]);
  MPI_Irecv( this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[29]);

  MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[30]);
  MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[31]);
  MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[32]);
  MPI_Irecv( this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[33]);
  MPI_Irecv( this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[34]);
  MPI_Irecv( this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[35]);

  this->num_req = 36 ;

  if( this->id.y == 0 ){
    this->BSIM_var[varid]->val->sendtobe_nn(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nn(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nn(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nn, this->xx_size, MPI_Float, this->rank_ss, 6, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nn, this->xx_size, MPI_Float, this->rank_ss, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nn, this->xx_size, MPI_Float, this->rank_ss, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ss, this->xx_size, MPI_Float, this->rank_ss,  9, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ss, this->xx_size, MPI_Float, this->rank_ss, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ss, this->xx_size, MPI_Float, this->rank_ss, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }
  if( this->id.y == this->num_process.y-1 ){
    this->BSIM_var[varid]->val->sendtobe_ss(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ss(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ss(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ss, this->xx_size, MPI_Float, this->rank_nn,  9, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ss, this->xx_size, MPI_Float, this->rank_nn, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ss, this->xx_size, MPI_Float, this->rank_nn, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nn, this->xx_size, MPI_Float, this->rank_nn, 6, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nn, this->xx_size, MPI_Float, this->rank_nn, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nn, this->xx_size, MPI_Float, this->rank_nn, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }
}


void Parallel_Domain::wait_EW_bc_periodic(const lpInt varid)
{
  MPI_Waitall(this->num_req, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  this->BSIM_var[varid]->val->copy_sw_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_se_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_ww_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_ee_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_nw_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->val->copy_ne_from(*this->receive_edge[0]);
  this->BSIM_var[varid]->grad_x->copy_sw_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_se_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_ww_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_ee_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_nw_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_x->copy_ne_from(*this->receive_edge[1]);
  this->BSIM_var[varid]->grad_y->copy_sw_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_se_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_ww_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_ee_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_nw_from(*this->receive_edge[2]);
  this->BSIM_var[varid]->grad_y->copy_ne_from(*this->receive_edge[2]);

  if( this->id.y == 0 ){
    this->BSIM_var[varid]->val->copy_ss_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ss_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ss_from(*this->receive_edge[2]);
  }
  if( this->id.y == this->num_process.y-1 ){
    this->BSIM_var[varid]->val->copy_nn_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nn_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nn_from(*this->receive_edge[2]);
  }

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


void Parallel_Domain::exchange_all_bc_external(const lpInt varid)
{
  this->num_req = 0 ;
  // If( not global BC)
  //   Array transfer + send + receive
  // else
  //   IO_read_external_BC()
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ne(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ne(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ne(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->sw) *** ### ***
    //std::cout << "External sw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_ne != -1 ){
    this->BSIM_var[varid]->val->sendtobe_sw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_sw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_sw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ne) *** ### ***
    //std::cout << "External ne of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_ss != -1 ){
    this->BSIM_var[varid]->val->sendtobe_nn(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nn(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nn(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nn, this->xx_size, MPI_Float, this->rank_ss, 6, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nn, this->xx_size, MPI_Float, this->rank_ss, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nn, this->xx_size, MPI_Float, this->rank_ss, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ss, this->xx_size, MPI_Float, this->rank_ss,  9, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ss, this->xx_size, MPI_Float, this->rank_ss, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ss, this->xx_size, MPI_Float, this->rank_ss, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ss) *** ### ***
    //std::cout << "External ss of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_nn != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ss(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ss(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ss(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ss, this->xx_size, MPI_Float, this->rank_nn,  9, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ss, this->xx_size, MPI_Float, this->rank_nn, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ss, this->xx_size, MPI_Float, this->rank_nn, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nn, this->xx_size, MPI_Float, this->rank_nn, 6, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nn, this->xx_size, MPI_Float, this->rank_nn, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nn, this->xx_size, MPI_Float, this->rank_nn, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->nn) *** ### ***
    //std::cout << "External nn of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->sendtobe_nw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->se) *** ### ***
    //std::cout << "External se of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_se(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_se(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_se(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->nw) *** ### ***
    //std::cout << "External nw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_ww != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ee(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ee(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ee(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ww, 18, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ww, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ww, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ww, 21, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ww, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ww, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ww) *** ### ***
    //std::cout << "External ww of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_ee != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ww(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ww(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ww(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ee, 21, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ee, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ee, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ee, 18, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ee, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ee, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ee) *** ### ***
    //std::cout << "External ee of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
}


void Parallel_Domain::wait_all_bc_external(const lpInt varid)
{
  MPI_Waitall(this->num_req, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  //this->BSIM_var[varid]->val->copy_edge_from(*this->receive_edge[0]);
  //this->BSIM_var[varid]->grad_x->copy_edge_from(*this->receive_edge[1]);
  //this->BSIM_var[varid]->grad_y->copy_edge_from(*this->receive_edge[2]);

  // Temporary
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->copy_sw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_sw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_sw_from(*this->receive_edge[2]);
  }
  if( this->rank_ss != -1 ){
    this->BSIM_var[varid]->val->copy_ss_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ss_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ss_from(*this->receive_edge[2]);
  }
  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->copy_se_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_se_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_se_from(*this->receive_edge[2]);
  }
  if( this->rank_ww != -1 ){
    this->BSIM_var[varid]->val->copy_ww_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ww_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ww_from(*this->receive_edge[2]);
  }
  if( this->rank_ee != -1 ){
    this->BSIM_var[varid]->val->copy_ee_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ee_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ee_from(*this->receive_edge[2]);
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->copy_nw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nw_from(*this->receive_edge[2]);
  }
  if( this->rank_nn != -1  ){
    this->BSIM_var[varid]->val->copy_nn_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nn_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nn_from(*this->receive_edge[2]);
  }
  if( this->rank_ne != -1  ){
    this->BSIM_var[varid]->val->copy_ne_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ne_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ne_from(*this->receive_edge[2]);
  }

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


void Parallel_Domain::exchange_NS_bc_external(const lpInt varid)
{
  this->num_req = 0 ;

  // If( not global BC)
  //   Array transfer + send + receive
  // else
  //   IO_read_external_BC()
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ne(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ne(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ne(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->sw) *** ### ***
    //std::cout << "External sw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_ne != -1 ){
    this->BSIM_var[varid]->val->sendtobe_sw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_sw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_sw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ne) *** ### ***
    //std::cout << "External ne of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_ss != -1 ){
    this->BSIM_var[varid]->val->sendtobe_nn(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nn(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nn(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nn, this->xx_size, MPI_Float, this->rank_ss, 6, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nn, this->xx_size, MPI_Float, this->rank_ss, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nn, this->xx_size, MPI_Float, this->rank_ss, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ss, this->xx_size, MPI_Float, this->rank_ss,  9, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ss, this->xx_size, MPI_Float, this->rank_ss, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ss, this->xx_size, MPI_Float, this->rank_ss, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ss) *** ### ***
    //std::cout << "External ss of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_nn != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ss(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ss(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ss(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ss, this->xx_size, MPI_Float, this->rank_nn,  9, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ss, this->xx_size, MPI_Float, this->rank_nn, 10, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ss, this->xx_size, MPI_Float, this->rank_nn, 11, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nn, this->xx_size, MPI_Float, this->rank_nn, 6, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nn, this->xx_size, MPI_Float, this->rank_nn, 7, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nn, this->xx_size, MPI_Float, this->rank_nn, 8, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->nn) *** ### ***
    //std::cout << "External nn of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->sendtobe_nw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->se) *** ### ***
    //std::cout << "External se of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_se(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_se(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_se(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->nw) *** ### ***
    //std::cout << "External nw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
}


void Parallel_Domain::wait_NS_bc_external(const lpInt varid)
{
  MPI_Waitall(this->num_req, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  //this->BSIM_var[varid]->val->copy_edge_from(*this->receive_edge[0]);
  //this->BSIM_var[varid]->grad_x->copy_edge_from(*this->receive_edge[1]);
  //this->BSIM_var[varid]->grad_y->copy_edge_from(*this->receive_edge[2]);

  // Temporary
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->copy_sw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_sw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_sw_from(*this->receive_edge[2]);
  }
  if( this->rank_ss != -1 ){
    this->BSIM_var[varid]->val->copy_ss_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ss_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ss_from(*this->receive_edge[2]);
  }
  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->copy_se_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_se_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_se_from(*this->receive_edge[2]);
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->copy_nw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nw_from(*this->receive_edge[2]);
  }
  if( this->rank_nn != -1  ){
    this->BSIM_var[varid]->val->copy_nn_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nn_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nn_from(*this->receive_edge[2]);
  }
  if( this->rank_ne != -1  ){
    this->BSIM_var[varid]->val->copy_ne_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ne_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ne_from(*this->receive_edge[2]);
  }

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


void Parallel_Domain::exchange_EW_bc_external(const lpInt varid)
{
  this->num_req = 0 ;

  // If( not global BC)
  //   Array transfer + send + receive
  // else
  //   IO_read_external_BC()
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ne(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ne(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ne(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ne, this->xy_size, MPI_Float, this->rank_sw, 0, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ne, this->xy_size, MPI_Float, this->rank_sw, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ne, this->xy_size, MPI_Float, this->rank_sw, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->sw, this->xy_size, MPI_Float, this->rank_sw, 3, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->sw, this->xy_size, MPI_Float, this->rank_sw, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->sw, this->xy_size, MPI_Float, this->rank_sw, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->sw) *** ### ***
    //std::cout << "External sw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_ne != -1 ){
    this->BSIM_var[varid]->val->sendtobe_sw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_sw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_sw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->sw, this->xy_size, MPI_Float, this->rank_ne, 3, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->sw, this->xy_size, MPI_Float, this->rank_ne, 4, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->sw, this->xy_size, MPI_Float, this->rank_ne, 5, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ne, this->xy_size, MPI_Float, this->rank_ne, 0, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ne, this->xy_size, MPI_Float, this->rank_ne, 1, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ne, this->xy_size, MPI_Float, this->rank_ne, 2, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ne) *** ### ***
    //std::cout << "External ne of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->sendtobe_nw(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_nw(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_nw(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->nw, this->xy_size, MPI_Float, this->rank_se, 12, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->nw, this->xy_size, MPI_Float, this->rank_se, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->nw, this->xy_size, MPI_Float, this->rank_se, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->se, this->xy_size, MPI_Float, this->rank_se, 15, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->se, this->xy_size, MPI_Float, this->rank_se, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->se, this->xy_size, MPI_Float, this->rank_se, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->se) *** ### ***
    //std::cout << "External se of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->sendtobe_se(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_se(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_se(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->se, this->xy_size, MPI_Float, this->rank_nw, 15, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->se, this->xy_size, MPI_Float, this->rank_nw, 16, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->se, this->xy_size, MPI_Float, this->rank_nw, 17, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->nw, this->xy_size, MPI_Float, this->rank_nw, 12, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->nw, this->xy_size, MPI_Float, this->rank_nw, 13, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->nw, this->xy_size, MPI_Float, this->rank_nw, 14, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->nw) *** ### ***
    //std::cout << "External nw of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }

  if( this->rank_ww != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ee(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ee(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ee(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ww, 18, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ww, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ww, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ww, 21, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ww, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ww, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ww) *** ### ***
    //std::cout << "External ww of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
  if( this->rank_ee != -1 ){
    this->BSIM_var[varid]->val->sendtobe_ww(*this->sendtobe_edge[0]);
    this->BSIM_var[varid]->grad_x->sendtobe_ww(*this->sendtobe_edge[1]);
    this->BSIM_var[varid]->grad_y->sendtobe_ww(*this->sendtobe_edge[2]);

    MPI_Isend(this->sendtobe_edge[0]->ww, this->yy_size, MPI_Float, this->rank_ee, 21, MPI_COMM_WORLD, &mpi_request[this->num_req]);
    MPI_Isend(this->sendtobe_edge[1]->ww, this->yy_size, MPI_Float, this->rank_ee, 22, MPI_COMM_WORLD, &mpi_request[this->num_req+1]);
    MPI_Isend(this->sendtobe_edge[2]->ww, this->yy_size, MPI_Float, this->rank_ee, 23, MPI_COMM_WORLD, &mpi_request[this->num_req+2]);

    MPI_Irecv(this->receive_edge[0]->ee, this->yy_size, MPI_Float, this->rank_ee, 18, MPI_COMM_WORLD, &mpi_request[this->num_req+3]);
    MPI_Irecv(this->receive_edge[1]->ee, this->yy_size, MPI_Float, this->rank_ee, 19, MPI_COMM_WORLD, &mpi_request[this->num_req+4]);
    MPI_Irecv(this->receive_edge[2]->ee, this->yy_size, MPI_Float, this->rank_ee, 20, MPI_COMM_WORLD, &mpi_request[this->num_req+5]);

    this->num_req += 6 ;
  }else{
    // IO_read_external_BC(this->receive_edge[0]->ee) *** ### ***
    //std::cout << "External ee of (" << this->id.x << "," << this->id.y << ")" << std::endl;
  }
}


void Parallel_Domain::wait_EW_bc_external(const lpInt varid)
{
  MPI_Waitall(this->num_req, mpi_request, MPI_STATUSES_IGNORE);

  // Array receive
  //this->BSIM_var[varid]->val->copy_edge_from(*this->receive_edge[0]);
  //this->BSIM_var[varid]->grad_x->copy_edge_from(*this->receive_edge[1]);
  //this->BSIM_var[varid]->grad_y->copy_edge_from(*this->receive_edge[2]);

  // Temporary
  if( this->rank_sw != -1 ){
    this->BSIM_var[varid]->val->copy_sw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_sw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_sw_from(*this->receive_edge[2]);
  }
  if( this->rank_se != -1 ){
    this->BSIM_var[varid]->val->copy_se_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_se_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_se_from(*this->receive_edge[2]);
  }
  if( this->rank_ww != -1 ){
    this->BSIM_var[varid]->val->copy_ww_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ww_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ww_from(*this->receive_edge[2]);
  }
  if( this->rank_ee != -1 ){
    this->BSIM_var[varid]->val->copy_ee_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ee_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ee_from(*this->receive_edge[2]);
  }
  if( this->rank_nw != -1 ){
    this->BSIM_var[varid]->val->copy_nw_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_nw_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_nw_from(*this->receive_edge[2]);
  }
  if( this->rank_ne != -1  ){
    this->BSIM_var[varid]->val->copy_ne_from(*this->receive_edge[0]);
    this->BSIM_var[varid]->grad_x->copy_ne_from(*this->receive_edge[1]);
    this->BSIM_var[varid]->grad_y->copy_ne_from(*this->receive_edge[2]);
  }

  MPI_Barrier(MPI_COMM_WORLD); // required, otherwise wrong at large MPI size
}


// -------------------------------------------------------------------


lpInt Parallel_Domain::rank_from_id_periodicBC(const lpInt idx, const lpInt idy)
{
  return static_cast<lpInt>( this->mpi_map[   this->modulo(idy, this->num_process.y)*this->num_process.x
                                            + this->modulo(idx, this->num_process.x) ]  );
}


lpInt Parallel_Domain::rank_from_id_externalBC(const lpInt idx, const lpInt idy)
{
  if( idx < 0 || idx >= this->num_process.x ){ return -1 ; }
  if( idy < 0 || idy >= this->num_process.y ){ return -1 ; }

  return static_cast<lpInt>( this->mpi_map[ idy*this->num_process.x + idx ] );
}


Vec2<lpInt> Parallel_Domain::compute_range_local_domain(const lpInt proc_id, const lpInt num_proc, const lpInt stride, const lpInt nx)
{
  lpInt remain = static_cast<lpInt>( nx - stride*num_proc ) ;

  lpInt left  = static_cast<lpInt>( (  proc_id<remain) ?     proc_id*(stride+1) :     proc_id*stride + remain ) ;
  lpInt right = static_cast<lpInt>( (proc_id+1<remain) ? (proc_id+1)*(stride+1) : (proc_id+1)*stride + remain ) ;

return Vec2<lpInt>(left, right) ; }


void Parallel_Domain::allocate_variable(const Vec2<lpInt> grid_size, const Vec2<lpInt> edge_size, const Vec2<Float> pos0, const Vec2<Float> spacing)
{
  for(lpInt i=0 ; i != BSIM_TOTAL_NUM_VAR ; ++i)
  {
    this->BSIM_var.push_back( new CubicGrid2D(grid_size, edge_size, pos0, spacing) );
  }

  for(lpInt i=0 ; i != 3 ; ++i)
  {
    this->sendtobe_edge.push_back( new Shared2D<Float>(grid_size, edge_size) );
    this->receive_edge.push_back( new Shared2D<Float>(grid_size, edge_size) );
  }
}


void Parallel_Domain::deallocate_variable()
{
  for(lpInt i=0 ; i != static_cast<lpInt>( this->BSIM_var.size() ) ; ++i)
  {
    delete this->BSIM_var[i] ;
  }

  for(lpInt i=0 ; i != static_cast<lpInt>( this->sendtobe_edge.size() ) ; ++i)
  {
    delete this->sendtobe_edge[i] ;
  }

  for(lpInt i=0 ; i != static_cast<lpInt>( this->receive_edge.size() ) ; ++i)
  {
    delete this->receive_edge[i] ;
  }

}


} // NAMESPACE BSIM
