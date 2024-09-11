/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  dynamics.cpp (main):
    Dynamics procedures of BSIM main loop

\*****************************************************************************/

#include "dynamics.hpp"

namespace BSIM_DYNAMICS
{

// dx_unequal = (x_unequal[j+1] - x_unequal[j-1])/2
BSIM::Vec2<Float> forwardX_advect(BSIM::CubicGrid2D *var,
                                  const Float local_x_depart, const lpInt jy,
                                  const Float dt, const Float dx_unequal, const Float u_grad, Float *temp)
{
  const lpInt iL = var->floor_local_x_to_ix<lpInt>(local_x_depart);
  const Float l  = local_x_depart - iL*var->spacing.x ;
  var->x_coeff(iL, jy, temp);

  const Float val  = BSIM::func::grid2d::compute_val(l, temp) * dx_unequal/var->spacing.x ;
  const Float grad = BSIM::func::grid2d::compute_grad(l, temp) * (1.0 - u_grad*dt) * dx_unequal/var->spacing.x ;

return BSIM::Vec2<Float>(val,grad); }


// dy_unequal = (y_unequal[j+1] - y_unequal[j-1])/2
BSIM::Vec2<Float> forwardY_advect(BSIM::CubicGrid2D *var,
                                  const lpInt ix, const Float local_y_depart,
                                  const Float dt, const Float dy_unequal, const Float v_grad, Float *temp)
{
  const lpInt jL = var->floor_local_y_to_jy<lpInt>(local_y_depart);
  const Float l  = local_y_depart - jL*var->spacing.y ;
  var->y_coeff(ix, jL, temp);

  const Float val  = BSIM::func::grid2d::compute_val(l, temp) * dy_unequal/var->spacing.y ;
  const Float grad = BSIM::func::grid2d::compute_grad(l, temp) * (1.0 - v_grad*dt) * dy_unequal/var->spacing.y ;

return BSIM::Vec2<Float>(val,grad); }


BSIM::Vec2<Float> forward_diffuse(const hpFloat Ddtx4, const Float spacing, const Float distance, Float *coeff)
{
  //var->x_coeff(ix, jy, temp);
  //var->y_coeff(ix, jy, temp);

  // Diffuse val
  const hpFloat val = func::integrate_PolydegNGaussian_c4(distance, spacing, Ddtx4, coeff) / std::sqrt(BSIM::CONST::PI*Ddtx4) ;

  // Diffuse grad_x
  coeff[4] = coeff[3] ;
  coeff[3] = coeff[2] + distance*coeff[3] ;
  coeff[2] = coeff[1] + distance*coeff[2] ;
  coeff[1] = coeff[0] + distance*coeff[1] ;
  coeff[0] =            distance*coeff[0] ;
  const hpFloat grad = func::integrate_PolydegNGaussian_c5(distance, spacing, Ddtx4, coeff) * 2.0/Ddtx4 / std::sqrt(BSIM::CONST::PI*Ddtx4);

return BSIM::Vec2<Float>(val,grad); }


void diffuse_in_hopfcole_cancelcommon(const hpFloat Kdtx4, const Float spacing, const Float distance, Float *temp, hpFloat *workspace)
{
  Float tmp = spacing*0.5 ;
  const hpFloat Const = std::exp( -0.5 *(temp[0] + (temp[1] + (temp[2] + (temp[3] + (temp[4] + temp[5]*tmp)*tmp )*tmp )*tmp )*tmp ) ) ;

  temp[1] = temp[1] + (temp[2] + (1.5*temp[3] + (temp[4] + 0.625*temp[5]*spacing)*spacing )*tmp )*spacing ;
  temp[2] = temp[2] + (1.2*temp[3] + (1.2*temp[4] + temp[5]*spacing)*spacing )*spacing*1.25 ;
  temp[3] = temp[3] + (temp[4] + 1.25*temp[5]*spacing)*spacing*2 ;
  temp[4] = temp[4] + 2.5*temp[5]*spacing ;
  //temp[5] = temp[5] ;

  // ------------------------

  workspace[10] = 1.0 ;

  workspace[11] = -0.5*temp[1] ;

  workspace[12] = 0.125*( temp[1]*temp[1]  -  4.*temp[2] ) ;

  workspace[13] = 0.0625*( - temp[1]*temp[1]/3  +  4.*temp[2] )*temp[1] ;
  workspace[13] += - 0.5*temp[3] ;

  workspace[14] = 0.0078125*( temp[1]*temp[1]/3  -  8.*temp[2] )*temp[1]*temp[1] ;
  workspace[14] += 0.125*( temp[2]*temp[2]  +  2.*temp[1]*temp[3]  -  4.*temp[4] ) ;

  workspace[15] = (0.00078125/3)*( - temp[1]*temp[1] + 40*temp[2] )*temp[1]*temp[1]*temp[1] ;
  workspace[15] += (0.0625)*( - temp[1]*temp[3] - temp[2]*temp[2] + 4*temp[4] )*temp[1] ;
  workspace[15] += 0.25*temp[2]*temp[3]  -  0.5*temp[5] ;

  workspace[16] = 0.0625*( ( 0.0625*( 0.05/3*temp[1]*temp[1] - temp[2] )*temp[1] + 0.5*temp[3] )*temp[1]/3 - temp[4] )*temp[1]*temp[1] ;
  workspace[16] += 0.015625*( temp[1]*temp[1] - (4/3)*temp[2] )*temp[2]*temp[2] ;
  workspace[16] += 0.25*( 0.5*( - temp[2]*temp[3] + 2*temp[5] )*temp[1] + ( 0.5*temp[3]*temp[3] + temp[2]*temp[4] ) ) ;

  workspace[17] = ( 0.000390625*( ( (-0.25/21)*temp[1]*temp[1] + temp[2] )*temp[1]*temp[1] + 10.*( - temp[1]*temp[3] + 8.*temp[4] ) )*temp[1]/3 + 0.03125*( (-0.25/3)*temp[1]*temp[2] + temp[3] )*temp[2] )*temp[1]*temp[1] ;
  workspace[17] += 0.0625*( ( (0.5/3)*temp[1]*temp[2] - temp[3] )*temp[2]*temp[2] - ( temp[1]*temp[5] + 2*temp[2]*temp[4] + temp[3]*temp[3] )*temp[1] ) ;
  workspace[17] += 0.25*( temp[3]*temp[4] + temp[2]*temp[5] ) ;

  // ------------------------

  //workspace[0-7] = workspace[10-17]
  // P
  workspace[0] = Const*func::integrate_PolydegNGaussian_c8_mid(distance+tmp, tmp, Kdtx4, &workspace[10]) ;

  hpFloat tmp1 = tmp + distance ;
  workspace[1] =                 tmp1*workspace[10] ;
  workspace[2] = workspace[10] + tmp1*workspace[11] ;
  workspace[3] = workspace[11] + tmp1*workspace[12] ;
  workspace[4] = workspace[12] + tmp1*workspace[13] ;
  workspace[5] = workspace[13] + tmp1*workspace[14] ;
  workspace[6] = workspace[14] + tmp1*workspace[15] ;
  workspace[7] = workspace[15] + tmp1*workspace[16] ;
  workspace[8] = workspace[16] + tmp1*workspace[17] ;
  workspace[9] = workspace[17] ;

  // Px
  workspace[1] = Const*func::integrate_PolydegNGaussian_c9_mid(distance+tmp, tmp, Kdtx4, &workspace[1]) ;

  const hpFloat tmp2 = tmp1*tmp1 - Kdtx4/2 ;
  tmp1 *= 2 ;
  workspace[2]  =                                      tmp2*workspace[10] ;
  workspace[3]  =                 tmp1*workspace[10] + tmp2*workspace[11] ;
  workspace[4]  = workspace[10] + tmp1*workspace[11] + tmp2*workspace[12] ;
  workspace[5]  = workspace[11] + tmp1*workspace[12] + tmp2*workspace[13] ;
  workspace[6]  = workspace[12] + tmp1*workspace[13] + tmp2*workspace[14] ;
  workspace[7]  = workspace[13] + tmp1*workspace[14] + tmp2*workspace[15] ;
  workspace[8]  = workspace[14] + tmp1*workspace[15] + tmp2*workspace[16] ;
  workspace[9]  = workspace[15] + tmp1*workspace[16] + tmp2*workspace[17] ;
  workspace[10] = workspace[16] + tmp1*workspace[17] ;
  workspace[11] = workspace[17] ;

  // Pxx
  workspace[2] = Const*func::integrate_PolydegNGaussian_c10_mid(distance+tmp, tmp, Kdtx4, &workspace[2]) ;

return; }


// -------------------------------------------

namespace func
{

hpFloat integrate_PolydegNGaussian_c4(const hpFloat x0, const hpFloat l, const hpFloat b, const Float *cN)
{
  hpFloat sol0, sol1, sol2 ;

  // 1 --- Integration of c0*exp(-(y+x0)^2/b)
  sol0 =  0.5*std::sqrt(BSIM::CONST::PI*b) * (std::erf( (x0+l)/std::sqrt(b) ) - std::erf( x0/std::sqrt(b) )) ;

  // y --- Integration of c1*y*exp(-(y+x0)^2/b)
  sol1 = -sol0*x0 - 0.5*b * (std::exp( - std::pow(x0+l,2)/b ) - std::exp( - std::pow(x0,2)/b )) ;

  // y**k where k=2 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
  hpFloat temp = l*std::exp( - std::pow(x0+l,2)/b ) ;
  sol2 = 0.5*b*(sol0 - temp) - x0*sol1 ;

  // y**k where k=3 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
  //sol3 = 0.5*b*(2*sol1 - l*temp) - x0*sol2 ;

return cN[3]*(0.5*b*(2*sol1 - l*temp) - x0*sol2) + cN[2]*sol2 + cN[1]*sol1 + cN[0]*sol0 ; } // Sum from small to large --> avoid round-off error


hpFloat integrate_PolydegNGaussian_c5(const hpFloat x0, const hpFloat l, const hpFloat b, const Float *cN)
{
  hpFloat sol0, sol1, sol2, sol3 ;

  // 1 --- Integration of c0*exp(-(y+x0)^2/b)
  sol0 =  0.5*std::sqrt(BSIM::CONST::PI*b) * (std::erf( (x0+l)/std::sqrt(b) ) - std::erf( x0/std::sqrt(b) )) ;

  // y --- Integration of c1*y*exp(-(y+x0)^2/b)
  sol1 = -sol0*x0 - 0.5*b * (std::exp( - std::pow(x0+l,2)/b ) - std::exp( - std::pow(x0,2)/b )) ;

  // y**k where k=2 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
  hpFloat temp = l*std::exp( - std::pow(x0+l,2)/b ) ;
  sol2 = 0.5*b*(sol0 - temp) - x0*sol1 ;

  // y**k where k=3 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
  sol3 = 0.5*b*(2*sol1 - l*temp) - x0*sol2 ;

  // y**k where k=4 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = 0 to l ONLY ***)
  //sol4 = 0.5*b*(3*sol2 - l*l*temp) - x0*sol3 ;

return cN[4]*(0.5*b*(3*sol2 - l*l*temp) - x0*sol3) + cN[3]*sol3 + cN[2]*sol2 + cN[1]*sol1 + cN[0]*sol0 ; } // Sum from small to large --> avoid round-off error


hpFloat integrate_PolydegNGaussian_c8_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN)
{
  hpFloat sol0, sol1, sol2, sol3, sol4, sol5, sol6 ;

  // 1 --- Integration of c0*exp(-(y+x0)^2/b)
  sol0 =  0.5*std::sqrt(BSIM::CONST::PI*b) * (std::erf( (x0+l)/std::sqrt(b) ) - std::erf( (x0-l)/std::sqrt(b) )) ;

  // y --- Integration of c1*y*exp(-(y+x0)^2/b)
  sol1 = - sol0*x0 - 0.5*b * (std::exp( - std::pow(x0+l,2)/b ) - std::exp( - std::pow(x0-l,2)/b )) ;

  // y**k where k=2 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  hpFloat temp_p = l*std::exp( - std::pow(x0+l,2)/b ) ;
  hpFloat temp_n = l*std::exp( - std::pow(x0-l,2)/b ) ;
  sol2 = 0.5*b*(sol0 - temp_p - temp_n) - x0*sol1 ;

  // y**k where k=3 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol3 = 0.5*b*(2*sol1 - temp_p - temp_n) - x0*sol2 ;

  // y**k where k=4 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol4 = 0.5*b*(3*sol2 - temp_p - temp_n) - x0*sol3 ;

  // y**k where k=5 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol5 = 0.5*b*(4*sol3 - temp_p - temp_n) - x0*sol4 ;

  // y**k where k=6 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol6 = 0.5*b*(5*sol4 - temp_p - temp_n) - x0*sol5 ;

  // y**k where k>1 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  //sol[k+2] +=  0.5*b*(  (k+1)*sol[k] - l**k*temp_p - (-l)**k*temp_n)  ) - x0*sol[k+1]
  temp_p *=  l ;
  temp_n *= -l ;

return cN[7]*(0.5*b*(6*sol5 - temp_p - temp_n) - x0*sol6) + cN[6]*sol6 + cN[5]*sol5 + cN[4]*sol4
                                                          + cN[3]*sol3 + cN[2]*sol2 + cN[1]*sol1
                                                          + cN[0]*sol0 ; } // Sum from small to large --> avoid round-off error


hpFloat integrate_PolydegNGaussian_c9_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN)
{
  hpFloat sol0, sol1, sol2, sol3, sol4, sol5, sol6, sol7 ;

  // 1 --- Integration of c0*exp(-(y+x0)^2/b)
  sol0 =  0.5*std::sqrt(BSIM::CONST::PI*b) * (std::erf( (x0+l)/std::sqrt(b) ) - std::erf( (x0-l)/std::sqrt(b) )) ;

  // y --- Integration of c1*y*exp(-(y+x0)^2/b)
  sol1 = - sol0*x0 - 0.5*b * (std::exp( - std::pow(x0+l,2)/b ) - std::exp( - std::pow(x0-l,2)/b )) ;

  // y**k where k=2 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  hpFloat temp_p = l*std::exp( - std::pow(x0+l,2)/b ) ;
  hpFloat temp_n = l*std::exp( - std::pow(x0-l,2)/b ) ;
  sol2 = 0.5*b*(sol0 - temp_p - temp_n) - x0*sol1 ;

  // y**k where k=3 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol3 = 0.5*b*(2*sol1 - temp_p - temp_n) - x0*sol2 ;

  // y**k where k=4 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol4 = 0.5*b*(3*sol2 - temp_p - temp_n) - x0*sol3 ;

  // y**k where k=5 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol5 = 0.5*b*(4*sol3 - temp_p - temp_n) - x0*sol4 ;

  // y**k where k=6 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol6 = 0.5*b*(5*sol4 - temp_p - temp_n) - x0*sol5 ;

  // y**k where k=7 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol7 = 0.5*b*(6*sol5 - temp_p - temp_n) - x0*sol6 ;

  // y**k where k>1 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  //sol[k+2] +=  0.5*b*(  (k+1)*sol[k] - l**k*temp_p - (-l)**k*temp_n)  ) - x0*sol[k+1]
  temp_p *=  l ;
  temp_n *= -l ;

return cN[8]*(0.5*b*(7*sol6 - temp_p - temp_n) - x0*sol7) + cN[7]*sol7 + cN[6]*sol6 + cN[5]*sol5
                                                          + cN[4]*sol4 + cN[3]*sol3 + cN[2]*sol2
                                                          + cN[1]*sol1 + cN[0]*sol0 ; } // Sum from small to large --> avoid round-off error


hpFloat integrate_PolydegNGaussian_c10_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN)
{
  hpFloat sol0, sol1, sol2, sol3, sol4, sol5, sol6, sol7, sol8 ;

  // 1 --- Integration of c0*exp(-(y+x0)^2/b)
  sol0 =  0.5*std::sqrt(BSIM::CONST::PI*b) * (std::erf( (x0+l)/std::sqrt(b) ) - std::erf( (x0-l)/std::sqrt(b) )) ;

  // y --- Integration of c1*y*exp(-(y+x0)^2/b)
  sol1 = - sol0*x0 - 0.5*b * (std::exp( - std::pow(x0+l,2)/b ) - std::exp( - std::pow(x0-l,2)/b )) ;

  // y**k where k=2 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  hpFloat temp_p = l*std::exp( - std::pow(x0+l,2)/b ) ;
  hpFloat temp_n = l*std::exp( - std::pow(x0-l,2)/b ) ;
  sol2 = 0.5*b*(sol0 - temp_p - temp_n) - x0*sol1 ;

  // y**k where k=3 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol3 = 0.5*b*(2*sol1 - temp_p - temp_n) - x0*sol2 ;

  // y**k where k=4 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol4 = 0.5*b*(3*sol2 - temp_p - temp_n) - x0*sol3 ;

  // y**k where k=5 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol5 = 0.5*b*(4*sol3 - temp_p - temp_n) - x0*sol4 ;

  // y**k where k=6 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol6 = 0.5*b*(5*sol4 - temp_p - temp_n) - x0*sol5 ;

  // y**k where k=7 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol7 = 0.5*b*(6*sol5 - temp_p - temp_n) - x0*sol6 ;

  // y**k where k=8 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  temp_p *=  l ;
  temp_n *= -l ;
  sol8 = 0.5*b*(7*sol6 - temp_p - temp_n) - x0*sol7 ;

  // y**k where k>1 --- Integration of ck**(y**k)*exp(-(y+x0)^2/b) from xi to xi+l (local y = -l to l ONLY ***)
  //sol[k+2] +=  0.5*b*(  (k+1)*sol[k] - l**k*temp_p - (-l)**k*temp_n)  ) - x0*sol[k+1]
  temp_p *=  l ;
  temp_n *= -l ;

return cN[9]*(0.5*b*(8*sol7 - temp_p - temp_n) - x0*sol8) + cN[8]*sol8 + cN[7]*sol7 + cN[6]*sol6
                                                          + cN[5]*sol5 + cN[4]*sol4 + cN[3]*sol3
                                                          + cN[2]*sol2 + cN[1]*sol1 + cN[0]*sol0 ; } // Sum from small to large --> avoid round-off error


} // NAMESPACE FUNC


} // NAMESPACE BSIM_DYNAMICS
