/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  datetime.cpp (main):
    date time in YYYY-MM-DD_HH-MM-SS and related functions

\*****************************************************************************/

#include "datetime.hpp"

namespace BSIM
{

DateTime::DateTime(const bool is_date): year(0), month(0), day(0), hour(0), minute(0), second(0), isDate(is_date)
{

}


DateTime::DateTime(const Uint yy, const Uint mm, const Uint dd, const Uint hr, const Uint min, const Uint sec,
                   const bool is_date):
          year(yy), month(mm), day(dd), hour(hr), minute(min), second(sec), isDate(is_date)
{
  this->recheck();
}


DateTime::~DateTime()
{

}


bool DateTime::operator== (const DateTime &other)
{
  if( this->year   != other.year   ){ return false ; }
  if( this->month  != other.month  ){ return false ; }
  if( this->day    != other.day    ){ return false ; }
  if( this->hour   != other.hour   ){ return false ; }
  if( this->minute != other.minute ){ return false ; }
  if( this->second != other.second ){ return false ; }

return true; }


bool DateTime::operator<= (const DateTime &other)
{
  if( this->year > other.year ){
    return false ;
  }else if( this->year == other.year ){
    if( this->month > other.month )
    {
      return false ;
    }else if( this->month == other.month ){
      if( this->day > other.day ){
        return false ;
      }else if( this->day == other.day ){
        if( this->hour > other.hour ){
          return false ;
        }else if( this->hour == other.hour ){
          if( this->minute > other.minute ){
            return false ;
          }else if( this->minute == other.minute ){
            if( this->second > other.second ){ return false ; }
          }
        }
      }
    }
  }

return true; }


void DateTime::operator+= (const DateTime &other)
{
  if( other.year != 0 || other.month !=0 ){
    std::cout
    << "\nWARNING while running BSIM::datetime.cpp \n"
    << "--> increasing datetime by YEAR or MONTH could lead to incorrect result.\n" ;
  }

  this->year   += other.year   ;
  this->month  += other.month  ;
  this->day    += other.day    ;
  this->hour   += other.hour   ;
  this->minute += other.minute ;
  this->second += other.second ;

  this->recheck();
}


void DateTime::operator+= (const Uint added_second)
{
  this->second += added_second ;
  this->recheck();
}


bool DateTime::from_string(const std::string &str_in)
{
  size_t mid = str_in.find('_') ;
  std::string date = str_in.substr(0, mid);
  std::string time = str_in.substr(mid+1);

  if( this->isDate)
  {
    // year
    mid = date.find('-');
    if( 0 < mid && mid < date.length() ){
      year = str2Uint( date.substr(0,mid) );
      date = date.substr(mid+1);
    }else{
      std::cerr
      << "\nError while running BSIM::datetime.cpp \n"
      << "--> Cannot parse YEAR from " << str_in << "\n" ;
      return false;
    }

    // month
    mid = date.find('-');
    if( 0 < mid && mid < date.length() ){
      month = str2Uint( date.substr(0,mid) );
      date = date.substr(mid+1);
    }else{
      std::cerr
      << "\nError while running BSIM::datetime.cpp \n"
      << "--> Cannot parse MONTH from " << str_in << "\n" ;
      return false;
    }
  }

  // day
  if( date.length() != 0 ){
    day = str2Uint( date );
  }else{
    std::cerr
    << "\nError while running BSIM::datetime.cpp \n"
    << "--> Cannot parse DAY from " << str_in << "\n" ;
    return false;
  }

  // hour
  mid = time.find('-');
  if( 0 < mid && mid < time.length() ){
    hour = str2Uint( time.substr(0,mid) );
    time = time.substr(mid+1);
  }else{
    std::cerr
    << "\nError while running BSIM::datetime.cpp \n"
    << "--> Cannot parse HOUR from " << str_in << "\n" ;
    return false;
  }

  // minute
  mid = time.find('-');
  if( 0 < mid && mid < time.length() ){
    minute = str2Uint( time.substr(0,mid) );
    time = time.substr(mid+1);
  }else{
    std::cerr
    << "\nError while running BSIM::datetime.cpp \n"
    << "--> Cannot parse MINUTE from " << str_in << "\n" ;
    return false;
  }

  // second
  if( time.length() != 0 ){
    second = str2Uint( time );
  }else{
    std::cerr
    << "\nError while running BSIM::datetime.cpp \n"
    << "--> Cannot parse SECOND from " << str_in << "\n" ;
    return false;
  }

  this->recheck();

return true; }


const std::string DateTime::to_str()
{
  std::string timestr ;

  if( this->isDate )
  {
    timestr = std::to_string(this->year) + '-' ;

    if( this->month < 10 )
      timestr += '0' + std::to_string(this->month) + '-' ;
    else
      timestr += std::to_string(this->month) + '-' ;
  }

  if( this->day < 10 )
    timestr += '0' + std::to_string(this->day) + '_' ;
  else
    timestr += std::to_string(this->day) + '_' ;

  if( this->hour < 10 )
    timestr += '0' + std::to_string(this->hour) + '-' ;
  else
    timestr += std::to_string(this->hour) + '-' ;

  if( this->minute < 10 )
    timestr += '0' + std::to_string(this->minute) + '-' ;
  else
    timestr += std::to_string(this->minute) + '-' ;

  if( this->second < 10 )
    timestr += '0' + std::to_string(this->second) ;
  else
    timestr += std::to_string(this->second) ;

return timestr; }


hpUint DateTime::to_second()
{
  if( isDate ){
      std::cout
      << "\nWARNING while running BSIM::datetime.cpp \n"
      << "--> to_second() is issued while isDate = True.\n" ;
  }

  hpUint duration = static_cast<hpUint>(this->second) ;
  duration += static_cast<hpUint>(this->minute)*60 ;
  duration += static_cast<hpUint>(this->hour)*60*60 ;
  duration += static_cast<hpUint>(this->day)*24*60*60 ;

return duration ; }


void DateTime::recheck()
{
  Uint remain = this->second % 60 ;
  this->minute += (this->second - remain)/60 ;
  this->second = remain ;

  remain = this->minute % 60 ;
  this->hour += (this->minute - remain)/60 ;
  this->minute = remain ;

  remain = this->hour % 24 ;
  this->day += (this->hour - remain)/24 ;
  this->hour = remain ;

  if( this->isDate ){
    // Pre-Months to Days
    const Uint nDay[11] = {31,
                          (this->year%4 == 0) ? static_cast<Uint>(29) : static_cast<Uint>(28),
                           31, 30, 31, 30, 31, 31, 30, 31, 30} ;

    this->day = this->day - 1 ;  // date (timestamp) to days (amount of time)
    for(Uint i=0 ; i != 11 ; ++i)
    {
      if( this->month > i+1 ){
        this->day += nDay[i] ;
      }
    }

    //Days to Year
    while( this->day >= 365 )
    {
      if( this->year % 4 == 0 )
      {
        if( this->day >= 366 ){
          this->day  -= 366 ;
          this->year +=  1  ;
        }else{
          break;
        }
      }else{
        if( day >= 365 )
        {
          this->day  -= 365 ;
          this->year +=  1  ;
        }else{
          break;
        }
      }
    }

    // Days to Month
    this->month = 1 ;
    for(Uint i=0 ; i != 11 ; ++i)
    {
      if( this->day >= nDay[i] )
      {
        this->day   -= nDay[i] ;
        this->month +=   1     ;
      }else{
        break;
      }
    }
    this->day = this->day + 1 ; // days (amount of time) to date (timestamp)

  }else{

    if( this->year != 0 || this->month !=0 ){
      std::cout
      << "\nCAUTION while running BSIM::datetime.cpp \n"
      << "--> time interval with non-zero YEAR and/or MONTH is ambiguous, BSIM will set them to zero." << std::endl ;
    }
    this->year  = 0 ;
    this->month = 0 ;
  }

}

} // NAMESPACE BSIM

