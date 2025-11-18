#pragma once

#if defined(FEATURE_BABY_BEAR)
  #include "ff/bb31_t.hpp"
  #include <ff/bb31_septic_extension_t.hpp>
using field_t = bb31_t;
using septic_curve_t = bb31_septic_curve_t;
using septic_extension_t = bb31_septic_extension_t;
#elif defined(FEATURE_KOALA_BEAR)
  #include "ff/kb31_t.hpp"
  #include <ff/kb31_septic_extension_t.hpp>
using field_t = kb31_t;
using septic_curve_t = kb31_septic_curve_t;
using septic_extension_t = kb31_septic_extension_t;
#else
  #error "no FEATURE"
#endif