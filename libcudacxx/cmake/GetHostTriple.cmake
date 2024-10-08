# Returns the host triple.
# Invokes config.guess

function( get_host_triple var )
  if( MSVC )
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      set( value "x86_64-pc-windows-msvc" )
    else()
      set( value "i686-pc-windows-msvc" )
    endif()
  elseif( MINGW AND NOT MSYS )
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      set( value "x86_64-w64-windows-gnu" )
    else()
      set( value "i686-pc-windows-gnu" )
    endif()
  else( MSVC )
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL Windows AND NOT MSYS)
      message(WARNING "unable to determine host target triple")
    else()
      set(config_guess ${LLVM_PATH}/cmake/config.guess)
      execute_process(COMMAND sh ${config_guess}
        RESULT_VARIABLE TT_RV
        OUTPUT_VARIABLE TT_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if( NOT TT_RV EQUAL 0 )
        message(FATAL_ERROR "Failed to execute ${config_guess}")
      endif( NOT TT_RV EQUAL 0 )
      set( value ${TT_OUT} )
    endif()
  endif( MSVC )
  set( ${var} ${value} PARENT_SCOPE )
endfunction( get_host_triple var )
