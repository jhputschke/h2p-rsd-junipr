cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(h2p_rsd_junipr_cpp LANGUAGES CXX)

# C++17 to match ROOT 6.36 / FastJet.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC -Wno-deprecated-declarations")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules")

# ---- ROOT (>= 6.36, RNTuple) -------------------------------------------------
execute_process(COMMAND root-config --prefix OUTPUT_VARIABLE ROOT_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(ROOT_PREFIX)
  list(APPEND CMAKE_PREFIX_PATH "${ROOT_PREFIX}")
endif()
find_package(ROOT CONFIG REQUIRED COMPONENTS RIO Tree ROOTNTuple)
message(STATUS "ROOT ${ROOT_VERSION} found")

# ---- FastJet + fjcontrib LundPlane ------------------------------------------
find_package(FASTJET REQUIRED)
find_library(LUNDPLANE_LIB NAMES LundPlane
             HINTS ${FASTJET_LIBS_DIR} /opt/homebrew/lib /usr/local/lib)
if(NOT LUNDPLANE_LIB)
  message(FATAL_ERROR "fjcontrib LundPlane library not found (libLundPlane).")
endif()
# fjcontrib headers may live beside the lib (homebrew installs them in /opt/homebrew/include).
find_path(FJCONTRIB_INCLUDE_DIR fastjet/contrib/LundGenerator.hh
          HINTS ${FASTJET_INCLUDE_DIR} /opt/homebrew/include /usr/local/include)
if(NOT FJCONTRIB_INCLUDE_DIR)
  message(FATAL_ERROR "fjcontrib LundPlane headers not found (LundGenerator.hh).")
endif()
list(APPEND FASTJET_INCLUDE_DIR ${FJCONTRIB_INCLUDE_DIR})
message(STATUS "LundPlane: ${LUNDPLANE_LIB} (headers: ${FJCONTRIB_INCLUDE_DIR})")

# ---- PYTHIA 8 (optional) -----------------------------------------------------
find_package(Pythia8)

# ---- libraries ---------------------------------------------------------------
add_library(lund_io STATIC src/lund_io.cpp)
target_include_directories(lund_io PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${FASTJET_INCLUDE_DIR})

add_library(lund_writer STATIC src/lund_writer.cpp)
target_include_directories(lund_writer PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${FASTJET_INCLUDE_DIR})
target_link_libraries(lund_writer PUBLIC lund_io ROOT::RIO ROOT::Tree ROOT::ROOTNTuple)

# ---- writer binary -----------------------------------------------------------
add_executable(write_lund_rntuple src/write_lund_rntuple.cpp)
target_link_libraries(write_lund_rntuple PRIVATE lund_io lund_writer ${LUNDPLANE_LIB} ${FASTJET_CONFIGLIBS})
if(PYTHIA8_FOUND)
  target_compile_definitions(write_lund_rntuple PRIVATE HAVE_PYTHIA8)
  target_include_directories(write_lund_rntuple PRIVATE ${PYTHIA8_INCLUDE_DIRS})
  target_link_libraries(write_lund_rntuple PRIVATE ${PYTHIA8_LIBRARIES})
  message(STATUS "write_lund_rntuple: PYTHIA event source enabled")
else()
  message(STATUS "write_lund_rntuple: PYTHIA not found, using toy event source")
endif()

# ---- dedicated PYTHIA driver (only when PYTHIA is available) ------------------
if(PYTHIA8_FOUND)
  add_executable(pythia_driver apps/pythia_driver.cpp)
  target_include_directories(pythia_driver PRIVATE ${PYTHIA8_INCLUDE_DIRS})
  target_link_libraries(pythia_driver PRIVATE lund_io lund_writer ${LUNDPLANE_LIB}
                        ${FASTJET_CONFIGLIBS} ${PYTHIA8_LIBRARIES})
endif()

# ---- tests -------------------------------------------------------------------
enable_testing()
add_executable(test_lund_io tests/test_lund_io.cpp)
target_link_libraries(test_lund_io PRIVATE lund_io ${LUNDPLANE_LIB} ${FASTJET_CONFIGLIBS})
add_test(NAME lund_io COMMAND test_lund_io)
