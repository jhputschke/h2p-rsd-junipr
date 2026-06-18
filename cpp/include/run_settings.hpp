// run_settings.hpp — drive the FastJet/grooming parameters from the same PYTHIA
// command card as the generator settings, instead of hardcoding them.
//
// PYTHIA's Settings database accepts *custom* entries registered before init, so
// one `.cmnd` card configures the generator (beams, process, MPI, seed, tune) AND
// the analysis (jet R/ptMin/acceptance/match cone, Soft Drop z_cut/beta/R0/kt
// floor, generator tag). No yaml-cpp, no bespoke parser, native comment handling.
// The parsed grooming params are written into the RNTuple provenance, so the
// recorded `z_cut/beta/kt_floor/generator` always match what produced the file.
//
//   registerAnalysisSettings(pythia);   // BEFORE readFile/init — defaults + schema
//   ... pythia.readFile(card); pythia.init(); ...
//   JetParams   jp  = readJetParams(pythia);
//   GroomParams g   = readGroomParams(pythia);
//   std::string gen = generatorTag(pythia);
#pragma once

#ifdef HAVE_PYTHIA8

#include "lund_io.hpp"

#include "Pythia8/Pythia.h"

#include <string>

namespace h2p {

// Register the analysis knobs (with the previous hardcoded values as defaults) so
// a card may override any of them, and Pythia validates the names/ranges.
inline void registerAnalysisSettings(Pythia8::Pythia& pythia) {
  Pythia8::Settings& s = pythia.settings;
  //              key                    default  hasMin hasMax  min     max
  s.addParm("HadronJet:R", 0.4, true, false, 0.0, 0.0);
  s.addParm("HadronJet:ptMin", 20.0, true, false, 0.0, 0.0);
  s.addParm("HadronJet:maxRap", 2.0, true, false, 0.0, 0.0);
  s.addParm("HadronJet:matchdR", 0.3, true, false, 0.0, 0.0);
  s.addParm("SoftDrop:zCut", 0.1, true, false, 0.0, 0.0);
  s.addParm("SoftDrop:beta", 0.0, false, false, 0.0, 0.0);  // may be 0, >0, or <0
  s.addParm("SoftDrop:R0", 1.0, true, false, 0.0, 0.0);
  s.addParm("SoftDrop:ktFloor", 1.0, true, false, 0.0, 0.0);
  s.addWord("Analysis:generatorTag", "PYTHIA-8:tune-Monash");
}

inline JetParams readJetParams(Pythia8::Pythia& pythia) {
  Pythia8::Settings& s = pythia.settings;
  JetParams jp;
  jp.R = s.parm("HadronJet:R");
  jp.jet_ptmin = s.parm("HadronJet:ptMin");
  jp.max_rap = s.parm("HadronJet:maxRap");
  jp.match_dR = s.parm("HadronJet:matchdR");
  return jp;
}

inline GroomParams readGroomParams(Pythia8::Pythia& pythia) {
  Pythia8::Settings& s = pythia.settings;
  GroomParams g;
  g.z_cut = s.parm("SoftDrop:zCut");
  g.beta = s.parm("SoftDrop:beta");
  g.R0 = s.parm("SoftDrop:R0");
  g.kt_floor = s.parm("SoftDrop:ktFloor");
  return g;
}

inline std::string generatorTag(Pythia8::Pythia& pythia) {
  return pythia.settings.word("Analysis:generatorTag");
}

}  // namespace h2p

#endif  // HAVE_PYTHIA8
