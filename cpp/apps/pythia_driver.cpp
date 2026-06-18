// pythia_driver — explicit PYTHIA 8.3 event source -> Lund RNTuple writer (§5).
//
// The pre-hadronization shower partons are read from the full event record (the
// string endpoints, status 71-79), NOT a final-state dump; MPI is disabled for a
// pure hadronization study (Bierlich et al., arXiv:2203.11601). Final-state
// visible particles are the hadron level. Both levels are clustered with
// anti-kt(R), matched, and primary-Lund-declustered into the shared schema.
//
// Configuration is a PYTHIA command card (run_settings.hpp): the same file sets
// the generator AND the jet/grooming parameters (custom registered settings), so
// nothing physics-related is hardcoded. Built-in defaults reproduce the previous
// behaviour when no card is given. A parallel herwig_driver (cluster model; Bellm
// et al., arXiv:1512.01178) would emit the identical schema for the §8 systematic.
//
// Usage:  pythia_driver [nEvents] [out.root] [seed] [card.cmnd]

#include "lund_io.hpp"
#include "lund_writer.hpp"
#include "run_settings.hpp"

#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  const std::uint64_t nEvents = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1000;
  const std::string out = (argc > 2) ? argv[2] : "jets.root";
  const int seed = (argc > 3) ? std::atoi(argv[3]) : 1;
  const std::string card = (argc > 4) ? argv[4] : "";

  fastjet::contrib::LundGenerator lund;
  Pythia8::Pythia pythia;
  h2p::registerAnalysisSettings(pythia);  // custom jet/grooming knobs + defaults

  // Built-in generation defaults (overridable by the card below).
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = 100.");
  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("Print:quiet = on");
  if (!card.empty()) {
    std::cout << "[pythia_driver] reading card " << card << "\n";
    pythia.readFile(card);  // user card overrides any of the above + the analysis knobs
  }
  // CLI seed wins over the card, so sweeps can script reproducible seeds.
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed % 900000000));

  if (!pythia.init()) {
    std::cerr << "[pythia_driver] init failed\n";
    return 1;
  }

  const h2p::JetParams jp = h2p::readJetParams(pythia);
  const h2p::GroomParams g = h2p::readGroomParams(pythia);
  const std::string gen = h2p::generatorTag(pythia);
  std::cout << "[pythia_driver] R=" << jp.R << " ptMin=" << jp.jet_ptmin
            << " z_cut=" << g.z_cut << " beta=" << g.beta << " kt_floor=" << g.kt_floor
            << " generator=" << gen << "\n";

  h2p::LundWriter writer(out, "Jets", g, gen);

  for (std::uint64_t iev = 0; iev < nEvents; ++iev) {
    if (!pythia.next()) continue;
    std::vector<fastjet::PseudoJet> hadrons, partons;
    for (int i = 0; i < pythia.event.size(); ++i) {
      const auto& prt = pythia.event[i];
      if (prt.isFinal() && prt.isVisible())
        hadrons.emplace_back(prt.px(), prt.py(), prt.pz(), prt.e());
      const int as = std::abs(prt.status());
      if (prt.isParton() && as >= 71 && as <= 79)
        partons.emplace_back(prt.px(), prt.py(), prt.pz(), prt.e());
    }
    const auto matched = h2p::getMatchedHadronPartonJets(hadrons, partons, jp);
    std::uint32_t jIdx = 0;
    for (const auto& [hj, pj] : matched) {
      h2p::LundSeq x = h2p::primaryLund(hj, lund, g);
      h2p::LundSeq y = h2p::primaryLund(pj, lund, g);
      if (x.lnkt.empty() && y.lnkt.empty()) continue;
      writer.fill(iev, jIdx++, pythia.info.weight(), hj, x, y);
    }
  }

  std::cout << "[pythia_driver] wrote " << writer.nWritten() << " jets to " << out << "\n";
  return 0;
}
