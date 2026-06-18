// pythia_driver — explicit PYTHIA 8.3 event source -> Lund RNTuple writer (§5).
//
// The pre-hadronization shower partons are read from the full event record (the
// string endpoints, status 71-79), NOT a final-state dump; MPI is disabled for a
// pure hadronization study (Bierlich et al., arXiv:2203.11601). Final-state
// visible particles are the hadron level. Both levels are clustered with
// anti-kt(R), matched, and primary-Lund-declustered into the shared schema.
//
// Built only when PYTHIA 8 is found. A parallel herwig_driver (cluster model;
// Bellm et al., arXiv:1512.01178) emitting the identical schema powers the
// dominant generator systematic of §8.
//
// Usage:  pythia_driver [nEvents] [out.root] [seed] [pTHatMin]

#include "lund_io.hpp"
#include "lund_writer.hpp"

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
  const double pTHatMin = (argc > 4) ? std::atof(argv[4]) : 100.0;

  const h2p::GroomParams g;
  const h2p::JetParams jp;
  fastjet::contrib::LundGenerator lund;

  Pythia8::Pythia pythia;
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(pTHatMin));
  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed % 900000000));
  pythia.readString("Print:quiet = on");
  if (!pythia.init()) {
    std::cerr << "[pythia_driver] init failed\n";
    return 1;
  }

  h2p::LundWriter writer(out, "Jets", g, "PYTHIA-8:tune-Monash");

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
