// write_lund_rntuple — promoted from the research script to a built binary.
//
// Serializes matched (hadron-level x, parton-level y) primary Lund sequences into
// a ROOT RNTuple ("Jets") consumed by the Python data pipeline (data/rntuple.py).
//
// Event source: PYTHIA 8.3 when compiled with HAVE_PYTHIA8 (the real generator) —
// final-state hadrons as x, the pre-hadronization shower partons (status 71-79,
// the string endpoints) as y, MPI disabled for a pure hadronization study
// (Bierlich et al., arXiv:2203.11601). Otherwise a self-contained toy source so
// the writer (clustering -> matching -> primaryLund -> RNTuple) always builds and
// runs off-cluster.
//
// Usage:  write_lund_rntuple [nEvents] [out.root] [seed] [card.cmnd]

#include "lund_io.hpp"
#include "lund_writer.hpp"
#include "run_settings.hpp"

#include "fastjet/PseudoJet.hh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef HAVE_PYTHIA8
#include "Pythia8/Pythia.h"
#endif

namespace {

struct Event {
  std::vector<fastjet::PseudoJet> hadrons;
  std::vector<fastjet::PseudoJet> partons;
  double weight = 1.0;
};

// Process one event: cluster + match both levels, primary-Lund-decluster each
// matched pair, and write one entry per jet.
std::uint32_t processEvent(std::uint64_t iev, const Event& ev,
                           const fastjet::contrib::LundGenerator& lund,
                           const h2p::GroomParams& g, const h2p::JetParams& jp,
                           h2p::LundWriter& writer) {
  const auto matched = h2p::getMatchedHadronPartonJets(ev.hadrons, ev.partons, jp);
  std::uint32_t jIdx = 0;
  for (const auto& [hadronJet, partonJet] : matched) {
    h2p::LundSeq x = h2p::primaryLund(hadronJet, lund, g);
    h2p::LundSeq y = h2p::primaryLund(partonJet, lund, g);
    if (x.lnkt.empty() && y.lnkt.empty()) continue;  // nothing survives grooming
    // hadron-level all-branch scalars: conditioning-side only (docs/PLAN_Input.md)
    const h2p::JetAux aux = h2p::fullLundAux(hadronJet, g);
    writer.fill(iev, jIdx++, ev.weight, hadronJet, x, y, aux);
  }
  return jIdx;
}

#ifndef HAVE_PYTHIA8
// Toy event source: one hard prong + a few collinear emissions at parton level,
// then hadron level = partons smeared + a couple of soft extras. Exercises the
// full writer path without a generator.
Event toyEvent(std::mt19937& rng) {
  std::normal_distribution<double> smear(0.0, 0.05);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Event ev;
  const double pt0 = 100.0, eta0 = 0.0, phi0 = 1.0;
  auto make = [](double pt, double eta, double phi) {
    fastjet::PseudoJet p;
    p.reset_PtYPhiM(pt, eta, phi, 0.0);
    return p;
  };
  // parton level: leading prong + collinear emissions
  ev.partons.push_back(make(pt0, eta0, phi0));
  for (int i = 0; i < 5; ++i) {
    double frac = 0.05 + 0.2 * u(rng);
    ev.partons.push_back(make(pt0 * frac, eta0 + 0.1 * (u(rng) - 0.5), phi0 + 0.1 * (u(rng) - 0.5)));
  }
  // hadron level: partons smeared
  for (const auto& p : ev.partons)
    ev.hadrons.push_back(make(p.pt() * (1.0 + smear(rng)), p.rap() + smear(rng), p.phi() + smear(rng)));
  // a couple of soft hadron-level extras (migration)
  for (int i = 0; i < 3; ++i)
    ev.hadrons.push_back(make(1.0 + 3.0 * u(rng), eta0 + 0.2 * (u(rng) - 0.5), phi0 + 0.2 * (u(rng) - 0.5)));
  return ev;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const std::uint64_t nEvents = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 200;
  const std::string out = (argc > 2) ? argv[2] : "jets.root";
  const int seed = (argc > 3) ? std::atoi(argv[3]) : 1;
  const std::string card = (argc > 4) ? argv[4] : "";  // PYTHIA path only

  h2p::GroomParams g;  // defaults: z_cut=0.1, beta=0, R0=1, kt_floor=1
  h2p::JetParams jp;   // defaults: R=0.4, ptmin=20, |y|<2, match 0.3
  std::string generator;
  fastjet::contrib::LundGenerator lund;  // reclusters with C/A internally

#ifdef HAVE_PYTHIA8
  Pythia8::Pythia pythia;
  h2p::registerAnalysisSettings(pythia);  // jet/grooming knobs as custom settings
  pythia.readString("Beams:eCM = 13000.");
  pythia.readString("HardQCD:all = on");
  pythia.readString("PhaseSpace:pTHatMin = 100.");
  pythia.readString("PartonLevel:MPI = off");  // pure hadronization study
  pythia.readString("Print:quiet = on");
  if (!card.empty()) pythia.readFile(card);  // card overrides generator + analysis knobs
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed % 900000000));
  if (!pythia.init()) {
    std::cerr << "[write_lund_rntuple] PYTHIA init failed\n";
    return 1;
  }
  jp = h2p::readJetParams(pythia);
  g = h2p::readGroomParams(pythia);
  generator = h2p::generatorTag(pythia);
  std::cout << "[write_lund_rntuple] PYTHIA source, " << nEvents << " events -> " << out << "\n";
#else
  generator = "toy-synthetic";
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::cout << "[write_lund_rntuple] toy source (no PYTHIA), " << nEvents << " events -> " << out
            << "\n";
#endif

  h2p::LundWriter writer(out, "Jets", g, generator);
  std::uint64_t nJets = 0;

  for (std::uint64_t iev = 0; iev < nEvents; ++iev) {
    Event ev;
#ifdef HAVE_PYTHIA8
    if (!pythia.next()) continue;
    ev.weight = pythia.info.weight();
    for (int i = 0; i < pythia.event.size(); ++i) {
      const auto& prt = pythia.event[i];
      if (prt.isFinal() && prt.isVisible()) {
        ev.hadrons.emplace_back(prt.px(), prt.py(), prt.pz(), prt.e());
      }
      const int as = std::abs(prt.status());
      if (prt.isParton() && as >= 71 && as <= 79) {  // pre-hadronization partons
        ev.partons.emplace_back(prt.px(), prt.py(), prt.pz(), prt.e());
      }
    }
#else
    ev = toyEvent(rng);
#endif
    nJets += processEvent(iev, ev, lund, g, jp, writer);
  }

  std::cout << "[write_lund_rntuple] wrote " << writer.nWritten() << " jets to " << out << "\n";
  return 0;
}
