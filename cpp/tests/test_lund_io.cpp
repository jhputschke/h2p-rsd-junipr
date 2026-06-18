// test_lund_io — unit tests for the Soft Drop boundary, the ln(kt) floor, and the
// greedy one-to-one hadron<->parton matching. Self-contained (no catch2/gtest):
// returns nonzero on first failure.
//
//   Soft Drop boundary  z > z_cut (Delta/R0)^beta   (Larkoski et al., 1402.2657)
//   ln(kt) floor        kt >= kt_floor               (perturbative band)

#include "lund_io.hpp"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(cond, msg)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::printf("  FAIL: %s\n", msg);                        \
      ++g_fail;                                                \
    } else {                                                   \
      std::printf("  ok:   %s\n", msg);                        \
    }                                                          \
  } while (0)

// Build a single C/A-clusterable jet from a hard + a soft prong and return its
// primary Lund sequence under the given grooming params.
static h2p::LundSeq declusterPair(double ptHard, double ptSoft, double dPhi,
                                  const h2p::GroomParams& g) {
  fastjet::PseudoJet a, b;
  a.reset_PtYPhiM(ptHard, 0.0, 0.0, 0.0);
  b.reset_PtYPhiM(ptSoft, 0.0, dPhi, 0.0);
  std::vector<fastjet::PseudoJet> parts{a, b};
  fastjet::JetDefinition jd(fastjet::antikt_algorithm, 1.0);
  auto* cs = new fastjet::ClusterSequence(parts, jd);
  auto jets = fastjet::sorted_by_pt(cs->inclusive_jets(0.0));
  cs->delete_self_when_unused();
  fastjet::contrib::LundGenerator lund;
  return h2p::primaryLund(jets.at(0), lund, g);
}

int main() {
  h2p::GroomParams g;  // z_cut=0.1, beta=0, kt_floor=1.0

  std::printf("[test_lund_io] Soft Drop boundary + kt floor\n");
  // symmetric-ish, well above both cuts -> kept
  auto kept = declusterPair(100.0, 40.0, 0.3, g);
  CHECK(kept.lnkt.size() == 1, "z>z_cut and kt>floor -> splitting kept");

  // very asymmetric: z ~ 5/105 < 0.1 -> dropped by Soft Drop
  auto z_drop = declusterPair(100.0, 5.0, 0.3, g);
  CHECK(z_drop.lnkt.empty(), "z<z_cut -> splitting removed by Soft Drop");

  // collinear: kt = pt_soft * Delta small -> below ln(kt) floor -> dropped
  auto kt_drop = declusterPair(100.0, 40.0, 0.01, g);
  CHECK(kt_drop.lnkt.empty(), "kt<kt_floor -> splitting removed by floor");

  std::printf("[test_lund_io] greedy one-to-one matching\n");
  h2p::JetParams jp;  // R=0.4, ptmin=20, match 0.3
  auto mk = [](double pt, double phi) {
    fastjet::PseudoJet p;
    p.reset_PtYPhiM(pt, 0.0, phi, 0.0);
    return p;
  };
  // two well-separated jets at both levels; hadrons slightly displaced
  std::vector<fastjet::PseudoJet> partons{mk(100.0, 0.0), mk(80.0, 2.5)};
  std::vector<fastjet::PseudoJet> hadrons{mk(98.0, 0.02), mk(82.0, 2.52)};
  auto matched = h2p::getMatchedHadronPartonJets(hadrons, partons, jp);
  CHECK(matched.size() == 2, "two jets -> two matched pairs");
  bool nearest = true;
  for (const auto& [hj, pj] : matched) nearest = nearest && (hj.squared_distance(pj) < 0.09);
  CHECK(nearest, "each hadron jet paired with its nearest parton jet (within cone)");

  // a hadron jet with no parton jet inside the cone is dropped
  std::vector<fastjet::PseudoJet> partons2{mk(100.0, 0.0)};
  std::vector<fastjet::PseudoJet> hadrons2{mk(98.0, 0.02), mk(50.0, 3.0)};
  auto matched2 = h2p::getMatchedHadronPartonJets(hadrons2, partons2, jp);
  CHECK(matched2.size() == 1, "unmatched hadron jet (no parton in cone) is dropped");

  if (g_fail) {
    std::printf("[test_lund_io] %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("[test_lund_io] all tests passed\n");
  return 0;
}
