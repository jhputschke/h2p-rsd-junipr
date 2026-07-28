// lund_io.hpp — primary-Lund declustering + hadron/parton jet matching.
//
// Promoted from write_lund_rntuple.cxx (the research script). FastJet-only (no
// ROOT), so the unit tests can exercise the Soft Drop boundary and the greedy
// one-to-one matching without a ROOT dependency.
//
// References:
//   anti-kt        Cacciari, Salam & Soyez, JHEP 04 (2008) 063, 0802.1189
//   FastJet        Cacciari, Salam & Soyez, EPJC 72 (2012) 1896, 1111.6097
//   Lund jet plane Dreyer, Salam & Soyez, JHEP 12 (2018) 064, 1807.04758
//   Soft Drop      Larkoski et al., JHEP 05 (2014) 146, 1402.2657
//   Recursive SD   Dreyer et al., JHEP 06 (2018) 093, 1804.03657
#pragma once

#include "fastjet/PseudoJet.hh"
#include "fastjet/contrib/LundGenerator.hh"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace h2p {

// Grooming / pipeline parameters (the boundary cut in the Lund plane + floor).
struct GroomParams {
  double z_cut = 0.1;     // CMS/ATLAS standard
  double beta = 0.0;      // beta = 0 -> mMDT-like, most NP-robust
  double R0 = 1.0;        // SD angular normalisation (drops out for beta = 0)
  double kt_floor = 1.0;  // GeV; lower edge of the perturbative band
};

// anti-kt jet clustering / matching parameters.
struct JetParams {
  double R = 0.4;           // anti-kt jet radius
  double jet_ptmin = 20.0;  // GeV; min jet pt at both levels
  double max_rap = 2.0;     // |y| acceptance on the hadron jet
  double match_dR = 0.3;    // hadron<->parton matching cone
};

// Primary Lund sequence as struct-of-arrays (maps onto an Awkward record).
struct LundSeq {
  std::vector<float> lnInvDelta;  // ln(1/DeltaR)
  std::vector<float> lnkt;        // ln(kt / GeV)
  std::vector<float> lnz;         // ln(z)
  std::vector<float> psi;         // azimuth of the softer prong about the harder
};

// THE pipeline grooming predicate: one definition shared by the persisted primary
// sequences (primaryLund) and the all-branch traversal (fullLundAux), so the two
// cannot drift. Soft Drop boundary (Larkoski et al., 1402.2657) + the perturbative
// kt floor that scopes the whole pipeline to the resolvable band.
inline bool passesGroom(double Delta, double kt, double z, const GroomParams& g) {
  if (Delta <= 0.0 || kt <= 0.0 || z <= 0.0) return false;
  if (z <= g.z_cut * std::pow(Delta / g.R0, g.beta)) return false;  // Soft Drop boundary
  return kt >= g.kt_floor;                                          // perturbative floor
}

// Decluster the primary Lund plane and keep splittings above the Soft Drop
// boundary z > z_cut (Delta/R0)^beta and the perturbative ln(kt) floor.
LundSeq primaryLund(const fastjet::PseudoJet& jet,
                    const fastjet::contrib::LundGenerator& lund,
                    const GroomParams& g);

// Per-jet ALL-BRANCH groomed scalars: the conditioning-side information the
// primary-only sequence structurally cannot carry (docs/PLAN_Input.md).
//
// `mg`/`ptg` are the **pipeline-groomed** jet mass and transverse momentum: taken from
// the momentum surviving exactly the predicate above (`passesGroom`, kt floor included),
// NOT the textbook z_cut-only Soft Drop quantities. One grooming definition per file, by
// design — the persisted sequences and these are groomed identically.
//
// `ptg` is deliberately paired with `mg` rather than shipping a mass-drop ratio m_g/m:
// the encoder already conditions on ln(m_g/pt), so ln(m_g/m) would be an invertible
// reparameterization handing it ln(m/pt) — the UNGROOMED mass, which is exactly what the
// grooming-first design excludes. ln(pt_g/pt) carries the same "how much did grooming
// remove" information while leaving nothing ungroomed reconstructable.
//
// The `kt_sec_*` / `sec_attach` fields summarize the SECONDARY planes' kinematics, not
// just their count: a single hard off-spine splitting (a genuinely three-pronged jet) and
// several soft ones give the same `n_all - n_primary` but different physics. All three
// are 0 when there is no off-spine passing splitting at all; consumers must gate on
// `n_all > n_primary` rather than reading 0 as a measurement.
struct JetAux {
  float mg = 0.f;               // pipeline-groomed jet mass [GeV]
  float ptg = 0.f;              // pipeline-groomed jet pT [GeV]
  std::uint32_t n_primary = 0;  // passing splittings on the hardest-branch spine
  std::uint32_t n_all = 0;      // passing splittings over ALL branches
  float kt_sec_max = 0.f;       // hardest OFF-SPINE passing splitting's kt [GeV]
  float kt_sec_sum = 0.f;       // sum of kt over off-spine passing splittings [GeV]
  std::uint32_t sec_attach = 0; // primary-node index the HARDEST secondary hangs off
};

// Traverse the full C/A tree (not just the primary spine) with recursive-Soft-Drop
// semantics (Dreyer et al., 1804.03657) under `passesGroom`: a passing splitting is
// counted and BOTH prongs are followed; a failing one drops the softer prong and
// continues down the harder one. Lund conventions of Dreyer, Salam & Soyez
// (1807.04758): Delta = DeltaR(p1, p2), z = pT2/(pT1+pT2), kt = pT2 * Delta.
//
// Guaranteed: `fullLundAux(jet, g).n_primary == primaryLund(jet, lund, g).lnkt.size()`
// — the spine is the primary plane and the predicate is shared (pinned by
// cpp/tests/test_lund_io.cpp).
JetAux fullLundAux(const fastjet::PseudoJet& jet, const GroomParams& g);

// Cluster two same-event particle collections with anti-kt(R) and geometrically
// pair the jets into (hadronJet, partonJet) tuples (greedy hardest-first,
// one-to-one within match_dR).
std::vector<std::pair<fastjet::PseudoJet, fastjet::PseudoJet>>
getMatchedHadronPartonJets(const std::vector<fastjet::PseudoJet>& hadrons,
                           const std::vector<fastjet::PseudoJet>& partons,
                           const JetParams& jp);

}  // namespace h2p
