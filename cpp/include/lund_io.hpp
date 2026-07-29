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
  // OPTIONAL secondary (off-spine) floor, used by `fullLundAux` ONLY. <= 0 mirrors
  // `kt_floor`, which is the historical single-floor behaviour and the default, so an
  // unchanged card reproduces an existing file bit-for-bit.
  //
  // Why an asymmetric floor is defensible at all: `kt_floor` does two jobs at once.
  // On the persisted sequences it scopes the PREDICTED and matched object to the band
  // where hadron<->parton correspondence holds. The aux scalars are CONDITIONING
  // inputs (hadron-level only, never targets), and nothing requires those to live in
  // that band — `jet_pt` and `|eta|` already do not. Meanwhile a fixed ABSOLUTE floor
  // is a far deeper cut off-spine, because a secondary plane hangs off a prong
  // carrying only z*pT of the jet: at the 1 GeV default 80.6% of jets have no passing
  // off-spine splitting at all, and dropping the floor to 0.2 multiplies <n_sec> by
  // ~9 (docs/PLAN_Input.md, "Is the 82.6% zero fraction a bug").
  double kt_floor_sec = -1.0;

  // The floor that applies OFF-SPINE, mirror sentinel resolved. Always read the
  // secondary floor through this — `kt_floor_sec` itself may be the sentinel.
  double secondaryFloor() const { return (kt_floor_sec > 0.0) ? kt_floor_sec : kt_floor; }

  // True when the two floors differ, i.e. the aux scalars are groomed differently
  // from the persisted sequences. Consumers that compare against a textbook Soft Drop
  // quantity, or against another file, must branch on this.
  bool asymmetricFloor() const { return secondaryFloor() != kt_floor; }
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
//
// The floor is an explicit argument in the general form because `fullLundAux` may
// apply a looser one off-spine (`GroomParams::kt_floor_sec`). The DEFINITION is still
// single-sourced — the Soft Drop boundary, the degenerate-kinematics rejection and
// the shape of the cut all live here once; only the floor VALUE varies, and only for
// the conditioning-side traversal. `primaryLund` uses the 4-argument form, i.e.
// `g.kt_floor`, always: the persisted sequences have exactly one floor, by design.
inline bool passesGroom(double Delta, double kt, double z, const GroomParams& g,
                        double kt_floor) {
  if (Delta <= 0.0 || kt <= 0.0 || z <= 0.0) return false;
  if (z <= g.z_cut * std::pow(Delta / g.R0, g.beta)) return false;  // Soft Drop boundary
  return kt >= kt_floor;                                            // perturbative floor
}

inline bool passesGroom(double Delta, double kt, double z, const GroomParams& g) {
  return passesGroom(Delta, kt, z, g, g.kt_floor);
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
// NOT the textbook z_cut-only Soft Drop quantities.
//
// They are therefore STRONGLY floor-dependent, which is easy to underestimate: a
// splitting under the floor is not a soft splitting (kt ~ pT_softer * Delta, so at
// Delta = 0.05 a sub-1-GeV-kt prong carries ~18 GeV), and it is DISCARDED here rather
// than merely left unrecorded as it is in `primaryLund`. Measured on 3220 jets from
// identical events, floor 1.0 -> 0.2: median pt_g/pt 0.394 -> 0.686, median m_g
// 2.54 -> 4.89 GeV, with only 1.4% of jets unchanged.
//
// Consequences, both load-bearing:
//   * Under `kt_floor_sec != kt_floor` these are groomed differently from the
//     persisted sequences. That is a deliberate choice for the counting/hardness aux
//     (`n_all`, `kt_sec_*`), which are absolute properties of the jet; it is more
//     delicate for `mg`/`ptg`, whose documented job is to be the COMPLEMENT of the
//     recorded sequence ("how much did this grooming remove"). Under an asymmetric
//     floor they are the complement of a tree the model never sees. Ship them as a
//     distinct, opt-in feature rather than silently redefining the symmetric ones.
//   * They cannot be recomputed downstream at another floor. The file records the
//     surviving primary splittings' (Delta, kt, z, psi) — shape, not momenta — and
//     sums the off-spine prongs away into four scalars, while m_g needs every kept
//     prong's 4-vector. Re-flooring in Python is exact for the SEQUENCES and
//     impossible for these.
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
// Splittings ON the spine use `g.kt_floor`; splittings off it use `g.secondaryFloor()`,
// which mirrors `kt_floor` unless a card set `SoftDrop:ktFloorSec`.
//
// Guaranteed: `fullLundAux(jet, g).n_primary == primaryLund(jet, lund, g).lnkt.size()`
// — the spine is the primary plane and the predicate is shared (pinned by
// cpp/tests/test_lund_io.cpp). This survives an asymmetric floor BECAUSE the spine
// keeps `kt_floor` and every predicate is evaluated on the UNGROOMED tree kinematics,
// so no off-spine decision can reach a spine one.
JetAux fullLundAux(const fastjet::PseudoJet& jet, const GroomParams& g);

// Cluster two same-event particle collections with anti-kt(R) and geometrically
// pair the jets into (hadronJet, partonJet) tuples (greedy hardest-first,
// one-to-one within match_dR).
std::vector<std::pair<fastjet::PseudoJet, fastjet::PseudoJet>>
getMatchedHadronPartonJets(const std::vector<fastjet::PseudoJet>& hadrons,
                           const std::vector<fastjet::PseudoJet>& partons,
                           const JetParams& jp);

}  // namespace h2p
