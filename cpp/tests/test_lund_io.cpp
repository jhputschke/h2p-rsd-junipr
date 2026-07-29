// test_lund_io — unit tests for the Soft Drop boundary, the ln(kt) floor, the greedy
// one-to-one hadron<->parton matching, and the all-branch aux observables.
// Self-contained (no catch2/gtest): returns nonzero on first failure.
//
//   Soft Drop boundary  z > z_cut (Delta/R0)^beta   (Larkoski et al., 1402.2657)
//   ln(kt) floor        kt >= kt_floor               (perturbative band)
//   all-branch aux      fullLundAux (docs/PLAN_Input.md): m_g + secondary-plane count

#include "lund_io.hpp"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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

static fastjet::PseudoJet mkPart(double pt, double y, double phi) {
  fastjet::PseudoJet p;
  p.reset_PtYPhiM(pt, y, phi, 0.0);
  return p;
}

// Cluster a massless particle list into one anti-kt(1.0) jet. The ClusterSequence
// outlives the call (delete_self_when_unused), so the returned jet keeps its
// constituents and its declustering history.
static fastjet::PseudoJet makeJet(const std::vector<fastjet::PseudoJet>& parts) {
  fastjet::JetDefinition jd(fastjet::antikt_algorithm, 1.0);
  auto* cs = new fastjet::ClusterSequence(parts, jd);
  auto jets = fastjet::sorted_by_pt(cs->inclusive_jets(0.0));
  cs->delete_self_when_unused();
  return jets.at(0);
}

// Two-prong fixture: one hard + one soft prong separated in azimuth.
static fastjet::PseudoJet pairJet(double ptHard, double ptSoft, double dPhi) {
  return makeJet({mkPart(ptHard, 0.0, 0.0), mkPart(ptSoft, 0.0, dPhi)});
}

// Three-prong fixture: a hard prong plus a two-particle secondary prong whose OWN
// splitting passes grooming — the configuration the primary sequence cannot see.
static fastjet::PseudoJet threeProngJet() {
  return makeJet({mkPart(100.0, 0.0, 0.0), mkPart(25.0, 0.0, 0.30), mkPart(20.0, 0.0, 0.40)});
}

static h2p::LundSeq declusterPair(double ptHard, double ptSoft, double dPhi,
                                  const h2p::GroomParams& g) {
  fastjet::contrib::LundGenerator lund;
  return h2p::primaryLund(pairJet(ptHard, ptSoft, dPhi), lund, g);
}

// INDEPENDENT reference implementation of `fullLundAux`'s splitting count, built on
// fjcontrib's own LundGenerator instead of raw has_parents/pt2 bookkeeping: walk the
// primary plane, and at every passing splitting recurse into the softer prong. Different
// declustering machinery, different harder/softer determination, and Delta/z/kt taken
// from LundDeclustering's cached values rather than recomputed.
//
// This is the guard against the one bug that would be invisible in the aggregate: a
// traversal that silently never leaves the hardest branch would still satisfy every
// primary-plane invariant while reporting n_all == n_primary for every jet.
static std::uint32_t refCountAll(const fastjet::PseudoJet& jet,
                                 const fastjet::contrib::LundGenerator& lund,
                                 const h2p::GroomParams& g) {
  std::uint32_t n = 0;
  for (const fastjet::contrib::LundDeclustering& d : lund(jet)) {
    if (!h2p::passesGroom(d.Delta(), d.kt(), d.z(), g)) continue;
    ++n;                                        // this primary-chain splitting
    n += refCountAll(d.softer(), lund, g);      // ...and everything inside the softer prong
  }
  return n;
}

// Same independent construction for the hardest OFF-SPINE kt. `on_spine` is true only at
// the top level: everything reached through a `softer()` prong is secondary by definition.
static double refSecKtMax(const fastjet::PseudoJet& jet,
                          const fastjet::contrib::LundGenerator& lund,
                          const h2p::GroomParams& g, bool on_spine) {
  double best = 0.0;
  for (const fastjet::contrib::LundDeclustering& d : lund(jet)) {
    if (!h2p::passesGroom(d.Delta(), d.kt(), d.z(), g)) continue;
    if (!on_spine) best = std::max(best, d.kt());
    best = std::max(best, refSecKtMax(d.softer(), lund, g, false));
  }
  return best;
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

  // ---- all-branch aux observables (docs/PLAN_Input.md) --------------------
  std::printf("[test_lund_io] fullLundAux: predicate consistency\n");
  fastjet::contrib::LundGenerator lund;
  // The guarantee the shared `passesGroom` buys: the aux spine IS the primary plane,
  // for every grooming setting, on every fixture.
  const std::vector<fastjet::PseudoJet> fixtures{
      pairJet(100.0, 40.0, 0.3),   // both cuts passed
      pairJet(100.0, 5.0, 0.3),    // fails Soft Drop
      pairJet(100.0, 40.0, 0.01),  // fails the kt floor
      threeProngJet(),
  };
  bool consistent = true;
  for (double z_cut : {0.0, 0.05, 0.1, 0.3}) {
    for (double kt_floor : {0.0, 0.5, 1.0, 5.0}) {
      // -1 is the mirror sentinel (symmetric); the rest are genuine asymmetric floors,
      // including one ABOVE kt_floor, since nothing forbids a tighter secondary cut.
      for (double kt_floor_sec : {-1.0, 0.0, 0.2, 1.0, 10.0}) {
        h2p::GroomParams gg;
        gg.z_cut = z_cut;
        gg.kt_floor = kt_floor;
        gg.kt_floor_sec = kt_floor_sec;
        for (const auto& jet : fixtures) {
          const auto n_prim = h2p::fullLundAux(jet, gg).n_primary;
          const auto n_seq = h2p::primaryLund(jet, lund, gg).lnkt.size();
          consistent = consistent && (n_prim == n_seq);
        }
      }
    }
  }
  // The load-bearing invariant: the spine keeps kt_floor whatever the off-spine floor
  // is, so the aux spine IS the persisted primary plane even when the two disagree.
  CHECK(consistent, "n_primary == primaryLund size over a (z_cut, kt_floor, kt_floor_sec) grid");

  // ---- asymmetric floor: mirror default, and what a looser off-spine floor moves --
  std::printf("[test_lund_io] fullLundAux: asymmetric kt floor\n");
  {
    h2p::GroomParams sym;  // defaults: kt_floor_sec = -1 -> mirrors kt_floor
    CHECK(sym.secondaryFloor() == sym.kt_floor, "default kt_floor_sec mirrors kt_floor");
    CHECK(!sym.asymmetricFloor(), "default GroomParams is not asymmetric");
    h2p::GroomParams explicit_mirror = sym;
    explicit_mirror.kt_floor_sec = sym.kt_floor;
    CHECK(!explicit_mirror.asymmetricFloor(), "kt_floor_sec == kt_floor is not asymmetric");

    h2p::GroomParams loose = sym;
    loose.kt_floor_sec = 0.05 * sym.kt_floor;  // far below the spine floor
    CHECK(loose.asymmetricFloor(), "kt_floor_sec != kt_floor is asymmetric");

    // Byte-for-byte: mirroring must reproduce the single-floor result exactly, or
    // every existing card silently changes meaning.
    bool mirror_identical = true, spine_frozen = true, sec_weakly_grows = true;
    for (const auto& jet : fixtures) {
      const h2p::JetAux a_sym = h2p::fullLundAux(jet, sym);
      const h2p::JetAux a_mir = h2p::fullLundAux(jet, explicit_mirror);
      const h2p::JetAux a_loose = h2p::fullLundAux(jet, loose);
      mirror_identical = mirror_identical && a_mir.n_all == a_sym.n_all &&
                         a_mir.n_primary == a_sym.n_primary && a_mir.mg == a_sym.mg &&
                         a_mir.ptg == a_sym.ptg && a_mir.kt_sec_sum == a_sym.kt_sec_sum;
      // the spine is untouched by the off-spine floor...
      spine_frozen = spine_frozen && a_loose.n_primary == a_sym.n_primary;
      // ...while off-spine counts and the kept momentum can only grow
      sec_weakly_grows = sec_weakly_grows && a_loose.n_all >= a_sym.n_all &&
                         a_loose.kt_sec_sum >= a_sym.kt_sec_sum && a_loose.ptg >= a_sym.ptg &&
                         a_loose.mg >= a_sym.mg;
    }
    CHECK(mirror_identical, "explicit kt_floor_sec == kt_floor reproduces the single-floor aux");
    CHECK(spine_frozen, "lowering the off-spine floor leaves n_primary unchanged");
    CHECK(sec_weakly_grows, "lowering the off-spine floor weakly increases n_all/kt_sec_sum/ptg/mg");

    // The three-prong fixture has a real secondary, so the effect is strict there:
    // its off-spine splitting is admitted only once the secondary floor drops below it.
    h2p::GroomParams tight_sec = sym;
    tight_sec.kt_floor_sec = 1e9;  // no off-spine splitting can pass
    const h2p::JetAux a_tight = h2p::fullLundAux(threeProngJet(), tight_sec);
    const h2p::JetAux a_open = h2p::fullLundAux(threeProngJet(), sym);
    CHECK(a_tight.n_all == a_tight.n_primary, "an unreachable off-spine floor -> no secondaries");
    CHECK(a_tight.kt_sec_max == 0.0f && a_tight.kt_sec_sum == 0.0f,
          "no secondary -> kt_sec_* stay 0 under an asymmetric floor too");
    CHECK(a_tight.n_primary == a_open.n_primary,
          "raising ONLY the off-spine floor cannot change the spine");
    CHECK(a_tight.ptg < a_open.ptg, "a tighter off-spine floor discards secondary momentum");
  }

  std::printf("[test_lund_io] fullLundAux: secondary plane + groomed mass\n");
  const fastjet::PseudoJet three = threeProngJet();
  const h2p::JetAux a3 = h2p::fullLundAux(three, g);
  CHECK(a3.n_all > a3.n_primary, "hard secondary splitting -> n_all > n_primary");
  CHECK(a3.mg > 0.0f, "resolved two-prong substructure -> m_g > 0");

  // groomed pt (the ln(pt_g/pt) aux source): nothing is dropped in this fixture
  CHECK(a3.ptg > 0.0f, "groomed pt is positive");
  CHECK(a3.ptg <= static_cast<float>(three.pt()) + 1e-3f, "pt_g <= ungroomed jet pt");
  CHECK(std::abs(a3.ptg - static_cast<float>(three.pt())) < 1e-2f,
        "nothing groomed away -> pt_g == pt");

  // secondary KINEMATICS, not just the count: the b/c splitting is the only off-spine
  // one, so kt_max == kt_sum, and it hangs off primary node 0 (the a/bc splitting).
  CHECK(a3.kt_sec_max > 0.0f, "hard secondary -> kt_sec_max > 0");
  CHECK(std::abs(a3.kt_sec_max - a3.kt_sec_sum) < 1e-3f,
        "single secondary -> kt_sec_max == kt_sec_sum");
  CHECK(a3.sec_attach == 0, "the secondary hangs off primary node 0");

  // Raising z_cut can only remove splittings and momentum: both are weakly decreasing.
  h2p::GroomParams tight = g;
  tight.z_cut = 0.4;
  const h2p::JetAux a3_tight = h2p::fullLundAux(three, tight);
  CHECK(a3_tight.n_all <= a3.n_all, "raising z_cut weakly decreases n_all");
  CHECK(a3_tight.mg <= a3.mg, "raising z_cut weakly decreases m_g");

  // Nothing survives grooming -> the jet is its hard prong: massless, no splittings.
  const fastjet::PseudoJet dropped = pairJet(100.0, 5.0, 0.3);
  const h2p::JetAux a_groomed = h2p::fullLundAux(dropped, g);
  CHECK(a_groomed.n_all == 0, "fully-groomed jet -> n_all == 0");
  CHECK(a_groomed.mg == 0.0f, "fully-groomed jet (massless prong) -> m_g == 0");
  CHECK(a_groomed.n_primary == 0, "fully-groomed jet -> n_primary == 0");
  CHECK(a_groomed.kt_sec_max == 0.0f && a_groomed.kt_sec_sum == 0.0f,
        "no secondary -> kt_sec_* are 0 (undefined, gate on n_sec)");
  // the soft prong was dropped, so pt_g is the hard prong alone -- strictly below pt
  CHECK(a_groomed.ptg < static_cast<float>(dropped.pt()) - 1.0f,
        "grooming removed momentum -> pt_g < pt");
  CHECK(std::abs(a_groomed.ptg - 100.0f) < 1.0f, "pt_g == the surviving hard prong");

  // The four fixtures above are shallow by construction. Real jets have deep, wide C/A
  // trees, which is where a spine/predicate drift would actually show up — so replay the
  // invariants on an ensemble of many-particle jets (fixed seed: deterministic test).
  std::printf("[test_lund_io] fullLundAux: invariants on many-particle jets\n");
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  bool ens_primary = true, ens_order = true, ens_mass = true;
  int n_with_secondary = 0;
  for (int ijet = 0; ijet < 200; ++ijet) {
    std::vector<fastjet::PseudoJet> parts;
    parts.push_back(mkPart(100.0, 0.0, 0.0));  // hard core
    for (int i = 0; i < 30; ++i) {             // a spray of softer constituents
      const double pt = 0.5 + 40.0 * std::pow(u01(rng), 3.0);
      parts.push_back(mkPart(pt, 0.6 * (u01(rng) - 0.5), 0.6 * (u01(rng) - 0.5)));
    }
    const fastjet::PseudoJet jet = makeJet(parts);
    const h2p::JetAux a = h2p::fullLundAux(jet, g);
    ens_primary = ens_primary && (a.n_primary == h2p::primaryLund(jet, lund, g).lnkt.size());
    ens_order = ens_order && (a.n_all >= a.n_primary);
    // grooming only ever REMOVES momentum, so the groomed mass cannot exceed the jet's
    ens_mass = ens_mass && (a.mg <= static_cast<float>(jet.m()) + 1e-3f);
    // grooming only removes momentum, and the secondary summaries stay self-consistent
    ens_mass = ens_mass && (a.ptg <= static_cast<float>(jet.pt()) + 1e-2f);
    ens_mass = ens_mass && (a.kt_sec_sum >= a.kt_sec_max - 1e-3f);
    ens_mass = ens_mass && ((a.n_all > a.n_primary) == (a.kt_sec_max > 0.0f));
    ens_mass = ens_mass && (a.n_primary == 0 || a.sec_attach < a.n_primary);
    if (a.n_all > a.n_primary) ++n_with_secondary;
  }
  CHECK(ens_primary, "n_primary == primaryLund size on 200 deep 31-particle jets");
  CHECK(ens_order, "n_all >= n_primary on the ensemble");
  CHECK(ens_mass, "m_g/pt_g bounded + secondary summaries self-consistent on the ensemble");
  CHECK(n_with_secondary > 20, "the ensemble actually exercises secondary planes");

  // Cross-check the traversal against the independent LundGenerator-based reference,
  // over a grooming grid so both a starved and a busy working point are covered.
  std::printf("[test_lund_io] fullLundAux: n_all vs an independent implementation\n");
  std::mt19937 rng2(12345);
  bool ens_ref = true;
  std::uint32_t ref_total = 0, off_spine_total = 0;
  for (int ijet = 0; ijet < 200; ++ijet) {
    std::vector<fastjet::PseudoJet> parts;
    parts.push_back(mkPart(100.0, 0.0, 0.0));
    for (int i = 0; i < 30; ++i) {
      const double pt = 0.5 + 40.0 * std::pow(u01(rng2), 3.0);
      parts.push_back(mkPart(pt, 0.6 * (u01(rng2) - 0.5), 0.6 * (u01(rng2) - 0.5)));
    }
    const fastjet::PseudoJet jet = makeJet(parts);
    for (double kt_floor : {0.2, 1.0, 5.0}) {
      h2p::GroomParams gg;
      gg.kt_floor = kt_floor;
      const h2p::JetAux a = h2p::fullLundAux(jet, gg);
      const std::uint32_t ref = refCountAll(jet, lund, gg);
      ens_ref = ens_ref && (a.n_all == ref);
      // ...and the secondary hardness, against the same independent construction
      ens_ref = ens_ref && (std::abs(a.kt_sec_max - refSecKtMax(jet, lund, gg, true)) < 1e-3);
      ref_total += ref;
      off_spine_total += a.n_all - a.n_primary;
    }
  }
  CHECK(ens_ref, "n_all AND kt_sec_max match the independent LundGenerator construction");
  CHECK(off_spine_total > 0, "the count genuinely leaves the hardest branch (n_sec > 0 somewhere)");
  std::printf("        (reference counted %u passing splittings, %u of them off-spine)\n",
              ref_total, off_spine_total);

  if (g_fail) {
    std::printf("[test_lund_io] %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("[test_lund_io] all tests passed\n");
  return 0;
}
