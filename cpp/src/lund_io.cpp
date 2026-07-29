#include "lund_io.hpp"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/tools/Recluster.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace h2p {

LundSeq primaryLund(const fastjet::PseudoJet& jet,
                    const fastjet::contrib::LundGenerator& lund,
                    const GroomParams& g) {
  LundSeq s;
  for (const fastjet::contrib::LundDeclustering& d : lund(jet)) {
    if (!passesGroom(d.Delta(), d.kt(), d.z(), g)) continue;
    s.lnInvDelta.push_back(static_cast<float>(std::log(1.0 / d.Delta())));
    s.lnkt.push_back(static_cast<float>(std::log(d.kt())));
    s.lnz.push_back(static_cast<float>(std::log(d.z())));
    s.psi.push_back(static_cast<float>(d.psi()));
  }
  return s;
}

namespace {

// One node of the all-branch recursion. Returns the 4-momentum KEPT below `p`;
// `on_spine` marks the hardest-branch chain, which is exactly the primary plane
// LundGenerator walks (it declusters, takes the larger-pt prong, and repeats).
// `attach` is the primary-node index whose softer prong this subtree descends from
// (meaningful only off-spine), so secondary kinematics can be tied back to the primary
// emission that opened their plane.
fastjet::PseudoJet groomRecurse(const fastjet::PseudoJet& p, const GroomParams& g,
                                bool on_spine, std::uint32_t attach, JetAux& aux) {
  fastjet::PseudoJet p1, p2;
  if (!p.has_parents(p1, p2)) return p;                  // leaf: a single constituent
  if (p1.pt2() < p2.pt2()) std::swap(p1, p2);            // p1 harder, as LundDeclustering does

  const double Delta = p1.delta_R(p2);
  const double pt_sum = p1.pt() + p2.pt();
  const double z = (pt_sum > 0.0) ? p2.pt() / pt_sum : 0.0;
  const double kt = p2.pt() * Delta;

  // `on_spine` is a property of THIS node, so it selects the floor for THIS splitting:
  // the spine is the primary plane and must stay on `kt_floor` (that is what makes
  // n_primary == primaryLund's length), while an off-spine splitting may use the looser
  // `secondaryFloor()`. The two coincide unless a card set SoftDrop:ktFloorSec.
  if (!passesGroom(Delta, kt, z, g, on_spine ? g.kt_floor : g.secondaryFloor())) {
    // recursive Soft Drop: discard the softer prong, keep walking the harder one
    return groomRecurse(p1, g, on_spine, attach, aux);
  }
  ++aux.n_all;

  std::uint32_t child_attach = attach;
  if (on_spine) {
    // this splitting IS primary node number `n_primary` (0-based, declustering order,
    // i.e. widest-angle first for C/A) -- the index the plane it opens hangs off
    child_attach = aux.n_primary;
    ++aux.n_primary;
  } else {
    // an off-spine splitting: this is secondary-plane structure the primary sequence
    // structurally cannot represent, so record its hardness, not just its existence
    const float ktf = static_cast<float>(kt);
    aux.kt_sec_sum += ktf;
    if (ktf > aux.kt_sec_max) {
      aux.kt_sec_max = ktf;
      aux.sec_attach = attach;
    }
  }
  // A passing splitting is resolved: BOTH prongs stay, and the softer one opens a
  // secondary Lund plane the primary sequence never sees.
  return groomRecurse(p1, g, on_spine, child_attach, aux)
       + groomRecurse(p2, g, false, child_attach, aux);
}

}  // namespace

JetAux fullLundAux(const fastjet::PseudoJet& jet, const GroomParams& g) {
  JetAux aux;
  if (!jet.has_constituents()) return aux;
  // The SAME C/A reclustering LundGenerator performs internally (its default
  // JetDefinition(cambridge_algorithm, max_allowable_R)), so the tree walked here and
  // the tree behind `primaryLund` are one and the same.
  const fastjet::Recluster recluster(
      fastjet::JetDefinition(fastjet::cambridge_algorithm,
                             fastjet::JetDefinition::max_allowable_R));
  const fastjet::PseudoJet ca = recluster(jet);
  const fastjet::PseudoJet kept = groomRecurse(ca, g, /*on_spine=*/true, /*attach=*/0, aux);
  aux.mg = static_cast<float>(std::sqrt(std::max(0.0, kept.m2())));  // m2 < 0 only by rounding
  aux.ptg = static_cast<float>(kept.pt());
  return aux;
}

std::vector<std::pair<fastjet::PseudoJet, fastjet::PseudoJet>>
getMatchedHadronPartonJets(const std::vector<fastjet::PseudoJet>& hadrons,
                           const std::vector<fastjet::PseudoJet>& partons,
                           const JetParams& jp) {
  const fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, jp.R);

  // Cluster one collection; transfer ClusterSequence ownership to the jets so
  // LundGenerator can later reach the constituents (or free now if empty).
  auto cluster = [&](const std::vector<fastjet::PseudoJet>& parts) {
    auto* cs = new fastjet::ClusterSequence(parts, jet_def);
    std::vector<fastjet::PseudoJet> jets = fastjet::sorted_by_pt(cs->inclusive_jets(jp.jet_ptmin));
    if (jets.empty())
      delete cs;
    else
      cs->delete_self_when_unused();
    return jets;
  };

  const std::vector<fastjet::PseudoJet> hadronJets = cluster(hadrons);
  const std::vector<fastjet::PseudoJet> partonJets = cluster(partons);

  std::vector<std::pair<fastjet::PseudoJet, fastjet::PseudoJet>> matched;
  matched.reserve(hadronJets.size());

  std::vector<bool> used(partonJets.size(), false);
  const double match_dR2 = jp.match_dR * jp.match_dR;

  for (const fastjet::PseudoJet& hj : hadronJets) {
    if (std::abs(hj.rap()) > jp.max_rap) continue;  // acceptance on the measured jet

    int best = -1;
    double best_dR2 = match_dR2;  // only accept within the cone
    for (std::size_t j = 0; j < partonJets.size(); ++j) {
      if (used[j]) continue;
      const double dR2 = hj.squared_distance(partonJets[j]);
      if (dR2 < best_dR2) {
        best_dR2 = dR2;
        best = static_cast<int>(j);
      }
    }
    if (best >= 0) {
      used[best] = true;
      matched.emplace_back(hj, partonJets[best]);
    }
  }
  return matched;
}

}  // namespace h2p
