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
fastjet::PseudoJet groomRecurse(const fastjet::PseudoJet& p, const GroomParams& g,
                                bool on_spine, JetAux& aux) {
  fastjet::PseudoJet p1, p2;
  if (!p.has_parents(p1, p2)) return p;                  // leaf: a single constituent
  if (p1.pt2() < p2.pt2()) std::swap(p1, p2);            // p1 harder, as LundDeclustering does

  const double Delta = p1.delta_R(p2);
  const double pt_sum = p1.pt() + p2.pt();
  const double z = (pt_sum > 0.0) ? p2.pt() / pt_sum : 0.0;
  const double kt = p2.pt() * Delta;

  if (!passesGroom(Delta, kt, z, g)) {
    // recursive Soft Drop: discard the softer prong, keep walking the harder one
    return groomRecurse(p1, g, on_spine, aux);
  }
  ++aux.n_all;
  if (on_spine) ++aux.n_primary;
  // A passing splitting is resolved: BOTH prongs stay, and the softer one opens a
  // secondary Lund plane the primary sequence never sees.
  return groomRecurse(p1, g, on_spine, aux) + groomRecurse(p2, g, false, aux);
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
  const fastjet::PseudoJet kept = groomRecurse(ca, g, /*on_spine=*/true, aux);
  aux.mg = static_cast<float>(std::sqrt(std::max(0.0, kept.m2())));  // m2 < 0 only by rounding
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
