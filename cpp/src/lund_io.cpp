#include "lund_io.hpp"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"

#include <cmath>
#include <cstddef>

namespace h2p {

LundSeq primaryLund(const fastjet::PseudoJet& jet,
                    const fastjet::contrib::LundGenerator& lund,
                    const GroomParams& g) {
  LundSeq s;
  for (const fastjet::contrib::LundDeclustering& d : lund(jet)) {
    const double Delta = d.Delta();
    const double kt = d.kt();
    const double z = d.z();
    if (Delta <= 0.0 || kt <= 0.0 || z <= 0.0) continue;
    if (z <= g.z_cut * std::pow(Delta / g.R0, g.beta)) continue;  // Soft Drop boundary
    if (kt < g.kt_floor) continue;                                // perturbative floor
    s.lnInvDelta.push_back(static_cast<float>(std::log(1.0 / Delta)));
    s.lnkt.push_back(static_cast<float>(std::log(kt)));
    s.lnz.push_back(static_cast<float>(std::log(z)));
    s.psi.push_back(static_cast<float>(d.psi()));
  }
  return s;
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
