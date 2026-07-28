#include "lund_writer.hpp"

#include <utility>

using ROOT::RNTupleModel;
using ROOT::RNTupleWriter;

namespace h2p {

LundWriter::LundWriter(const std::string& path, const std::string& ntuple, const GroomParams& g,
                       std::string generator)
    : g_(g), generator_(std::move(generator)) {
  auto model = RNTupleModel::Create();

  fEvent_ = model->MakeField<std::uint64_t>("event");
  fJetIdx_ = model->MakeField<std::uint32_t>("jet_index");
  fWeight_ = model->MakeField<double>("weight");

  fJetPt_ = model->MakeField<float>("jet_pt");
  fJetEta_ = model->MakeField<float>("jet_eta");
  fJetPhi_ = model->MakeField<float>("jet_phi");
  fJetM_ = model->MakeField<float>("jet_m");

  // Provenance kept per-entry (constant columns compress away); the generator
  // tag is the handle on the dominant PYTHIA-vs-HERWIG systematic of the posterior.
  fZcut_ = model->MakeField<float>("z_cut");
  fBeta_ = model->MakeField<float>("beta");
  fKtFloor_ = model->MakeField<float>("kt_floor");
  fGen_ = model->MakeField<std::string>("generator");

  fXi_ = model->MakeField<std::vector<float>>("x_lnInvDelta");
  fXk_ = model->MakeField<std::vector<float>>("x_lnkt");
  fXz_ = model->MakeField<std::vector<float>>("x_lnz");
  fXp_ = model->MakeField<std::vector<float>>("x_psi");

  fYi_ = model->MakeField<std::vector<float>>("y_lnInvDelta");
  fYk_ = model->MakeField<std::vector<float>>("y_lnkt");
  fYz_ = model->MakeField<std::vector<float>>("y_lnz");
  fYp_ = model->MakeField<std::vector<float>>("y_psi");

  // Hadron-level all-branch groomed scalars: CONDITIONING side only (the primary-only
  // x sequence cannot represent either). `x_mg` is the pipeline-groomed jet mass —
  // every primary node is recorded massless, and the secondary prongs that carry the
  // rest of it are discarded at write time; `x_nsec` counts the grooming-passing
  // splittings living on those discarded prongs. See docs/PLAN_Input.md.
  fXmg_ = model->MakeField<float>("x_mg");
  fXnsec_ = model->MakeField<std::uint32_t>("x_nsec");

  writer_ = RNTupleWriter::Recreate(std::move(model), ntuple, path);
}

void LundWriter::fill(std::uint64_t event, std::uint32_t jet_index, double weight,
                      const fastjet::PseudoJet& hadronJet, const LundSeq& x, const LundSeq& y,
                      const JetAux& aux) {
  *fEvent_ = event;
  *fJetIdx_ = jet_index;
  *fWeight_ = weight;

  *fJetPt_ = static_cast<float>(hadronJet.pt());
  *fJetEta_ = static_cast<float>(hadronJet.eta());
  *fJetPhi_ = static_cast<float>(hadronJet.phi_std());
  *fJetM_ = static_cast<float>(hadronJet.m());

  *fZcut_ = static_cast<float>(g_.z_cut);
  *fBeta_ = static_cast<float>(g_.beta);
  *fKtFloor_ = static_cast<float>(g_.kt_floor);
  *fGen_ = generator_;

  *fXi_ = x.lnInvDelta;
  *fXk_ = x.lnkt;
  *fXz_ = x.lnz;
  *fXp_ = x.psi;
  *fYi_ = y.lnInvDelta;
  *fYk_ = y.lnkt;
  *fYz_ = y.lnz;
  *fYp_ = y.psi;

  *fXmg_ = aux.mg;
  // n_all >= n_primary by construction (the spine is a subset of all branches); the
  // guard keeps a hypothetical inconsistency from wrapping around in unsigned arithmetic.
  *fXnsec_ = (aux.n_all >= aux.n_primary) ? (aux.n_all - aux.n_primary) : 0u;

  writer_->Fill();
  ++n_written_;
}

}  // namespace h2p
