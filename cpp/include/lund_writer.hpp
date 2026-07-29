// lund_writer.hpp — RNTuple serialization of matched (x, y) primary Lund
// sequences. One entry == one jet; by design there is NO node-level x<->y
// correspondence (the objective is jet-level). A one-entry "Meta" RNTuple records
// generator/tune/pipeline params for the §8 systematics step.
#pragma once

#include "lund_io.hpp"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <cstdint>
#include <memory>
#include <string>

namespace h2p {

class LundWriter {
 public:
  LundWriter(const std::string& path, const std::string& ntuple, const GroomParams& g,
             std::string generator);

  // Append one jet entry (the writer flushes & closes on destruction). `aux` is the
  // hadron-level all-branch summary (docs/PLAN_Input.md); it feeds the CONDITIONING
  // side only — the parton-level target y is untouched by it.
  void fill(std::uint64_t event, std::uint32_t jet_index, double weight,
            const fastjet::PseudoJet& hadronJet, const LundSeq& x, const LundSeq& y,
            const JetAux& aux);

  std::uint64_t nWritten() const { return n_written_; }

 private:
  GroomParams g_;
  std::string generator_;
  std::uint64_t n_written_ = 0;
  std::unique_ptr<ROOT::RNTupleWriter> writer_;

  std::shared_ptr<std::uint64_t> fEvent_;
  std::shared_ptr<std::uint32_t> fJetIdx_;
  std::shared_ptr<double> fWeight_;
  std::shared_ptr<float> fJetPt_, fJetEta_, fJetPhi_, fJetM_;
  std::shared_ptr<float> fZcut_, fBeta_, fKtFloor_, fKtFloorSec_;
  std::shared_ptr<std::string> fGen_;
  std::shared_ptr<std::vector<float>> fXi_, fXk_, fXz_, fXp_;
  std::shared_ptr<std::vector<float>> fYi_, fYk_, fYz_, fYp_;
  std::shared_ptr<float> fXmg_, fXptg_, fXktSecMax_, fXktSecSum_;
  std::shared_ptr<std::uint32_t> fXnsec_, fXsecAttach_;
};

}  // namespace h2p
