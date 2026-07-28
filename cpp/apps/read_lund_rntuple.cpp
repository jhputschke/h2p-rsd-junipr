// read_lund_rntuple — inspect a jets.root RNTuple written by write_lund_rntuple.
//
// Prints: the schema, the grooming/generator provenance (the "meta" information,
// written per-entry and constant by construction), the number of jets, and the
// full first-jet record (kinematics + the hadron-level x and parton-level y
// primary Lund sequences). ROOT-only (RNTuple); no FastJet/PYTHIA needed.
//
// Usage:  read_lund_rntuple [jets.root] [Jets]

#include <ROOT/RNTupleReader.hxx>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using ROOT::RNTupleReader;

namespace {

void printSeq(const std::string& title, const std::vector<float>& lnInvDelta,
              const std::vector<float>& lnkt, const std::vector<float>& lnz,
              const std::vector<float>& psi) {
  std::cout << "    " << title << " (" << lnInvDelta.size() << " primary splittings):\n";
  for (std::size_t i = 0; i < lnInvDelta.size(); ++i) {
    const float u = lnInvDelta[i], v = lnkt[i], z = lnz[i], p = psi[i];
    std::cout << "      [" << i << "] "
              << "ln1/DR=" << std::setw(6) << u << "  lnkt=" << std::setw(6) << v
              << "  lnz=" << std::setw(7) << z << "  psi=" << std::setw(7) << p
              << "   (kt=" << std::setw(7) << std::exp(v) << " GeV, DeltaR=" << std::exp(-u)
              << ", z=" << std::exp(z) << ")\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = (argc > 1) ? argv[1] : "jets.root";
  const std::string ntuple = (argc > 2) ? argv[2] : "Jets";

  std::unique_ptr<RNTupleReader> reader;
  try {
    reader = RNTupleReader::Open(ntuple, path);
  } catch (const std::exception& e) {
    std::cerr << "[read_lund_rntuple] could not open " << path << ":" << ntuple << " (" << e.what()
              << ")\n";
    return 1;
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "file:   " << path << "\n";
  std::cout << "ntuple: " << ntuple << "\n";

  // ---- schema (all field names/types) ------------------------------------
  std::cout << "\n--- schema ---\n";
  const auto& desc = reader->GetDescriptor();
  std::set<std::string> present;
  for (const auto& field : desc.GetTopLevelFields()) {
    present.insert(field.GetFieldName());
    std::cout << "  " << std::left << std::setw(16) << field.GetFieldName() << std::right
              << field.GetTypeName() << "\n";
  }
  // The aux conditioning columns post-date the original schema; a file written before
  // docs/PLAN_Input.md simply lacks them and must still read (RNTuple compatibility).
  const bool has_aux = present.count("x_mg") && present.count("x_nsec");

  const auto nJets = reader->GetNEntries();
  std::cout << "\nnumber of jets: " << nJets << "\n";
  if (nJets == 0) {
    std::cout << "(empty ntuple)\n";
    return 0;
  }

  // ---- field views -------------------------------------------------------
  auto vEvent = reader->GetView<std::uint64_t>("event");
  auto vJetIdx = reader->GetView<std::uint32_t>("jet_index");
  auto vWeight = reader->GetView<double>("weight");
  auto vPt = reader->GetView<float>("jet_pt");
  auto vEta = reader->GetView<float>("jet_eta");
  auto vPhi = reader->GetView<float>("jet_phi");
  auto vM = reader->GetView<float>("jet_m");
  auto vZcut = reader->GetView<float>("z_cut");
  auto vBeta = reader->GetView<float>("beta");
  auto vKtFloor = reader->GetView<float>("kt_floor");
  auto vGen = reader->GetView<std::string>("generator");
  auto vXi = reader->GetView<std::vector<float>>("x_lnInvDelta");
  auto vXk = reader->GetView<std::vector<float>>("x_lnkt");
  auto vXz = reader->GetView<std::vector<float>>("x_lnz");
  auto vXp = reader->GetView<std::vector<float>>("x_psi");
  auto vYi = reader->GetView<std::vector<float>>("y_lnInvDelta");
  auto vYk = reader->GetView<std::vector<float>>("y_lnkt");
  auto vYz = reader->GetView<std::vector<float>>("y_lnz");
  auto vYp = reader->GetView<std::vector<float>>("y_psi");

  // ---- meta / provenance (per-entry, constant by construction) -----------
  std::cout << "\n--- meta / provenance (from entry 0; written per-entry) ---\n";
  std::cout << "  generator : " << vGen(0) << "\n";
  std::cout << "  z_cut     : " << vZcut(0) << "\n";
  std::cout << "  beta      : " << vBeta(0) << "\n";
  std::cout << "  kt_floor  : " << vKtFloor(0) << " GeV\n";

  // ---- first jet ---------------------------------------------------------
  std::cout << "\n--- first jet (entry 0) ---\n";
  std::cout << "  event=" << vEvent(0) << "  jet_index=" << vJetIdx(0) << "  weight=" << vWeight(0)
            << "\n";
  std::cout << "  jet_pt=" << vPt(0) << " GeV  eta=" << vEta(0) << "  phi=" << vPhi(0)
            << "  m=" << vM(0) << " GeV\n";
  if (has_aux) {
    auto vXmg = reader->GetView<float>("x_mg");
    auto vXnsec = reader->GetView<std::uint32_t>("x_nsec");
    std::cout << "  aux (hadron-level, conditioning side): x_mg=" << vXmg(0)
              << " GeV  x_nsec=" << vXnsec(0) << " secondary passing splittings\n";
  } else {
    std::cout << "  aux: none (pre-PLAN_Input file: no x_mg / x_nsec columns)\n";
  }
  printSeq("x  hadron-level", vXi(0), vXk(0), vXz(0), vXp(0));
  printSeq("y  parton-level", vYi(0), vYk(0), vYz(0), vYp(0));

  return 0;
}
