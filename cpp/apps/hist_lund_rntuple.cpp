// hist_lund_rntuple — histogram a jets.root RNTuple written by write_lund_rntuple.
//
// Produces a ROOT file with:
//   jet/      jet pt (linear + log-binned), eta, phi, mass
//   lund_x/   hadron-level primary Lund: inclusive ln(1/DR), ln(kt), ln(z), psi,
//   lund_y/   parton-level primary Lund: the multiplicity, the 2D Lund plane, and
//             one full set of the four observables PER SPLITTING INDEX (split00, ...)
//   aux/      every aux (conditioning-side) column: x_mg, x_ptg, x_nsec,
//             x_kt_sec_max, x_kt_sec_sum, x_sec_attach
//   meta/     provenance strings (generator, grooming params) copied from entry 0
//
// All histograms are filled with the per-jet `weight` column and carry Sumw2.
// ROOT-only (RNTuple); no FastJet/PYTHIA needed.
//
// Usage:  hist_lund_rntuple [in.root] [out.root] [options]
//   --ntuple NAME   RNTuple name                       (default: Jets)
//   --nsplit N      per-splitting histograms for index 0..N-1  (default: 8)
//   --ptmax GEV     upper edge of the jet_pt / x_ptg axes      (default: 1000)
//   --mmax GEV      upper edge of the jet_m / x_mg axes        (default: 200)

#include <ROOT/RNTupleReader.hxx>

#include <TDirectory.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNamed.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using ROOT::RNTupleReader;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Axis conventions shared by the hadron- (x) and parton-level (y) blocks, so the two
// are bin-compatible and can be divided/subtracted downstream without rebinning.
constexpr int kNbLnInvDelta = 100;
constexpr double kLnInvDeltaLo = 0.0, kLnInvDeltaHi = 10.0;
constexpr int kNbLnkt = 120;
constexpr double kLnktLo = -4.0, kLnktHi = 8.0;
constexpr int kNbLnz = 100;
constexpr double kLnzLo = -10.0, kLnzHi = 0.0;
constexpr int kNbPsi = 100;
constexpr int kNbMult = 51;  // 0..50 splittings, integer-centred

TH1F* mk1(TDirectory* dir, const std::string& name, const std::string& title, int nb, double lo,
          double hi) {
  auto* h = new TH1F(name.c_str(), title.c_str(), nb, lo, hi);
  h->Sumw2();
  h->SetDirectory(dir);
  return h;
}

TH2F* mk2(TDirectory* dir, const std::string& name, const std::string& title, int nbx, double xlo,
          double xhi, int nby, double ylo, double yhi) {
  auto* h = new TH2F(name.c_str(), title.c_str(), nbx, xlo, xhi, nby, ylo, yhi);
  h->Sumw2();
  h->SetDirectory(dir);
  return h;
}

// The four primary-Lund observables of one splitting population (either "all
// splittings pooled" or "the n-th splitting only").
struct LundHists {
  TH1F* lnInvDelta = nullptr;
  TH1F* lnkt = nullptr;
  TH1F* lnz = nullptr;
  TH1F* psi = nullptr;

  static LundHists book(TDirectory* dir, const std::string& tag, const std::string& what) {
    LundHists h;
    h.lnInvDelta = mk1(dir, "h_" + tag + "_lnInvDelta", what + " ln(1/#DeltaR);ln(1/#DeltaR);jets",
                       kNbLnInvDelta, kLnInvDeltaLo, kLnInvDeltaHi);
    h.lnkt = mk1(dir, "h_" + tag + "_lnkt", what + " ln(k_{t}/GeV);ln(k_{t}/GeV);jets", kNbLnkt,
                 kLnktLo, kLnktHi);
    h.lnz = mk1(dir, "h_" + tag + "_lnz", what + " ln(z);ln(z);jets", kNbLnz, kLnzLo, kLnzHi);
    h.psi = mk1(dir, "h_" + tag + "_psi", what + " #psi;#psi;jets", kNbPsi, -kPi, kPi);
    return h;
  }

  void fill(float u, float k, float z, float p, double w) const {
    lnInvDelta->Fill(u, w);
    lnkt->Fill(k, w);
    lnz->Fill(z, w);
    psi->Fill(p, w);
  }
};

// One declustering level (x = hadron, y = parton): pooled observables, the
// multiplicity, the 2D Lund plane, and the per-splitting-index breakdown.
struct LevelHists {
  LundHists all;
  std::vector<LundHists> per_split;
  TH1F* mult = nullptr;
  TH2F* plane = nullptr;

  static LevelHists book(TDirectory* parent, const std::string& tag, const std::string& what,
                         int nsplit) {
    LevelHists L;
    L.all = LundHists::book(parent, tag + "_all", what + ", all splittings:");
    L.mult = mk1(parent, "h_" + tag + "_nsplit",
                 what + " primary multiplicity;n_{splittings};jets", kNbMult, -0.5,
                 kNbMult - 0.5);
    L.plane = mk2(parent, "h_" + tag + "_plane",
                  what + " primary Lund plane;ln(1/#DeltaR);ln(k_{t}/GeV)", kNbLnInvDelta,
                  kLnInvDeltaLo, kLnInvDeltaHi, kNbLnkt, kLnktLo, kLnktHi);
    L.per_split.reserve(nsplit);
    for (int n = 0; n < nsplit; ++n) {
      char sub[32], lbl[64];
      std::snprintf(sub, sizeof(sub), "split%02d", n);
      std::snprintf(lbl, sizeof(lbl), "%s, splitting %d:", what.c_str(), n);
      TDirectory* d = parent->mkdir(sub);
      char t[32];
      std::snprintf(t, sizeof(t), "%s_s%02d", tag.c_str(), n);
      L.per_split.push_back(LundHists::book(d, t, lbl));
    }
    return L;
  }

  // Returns the number of splittings that fell past the per-splitting cap.
  std::size_t fill(const std::vector<float>& u, const std::vector<float>& k,
                   const std::vector<float>& z, const std::vector<float>& p, double w) const {
    mult->Fill(static_cast<double>(u.size()), w);
    for (std::size_t i = 0; i < u.size(); ++i) {
      all.fill(u[i], k[i], z[i], p[i], w);
      plane->Fill(u[i], k[i], w);
      if (i < per_split.size()) per_split[i].fill(u[i], k[i], z[i], p[i], w);
    }
    return (u.size() > per_split.size()) ? (u.size() - per_split.size()) : 0;
  }
};

[[noreturn]] void usage(int code) {
  std::cerr << "usage: hist_lund_rntuple [in.root] [out.root] [--ntuple NAME] [--nsplit N]\n"
               "                         [--ptmax GEV] [--mmax GEV]\n";
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  std::string in = "jets.root", out = "lund_hists.root", ntuple = "Jets";
  int nsplit = 8;
  double ptmax = 1000.0, mmax = 200.0;

  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "[hist_lund_rntuple] " << what << " needs a value\n";
        usage(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") usage(0);
    else if (a == "--ntuple") ntuple = next("--ntuple");
    else if (a == "--nsplit") nsplit = std::stoi(next("--nsplit"));
    else if (a == "--ptmax") ptmax = std::stod(next("--ptmax"));
    else if (a == "--mmax") mmax = std::stod(next("--mmax"));
    else if (!a.empty() && a[0] == '-') { std::cerr << "[hist_lund_rntuple] unknown option " << a << "\n"; usage(2); }
    else pos.push_back(a);
  }
  if (pos.size() > 2) usage(2);
  if (pos.size() > 0) in = pos[0];
  if (pos.size() > 1) out = pos[1];
  if (nsplit < 0) { std::cerr << "[hist_lund_rntuple] --nsplit must be >= 0\n"; return 2; }

  std::unique_ptr<RNTupleReader> reader;
  try {
    reader = RNTupleReader::Open(ntuple, in);
  } catch (const std::exception& e) {
    std::cerr << "[hist_lund_rntuple] could not open " << in << ":" << ntuple << " (" << e.what()
              << ")\n";
    return 1;
  }

  // The aux conditioning columns post-date the original schema; a file written before
  // docs/PLAN_Input.md simply lacks them and must still histogram (RNTuple compatibility).
  std::set<std::string> present;
  for (const auto& f : reader->GetDescriptor().GetTopLevelFields()) present.insert(f.GetFieldName());
  const bool has_aux = present.count("x_mg") && present.count("x_nsec");
  const bool has_ptg = present.count("x_ptg");
  const bool has_sec = present.count("x_kt_sec_max") && present.count("x_kt_sec_sum") &&
                       present.count("x_sec_attach");

  const auto nJets = reader->GetNEntries();
  std::cout << "[hist_lund_rntuple] " << in << ":" << ntuple << " -> " << out << "  (" << nJets
            << " jets, aux " << (has_aux ? "present" : "absent") << ")\n";

  std::unique_ptr<TFile> file(TFile::Open(out.c_str(), "RECREATE"));
  if (!file || file->IsZombie()) {
    std::cerr << "[hist_lund_rntuple] could not create " << out << "\n";
    return 1;
  }

  // ---- book ---------------------------------------------------------------
  TDirectory* dJet = file->mkdir("jet");
  auto* hPt = mk1(dJet, "h_jet_pt", "jet p_{T};p_{T} [GeV];jets", 200, 0.0, ptmax);
  // Log-binned twin: the pt spectrum spans decades and the jet-pt dependence of the
  // observables (docs/PLAN_jet_xsection.md) is read off a log axis.
  std::vector<double> ptEdges(201);
  {
    const double lo = std::log10(10.0), hi = std::log10(std::max(ptmax, 20.0));
    for (int i = 0; i <= 200; ++i) ptEdges[i] = std::pow(10.0, lo + (hi - lo) * i / 200.0);
  }
  auto* hPtLog = new TH1F("h_jet_pt_log", "jet p_{T} (log bins);p_{T} [GeV];jets", 200,
                          ptEdges.data());
  hPtLog->Sumw2();
  hPtLog->SetDirectory(dJet);
  auto* hEta = mk1(dJet, "h_jet_eta", "jet #eta;#eta;jets", 100, -5.0, 5.0);
  auto* hPhi = mk1(dJet, "h_jet_phi", "jet #phi;#phi;jets", 100, -kPi, kPi);
  auto* hM = mk1(dJet, "h_jet_m", "jet mass;m [GeV];jets", 200, 0.0, mmax);

  LevelHists hx = LevelHists::book(file->mkdir("lund_x"), "x", "hadron level", nsplit);
  LevelHists hy = LevelHists::book(file->mkdir("lund_y"), "y", "parton level", nsplit);

  TH1F *hMg = nullptr, *hPtg = nullptr, *hNsec = nullptr;
  TH1F *hKtMax = nullptr, *hKtSum = nullptr, *hAttach = nullptr;
  if (has_aux) {
    TDirectory* dAux = file->mkdir("aux");
    hMg = mk1(dAux, "h_x_mg", "groomed jet mass;m_{g} [GeV];jets", 200, 0.0, mmax);
    hNsec = mk1(dAux, "h_x_nsec", "secondary passing splittings;n_{sec};jets", kNbMult, -0.5,
                kNbMult - 0.5);
    if (has_ptg)
      hPtg = mk1(dAux, "h_x_ptg", "groomed jet p_{T};p_{T,g} [GeV];jets", 200, 0.0, ptmax);
    if (has_sec) {
      // x_nsec == 0 leaves these three undefined (they are written as 0, not measured),
      // so they are filled only for jets with at least one off-spine passing splitting.
      hKtMax = mk1(dAux, "h_x_kt_sec_max",
                   "hardest secondary k_{t} (x_nsec>0 only);k_{t}^{sec,max} [GeV];jets", 200, 0.0,
                   100.0);
      hKtSum = mk1(dAux, "h_x_kt_sec_sum",
                   "summed secondary k_{t} (x_nsec>0 only);#Sigma k_{t}^{sec} [GeV];jets", 200, 0.0,
                   200.0);
      hAttach = mk1(dAux, "h_x_sec_attach",
                    "primary node of hardest secondary (x_nsec>0 only);node index;jets", kNbMult,
                    -0.5, kNbMult - 0.5);
    }
  }

  if (nJets == 0) {
    std::cout << "[hist_lund_rntuple] empty ntuple; writing empty histograms\n";
    file->Write();
    return 0;
  }

  // ---- read ---------------------------------------------------------------
  auto vWeight = reader->GetView<double>("weight");
  auto vPt = reader->GetView<float>("jet_pt");
  auto vEta = reader->GetView<float>("jet_eta");
  auto vPhi = reader->GetView<float>("jet_phi");
  auto vM = reader->GetView<float>("jet_m");
  auto vXi = reader->GetView<std::vector<float>>("x_lnInvDelta");
  auto vXk = reader->GetView<std::vector<float>>("x_lnkt");
  auto vXz = reader->GetView<std::vector<float>>("x_lnz");
  auto vXp = reader->GetView<std::vector<float>>("x_psi");
  auto vYi = reader->GetView<std::vector<float>>("y_lnInvDelta");
  auto vYk = reader->GetView<std::vector<float>>("y_lnkt");
  auto vYz = reader->GetView<std::vector<float>>("y_lnz");
  auto vYp = reader->GetView<std::vector<float>>("y_psi");

  // Provenance, per-entry and constant by construction (see lund_writer.cpp).
  {
    TDirectory* dMeta = file->mkdir("meta");
    auto put = [&](const char* k, const std::string& v) {
      dMeta->Add(new TNamed(k, v.c_str()));
    };
    put("generator", reader->GetView<std::string>("generator")(0));
    put("z_cut", std::to_string(reader->GetView<float>("z_cut")(0)));
    put("beta", std::to_string(reader->GetView<float>("beta")(0)));
    put("kt_floor", std::to_string(reader->GetView<float>("kt_floor")(0)));
    put("source", in + ":" + ntuple);
    put("n_jets", std::to_string(nJets));
  }

  std::size_t droppedX = 0, droppedY = 0, nsecPos = 0;
  for (std::uint64_t i = 0; i < nJets; ++i) {
    const double w = vWeight(i);
    hPt->Fill(vPt(i), w);
    hPtLog->Fill(vPt(i), w);
    hEta->Fill(vEta(i), w);
    hPhi->Fill(vPhi(i), w);
    hM->Fill(vM(i), w);

    droppedX += hx.fill(vXi(i), vXk(i), vXz(i), vXp(i), w);
    droppedY += hy.fill(vYi(i), vYk(i), vYz(i), vYp(i), w);
  }

  if (has_aux) {
    auto vMg = reader->GetView<float>("x_mg");
    auto vNsec = reader->GetView<std::uint32_t>("x_nsec");
    for (std::uint64_t i = 0; i < nJets; ++i) {
      const double w = vWeight(i);
      hMg->Fill(vMg(i), w);
      hNsec->Fill(static_cast<double>(vNsec(i)), w);
      if (vNsec(i) > 0) ++nsecPos;
    }
    if (has_ptg) {
      auto vPtg = reader->GetView<float>("x_ptg");
      for (std::uint64_t i = 0; i < nJets; ++i) hPtg->Fill(vPtg(i), vWeight(i));
    }
    if (has_sec) {
      auto vNsec2 = reader->GetView<std::uint32_t>("x_nsec");
      auto vMax = reader->GetView<float>("x_kt_sec_max");
      auto vSum = reader->GetView<float>("x_kt_sec_sum");
      auto vAtt = reader->GetView<std::uint32_t>("x_sec_attach");
      for (std::uint64_t i = 0; i < nJets; ++i) {
        if (vNsec2(i) == 0) continue;  // undefined, not zero
        const double w = vWeight(i);
        hKtMax->Fill(vMax(i), w);
        hKtSum->Fill(vSum(i), w);
        hAttach->Fill(static_cast<double>(vAtt(i)), w);
      }
    }
  }

  file->Write();

  std::cout << "[hist_lund_rntuple] per-splitting histograms: index 0.." << (nsplit - 1) << "\n";
  if (droppedX || droppedY)
    std::cout << "[hist_lund_rntuple] splittings beyond the --nsplit cap (pooled into *_all and "
                 "the plane, but NOT in any splitNN dir): x=" << droppedX << "  y=" << droppedY
              << "  (raise --nsplit to cover them)\n";
  if (has_aux && has_sec)
    std::cout << "[hist_lund_rntuple] secondary-kt histograms filled from " << nsecPos << " / "
              << nJets << " jets with x_nsec > 0\n";
  std::cout << "[hist_lund_rntuple] wrote " << out << "\n";
  return 0;
}
