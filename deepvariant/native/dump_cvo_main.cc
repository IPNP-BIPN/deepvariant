// Tiny dumper: read a TFRecord stream of CallVariantsOutput protos and
// print one line per record:
//     <chrom>\t<start1>\t<ref>\t<alt0>\t<alt1>...\t<argmax>
// Used during realigner/AlleleCounter parity work to diff our candidate
// set against upstream's. Not shipped in releases.
//
// Usage: dump_cvo <tfrecord-path>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <cvo.tfrecord>\n", argv[0]);
    return 2;
  }
  auto rdr = deepvariant::TFRecordReader::New(argv[1]);
  if (!rdr) {
    std::fprintf(stderr, "failed to open %s\n", argv[1]);
    return 1;
  }
  long total = 0;
  while (rdr->GetNext()) {
    learning::genomics::deepvariant::CallVariantsOutput cvo;
    if (!cvo.ParseFromString(rdr->record())) continue;
    const auto& v = cvo.variant();
    int argmax = 0;
    double best = -1.0;
    for (int i = 0; i < cvo.genotype_probabilities_size(); ++i) {
      if (cvo.genotype_probabilities(i) > best) {
        best = cvo.genotype_probabilities(i);
        argmax = i;
      }
    }
    std::cout << v.reference_name() << '\t' << (v.start() + 1) << '\t'
              << v.reference_bases();
    for (const auto& a : v.alternate_bases()) std::cout << '\t' << a;
    std::cout << '\t' << argmax;
    // Append all probabilities at full precision so we can diff against
    // upstream's intermediate CVOs at the postprocess input layer.
    for (int i = 0; i < cvo.genotype_probabilities_size(); ++i) {
      std::cout << '\t' << std::scientific << std::setprecision(17)
                << cvo.genotype_probabilities(i);
    }
    std::cout << '\n';
    ++total;
  }
  std::fprintf(stderr, "%ld records\n", total);
  return 0;
}
