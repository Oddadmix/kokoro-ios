import Foundation
import Testing
@testable import KokoroSwift

/// Verifies the CATT port against outputs produced by the reference PyTorch
/// implementation (see kikiri-tts scripts/convert_catt_to_mlx.py, which
/// asserts PyTorch ≡ MLX before writing the weights).
///
/// Requires the converted weights; skipped when the file is absent.
@Suite struct ArabicDiacritizerTests {
  static let weightsURL = URL(fileURLWithPath: NSString(
    string: "~/Documents/projects/catt/catt_eo.safetensors").expandingTildeInPath)

  /// (undiacritized input, expected diacritized output from PyTorch)
  static let cases: [(String, String)] = [
    ("مرحبا بك في نبرة", "مَرْحَبًا بِكَ فِي نَبْرَةَ"),
    ("العربية لغة جميلة جدا", "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ جِدًّا"),
    ("خرج الرجل من البيت صباحا", "خَرَجَ الرَّجُلُ مِنْ الْبَيْتِ صَبَاحًا"),
    ("قرأت كتابا عن الذكاء الاصطناعي", "قَرَأْتُ كِتَابًا عَنْ الذَّكَاءِ الِاصْطِنَاعِيِّ"),
  ]

  @Test func matchesPyTorchReference() throws {
    guard FileManager.default.fileExists(atPath: Self.weightsURL.path) else {
      print("catt_eo.safetensors not found — skipping")
      return
    }
    let diacritizer = try ArabicDiacritizer(modelPath: Self.weightsURL)
    for (input, expected) in Self.cases {
      let output = diacritizer.diacritize(input)
      #expect(output == expected, "input: \(input)")
    }
  }

  @Test func detectsDiacritizedText() {
    #expect(ArabicDiacritizer.isDiacritized("مَرْحَبًا بِكَ فِي نَبْرَة"))
    #expect(!ArabicDiacritizer.isDiacritized("مرحبا بك في نبرة"))
    #expect(!ArabicDiacritizer.isDiacritized("hello world"))
  }

  /// Numbers, symbols, and Latin must survive diacritization (agent replies
  /// contain amounts/temperatures that are the actual answer).
  @Test func preservesNumbersAndSymbols() throws {
    guard FileManager.default.fileExists(atPath: Self.weightsURL.path) else {
      print("catt_eo.safetensors not found — skipping"); return
    }
    let d = try ArabicDiacritizer(modelPath: Self.weightsURL)
    let out = d.diacritize("مائة دولار أمريكي = 87.35 يورو")
    #expect(out.contains("87.35"), "numbers dropped: \(out)")
    #expect(out.contains("="), "symbols dropped: \(out)")
    #expect(out.unicodeScalars.contains { (0x064B...0x0652).contains($0.value) },
            "Arabic went undiacritized: \(out)")
  }
}
