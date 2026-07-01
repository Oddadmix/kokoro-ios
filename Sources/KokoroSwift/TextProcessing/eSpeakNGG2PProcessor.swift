//
//  KokoroSwift
//

#if canImport(eSpeakNGLib)

import Foundation
import eSpeakNGLib

/// A G2P processor that uses the eSpeak NG library for phonemization.
/// Requires the eSpeakNGLib framework to be available at compile time.
final class eSpeakNGG2PProcessor : G2PProcessor {
  /// The underlying eSpeak NG engine instance.
  /// This property is initialized when `setLanguage(_:)` is called and remains
  /// `nil` until the processor is properly configured.
  private var eSpeakEngine: eSpeakNG?

  /// Configures the processor for the specified language.
  /// - Parameter language: The target language for phonemization.
  /// - Throws: `G2PProcessorError.unsupportedLanguage` if the language is not supported by eSpeak NG.
  func setLanguage(_ language: Language) throws {
    eSpeakEngine = try eSpeakNG()
    
    if let language = eSpeakNG.Language(rawValue: language.rawValue), let eSpeakEngine {
      try eSpeakEngine.setLanguage(language: language)
    } else {
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  /// Converts input text to phonetic representation.
  /// - Parameter input: The text string to be converted to phonemes.
  /// - Returns: A phonetic string representation of the input text.
  /// - Throws: `G2PProcessorError.processorNotInitialized` if `setLanguage(_:)` has not been called.
  func process(input: String) throws -> (String, [MToken]?) {
    guard let eSpeakEngine else { throw G2PProcessorError.processorNotInitialized }
    let phonemizedText = try eSpeakEngine.phonemize(text: input)
    return (Self.cleanPhonemes(phonemizedText), nil)
  }

  /// Mirror of the training front-end's `clean_phonemes` (scripts/arabic_g2p.py).
  /// espeak emits a syllable-boundary '.' mid-word which Kokoro's vocab maps to
  /// the sentence-period token (id 4) → phantom pauses. Strip those (a '.' with a
  /// non-whitespace neighbour on BOTH sides) while keeping sentence-final '.'.
  /// The dental/pharyngealization marks and bracket markers aren't in the Kokoro
  /// vocab, so `Tokenizer.tokenize` already drops them — we remove them here too
  /// so the returned phoneme string shown to callers is clean. `ʕ`/`ħ` are KEPT
  /// (they live on vocab slots 7/8).
  static func cleanPhonemes(_ s: String) -> String {
    var out = s.replacingOccurrences(
      of: "(?<=\\S)\\.(?=\\S)", with: "", options: .regularExpression)
    for marker in ["\u{032A}", "\u{02E4}", "[", "]", "{", "}"] {  // ̪  ˤ  [ ] { }
      out = out.replacingOccurrences(of: marker, with: "")
    }
    return out
  }
}

#endif
