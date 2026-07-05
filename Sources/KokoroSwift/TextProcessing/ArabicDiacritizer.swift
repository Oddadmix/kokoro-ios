//
//  KokoroSwift
//
//  Arabic diacritization (tashkeel) using the CATT encoder-only model:
//  "CATT: Character-based Arabic Tashkeel Transformer" (Apache 2.0,
//  https://github.com/abjadai/catt). This is a faithful port of the
//  reference tokenizer (tashkeel_tokenizer.py) and the encoder-only
//  forward pass (eo.py / transformer.py), verified against PyTorch.
//

import Foundation
import MLX
import MLXNN

/// Restores Arabic diacritics (tashkeel) on undiacritized text so the Kokoro
/// Arabic voice receives fully vowelized input. Load with the converted
/// `catt_eo.safetensors` weights (see kikiri-tts scripts/convert_catt_to_mlx.py).
public final class ArabicDiacritizer {
  // MARK: - Model hyperparameters (best_eo_mlm_ns_epoch_193)

  private enum Config {
    static let dModel = 512
    static let nLayers = 6
    static let nHeads = 16
    static let maxSeqLen = 1024
    static let layerNormEps: Float = 1e-12
  }

  // MARK: - Vocabulary (mirrors TashkeelTokenizer)

  /// Buckwalter letter vocabulary: [PAD, BOS, EOS] + 37 letters + [MASK]
  private static let letters: [String] =
    ["<PAD>", "<BOS>", "<EOS>"] +
    [" ", "$", "&", "'", "*", "<", ">", "A", "D", "E", "H", "S", "T", "Y", "Z",
     "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t",
     "v", "w", "x", "y", "z", "|", "}"] +
    ["<MASK>"]

  /// Tashkeel class vocabulary: [PAD, BOS, EOS] + 15 tags
  private static let tashkeelList: [String] =
    ["<PAD>", "<BOS>", "<EOS>"] +
    ["<NT>", "<SD>", "<SDD>", "<SF>", "<SFF>", "<SK>", "<SKK>",
     "F", "K", "N", "a", "i", "o", "u", "~"]

  /// Shaddah-combination tags → their Buckwalter expansion
  private static let tagExpansion: [String: String] = [
    "<SF>": "~a", "<SD>": "~u", "<SK>": "~i",
    "<SFF>": "~F", "<SDD>": "~N", "<SKK>": "~K",
  ]

  /// Single-character tashkeel marks in Buckwalter
  private static let tashkeelChars: Set<Character> = ["F", "N", "K", "a", "u", "i", "~", "o"]

  /// Buckwalter → Arabic Unicode (buck2uni from bw2ar.py)
  private static let buck2uni: [Character: Character] = [
    "'": "\u{0621}", "|": "\u{0622}", ">": "\u{0623}", "&": "\u{0624}",
    "<": "\u{0625}", "}": "\u{0626}", "A": "\u{0627}", "b": "\u{0628}",
    "p": "\u{0629}", "t": "\u{062A}", "v": "\u{062B}", "j": "\u{062C}",
    "H": "\u{062D}", "x": "\u{062E}", "d": "\u{062F}", "*": "\u{0630}",
    "r": "\u{0631}", "z": "\u{0632}", "s": "\u{0633}", "$": "\u{0634}",
    "S": "\u{0635}", "D": "\u{0636}", "T": "\u{0637}", "Z": "\u{0638}",
    "E": "\u{0639}", "g": "\u{063A}", "_": "\u{0640}", "f": "\u{0641}",
    "q": "\u{0642}", "k": "\u{0643}", "l": "\u{0644}", "m": "\u{0645}",
    "n": "\u{0646}", "h": "\u{0647}", "w": "\u{0648}", "Y": "\u{0649}",
    "y": "\u{064A}", "F": "\u{064B}", "N": "\u{064C}", "K": "\u{064D}",
    "a": "\u{064E}", "u": "\u{064F}", "i": "\u{0650}", "~": "\u{0651}",
    "o": "\u{0652}", "`": "\u{0670}", "{": "\u{0671}",
  ]

  private static let uni2buck: [Character: Character] =
    Dictionary(uniqueKeysWithValues: buck2uni.map { ($1, $0) })

  /// Lam-alef ligatures expand to two Buckwalter characters
  private static let ligatures: [Character: String] = [
    "\u{FEFB}": "lA", "\u{FEF7}": "l>", "\u{FEF5}": "l|", "\u{FEF9}": "l<",
  ]

  private static let lettersMap: [String: Int] =
    Dictionary(uniqueKeysWithValues: letters.enumerated().map { ($1, $0) })

  private static let noTashkeelTag = "<NT>"

  // MARK: - Weights

  private let weights: [String: MLXArray]
  private let positionalEncoding: MLXArray

  /// Loads the CATT encoder-only weights (catt_eo.safetensors).
  public init(modelPath: URL) throws {
    weights = try MLX.loadArrays(url: modelPath)

    // Fixed sinusoidal positional encoding, precomputed to maxSeqLen
    var encoding = [Float](repeating: 0, count: Config.maxSeqLen * Config.dModel)
    for pos in 0..<Config.maxSeqLen {
      for i in stride(from: 0, to: Config.dModel, by: 2) {
        let angle = Float(pos) / powf(10000, Float(i) / Float(Config.dModel))
        encoding[pos * Config.dModel + i] = sinf(angle)
        if i + 1 < Config.dModel {
          encoding[pos * Config.dModel + i + 1] = cosf(angle)
        }
      }
    }
    positionalEncoding = MLXArray(encoding, [Config.maxSeqLen, Config.dModel])
  }

  // MARK: - Public API

  /// True when the text already carries enough harakat (≥20% of Arabic
  /// letters marked) that diacritization can be skipped.
  public static func isDiacritized(_ text: String) -> Bool {
    let harakatRange: ClosedRange<UInt32> = 0x064B...0x0652
    let letterRanges: [ClosedRange<UInt32>] = [0x0621...0x063A, 0x0641...0x064A]
    var harakat = 0
    var letters = 0
    for scalar in text.unicodeScalars {
      if harakatRange.contains(scalar.value) { harakat += 1 }
      if letterRanges.contains(where: { $0.contains(scalar.value) }) { letters += 1 }
    }
    guard letters > 0 else { return false }
    return Float(harakat) / Float(letters) >= 0.2
  }

  /// Restores diacritics on Arabic text while preserving everything else.
  /// The text is split into runs of Arabic letters vs. non-Arabic content
  /// (digits, punctuation, Latin); only Arabic runs are diacritized, so
  /// numbers and symbols — e.g. "٨٧٫٣٥", "87.35 EUR" — survive verbatim.
  public func diacritize(_ text: String) -> String {
    var result = ""
    var segment = ""
    var segmentIsArabic: Bool? = nil

    func flush() {
      guard !segment.isEmpty else { return }
      result += (segmentIsArabic == true) ? diacritizeRun(segment) : segment
      segment = ""
    }

    for ch in text {
      let isLetter = ch.unicodeScalars.allSatisfy { Self.isArabicLetter($0) }
      let isSpace = ch == " " || ch == "\n" || ch == "\t"
      if isSpace {
        segment.append(ch)  // whitespace stays in the current run
      } else if isLetter {
        if segmentIsArabic == false { flush() }
        segmentIsArabic = true
        segment.append(ch)
      } else {
        if segmentIsArabic == true { flush() }
        segmentIsArabic = false
        segment.append(ch)
      }
    }
    flush()
    return result
  }

  /// True for Arabic letters/diacritics the CATT model handles (NOT digits).
  private static func isArabicLetter(_ s: Unicode.Scalar) -> Bool {
    switch s.value {
    case 0x0621...0x063A, 0x0641...0x0652, 0x0670, 0x0671,
         0xFEFB, 0xFEF7, 0xFEF5, 0xFEF9:
      return true
    default:
      return false
    }
  }

  /// Diacritizes a run that is known to be Arabic letters + spaces.
  /// Leading/trailing whitespace is preserved (toBuckwalter trims it).
  private func diacritizeRun(_ text: String) -> String {
    let ws = { (c: Character) in c == " " || c == "\n" || c == "\t" }
    let leading = String(text.prefix(while: ws))
    let trailing = String(text.reversed().prefix(while: ws).reversed())

    let bwChars = Self.toBuckwalter(text)
    guard !bwChars.isEmpty else { return text }

    // Truncate defensively to the positional-encoding capacity
    let chars = Array(bwChars.prefix(Config.maxSeqLen - 2))
    let inputIds = chars.map { Self.lettersMap[String($0)]! }

    let predicted = forward(inputIds: inputIds)

    // Force space positions to the no-tashkeel tag (as do_tashkeel_batch does)
    let spaceId = Self.lettersMap[" "]!
    var tags: [String] = []
    for (i, id) in inputIds.enumerated() {
      let classIndex = id == spaceId
        ? Self.tashkeelList.firstIndex(of: Self.noTashkeelTag)!
        : Int(predicted[i])
      tags.append(Self.tashkeelList[classIndex])
    }

    // Combine letters with tashkeel and transliterate back to Arabic
    var bwOut = ""
    for (ch, tag) in zip(chars, tags) {
      bwOut.append(ch)
      if let expansion = Self.tagExpansion[tag] {
        bwOut += expansion
      } else if tag.count == 1 {  // single tashkeel char; special tags are skipped
        bwOut += tag
      }
    }
    return leading + String(bwOut.map { Self.buck2uni[$0] ?? $0 }) + trailing
  }

  // MARK: - Text preparation (mirrors clean_text + ar2bw)

  /// Cleans the text and transliterates Arabic → Buckwalter, keeping only
  /// characters in the model's letter vocabulary.
  private static func toBuckwalter(_ text: String) -> [Character] {
    // clean_text: strip tatweel, normalize wasla, keep only Arabic block + space
    var cleaned = ""
    for scalar in text.unicodeScalars {
      switch scalar.value {
      case 0x0640:  // tatweel — dropped
        continue
      case 0x0671:  // alef wasla → bare alef
        cleaned.append("\u{0627}")
      case 0x0621...0x063A, 0x0641...0x0652, 0x0670,
           0xFEFB, 0xFEF7, 0xFEF5, 0xFEF9:
        cleaned.append(Character(scalar))
      default:
        cleaned.append(" ")
      }
    }
    let normalized = cleaned.split(separator: " ").joined(separator: " ")

    // ar2bw transliteration (ligatures expand to two chars); drop dagger alef
    var bw = ""
    for ch in normalized {
      if let lig = ligatures[ch] {
        bw += lig
      } else if let mapped = uni2buck[ch], mapped != "`" {
        bw.append(mapped)
      } else if ch == " " {
        bw.append(" ")
      }
    }

    // Unify shaddah-harakah order (shaddah first)
    for mark in ["a", "u", "i", "F", "N", "K"] {
      bw = bw.replacingOccurrences(of: "\(mark)~", with: "~\(mark)")
    }
    // Collapse duplicated harakat
    for mark in ["F", "N", "K", "a", "u", "i", "~", "o"] {
      bw = bw.replacingOccurrences(of: "\(mark)\(mark)", with: mark)
    }

    // Keep letters only (any stray tashkeel in the input is dropped so the
    // model predicts a fresh, complete set)
    return bw.filter { lettersMap[String($0)] != nil && !tashkeelChars.contains($0) }
  }

  // MARK: - Encoder forward pass (mirrors eo.py / transformer.py)

  /// Runs the encoder and returns the argmax tashkeel class per character.
  private func forward(inputIds: [Int]) -> [Int32] {
    let seqLen = inputIds.count
    let tokEmb = weights["encoder.emb.tok_emb.weight"]!
    let ids = MLXArray(inputIds.map { Int32($0) })

    var x = MLX.take(tokEmb, ids, axis: 0) + positionalEncoding[0..<seqLen]

    let headDim = Config.dModel / Config.nHeads
    let scale = MLXArray(sqrtf(Float(headDim)))

    for layer in 0..<Config.nLayers {
      let p = "encoder.layers.\(layer)."

      // Multi-head self-attention (projections have no bias; post-norm).
      // Single unpadded sequence → no attention mask required.
      let q = MLX.matmul(x, weights[p + "attention.w_q.weight"]!.transposed())
      let k = MLX.matmul(x, weights[p + "attention.w_k.weight"]!.transposed())
      let v = MLX.matmul(x, weights[p + "attention.w_v.weight"]!.transposed())

      func splitHeads(_ t: MLXArray) -> MLXArray {
        t.reshaped([seqLen, Config.nHeads, headDim]).transposed(1, 0, 2)
      }
      let scores = MLX.matmul(splitHeads(q), splitHeads(k).transposed(0, 2, 1)) / scale
      var attn = MLX.matmul(MLX.softmax(scores, axis: -1), splitHeads(v))
      attn = attn.transposed(1, 0, 2).reshaped([seqLen, Config.dModel])
      attn = MLX.matmul(attn, weights[p + "attention.w_concat.weight"]!.transposed())

      x = layerNorm(x + attn,
                    gamma: weights[p + "norm1.gamma"]!,
                    beta: weights[p + "norm1.beta"]!)

      // Position-wise feed-forward (ReLU)
      var h = MLX.matmul(x, weights[p + "ffn.linear1.weight"]!.transposed())
        + weights[p + "ffn.linear1.bias"]!
      h = MLX.maximum(h, MLXArray(Float(0)))
      h = MLX.matmul(h, weights[p + "ffn.linear2.weight"]!.transposed())
        + weights[p + "ffn.linear2.bias"]!

      x = layerNorm(x + h,
                    gamma: weights[p + "norm2.gamma"]!,
                    beta: weights[p + "norm2.beta"]!)
    }

    let logits = MLX.matmul(x, weights["decoder.weight"]!.transposed())
      + weights["decoder.bias"]!
    return MLX.argMax(logits, axis: -1).asArray(Int32.self)
  }

  private func layerNorm(_ x: MLXArray, gamma: MLXArray, beta: MLXArray) -> MLXArray {
    let mean = MLX.mean(x, axes: [-1], keepDims: true)
    let variance = MLX.variance(x, axes: [-1], keepDims: true)
    return gamma * (x - mean) / MLX.sqrt(variance + Config.layerNormEps) + beta
  }
}
