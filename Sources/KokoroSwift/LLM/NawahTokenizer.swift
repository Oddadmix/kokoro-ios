//
//  KokoroSwift
//
//  Byte-level BPE tokenizer (GPT-2 style) loaded from a Hugging Face
//  tokenizer.json, as used by the Nawah Arabic chat LLM. Supports the
//  ByteLevel pre-tokenizer with the standard GPT-2 split regex.
//

import Foundation

/// Byte-level BPE tokenizer for the Nawah LLM. Special tokens are NOT split
/// from text — build prompts by inserting their ids around encoded segments.
final class NawahTokenizer {
  enum TokenizerError: Error {
    case malformedTokenizerJSON
  }

  static let imStartId = 32000  // <|im_start|>
  static let imEndId = 32001    // <|im_end|> (also EOS)

  private let vocab: [String: Int]
  private let idToToken: [Int: String]
  private let mergeRanks: [String: Int]  // "left right" → rank
  private let specialIds: Set<Int>

  /// GPT-2 byte ↔ unicode tables: every byte maps to a printable character.
  private static let byteToChar: [UInt8: Character] = {
    var bytes: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
    var chars: [Int] = bytes
    var n = 0
    for b in 0...255 where !bytes.contains(b) {
      bytes.append(b)
      chars.append(256 + n)
      n += 1
    }
    var table: [UInt8: Character] = [:]
    for (b, c) in zip(bytes, chars) {
      table[UInt8(b)] = Character(UnicodeScalar(c)!)
    }
    return table
  }()

  private static let charToByte: [Character: UInt8] =
    Dictionary(uniqueKeysWithValues: byteToChar.map { ($1, $0) })

  /// Standard GPT-2 / ByteLevel(use_regex: true) split pattern
  private static let splitRegex = try! NSRegularExpression(
    pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
  )

  init(tokenizerPath: URL) throws {
    let data = try Data(contentsOf: tokenizerPath)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let model = json["model"] as? [String: Any],
          let vocabDict = model["vocab"] as? [String: Int],
          let merges = model["merges"] as? [Any]
    else {
      throw TokenizerError.malformedTokenizerJSON
    }

    var fullVocab = vocabDict
    var special: Set<Int> = []
    if let added = json["added_tokens"] as? [[String: Any]] {
      for entry in added {
        if let content = entry["content"] as? String, let id = entry["id"] as? Int {
          fullVocab[content] = id
          special.insert(id)
        }
      }
    }
    vocab = fullVocab
    idToToken = Dictionary(uniqueKeysWithValues: fullVocab.map { ($1, $0) })
    specialIds = special

    var ranks: [String: Int] = [:]
    for (rank, merge) in merges.enumerated() {
      // merges appear either as "a b" strings or ["a", "b"] pairs
      if let s = merge as? String {
        ranks[s] = rank
      } else if let pair = merge as? [String], pair.count == 2 {
        ranks["\(pair[0]) \(pair[1])"] = rank
      }
    }
    mergeRanks = ranks
  }

  // MARK: - Encoding

  /// Encodes plain text (no special-token splitting).
  func encode(_ text: String) -> [Int] {
    var ids: [Int] = []
    let ns = text as NSString
    let matches = Self.splitRegex.matches(in: text, range: NSRange(location: 0, length: ns.length))
    for match in matches {
      let piece = ns.substring(with: match.range)
      let mapped = String(piece.utf8.map { Self.byteToChar[$0]! })
      for token in bpe(mapped) {
        if let id = vocab[token] {
          ids.append(id)
        }
      }
    }
    return ids
  }

  /// Decodes ids back to text, skipping special tokens.
  func decode(_ ids: [Int]) -> String {
    var bytes: [UInt8] = []
    for id in ids where !specialIds.contains(id) {
      guard let token = idToToken[id] else { continue }
      for ch in token {
        if let b = Self.charToByte[ch] {
          bytes.append(b)
        }
      }
    }
    return String(decoding: bytes, as: UTF8.self)
  }

  /// Standard BPE merge loop over a byte-level-mapped word.
  private func bpe(_ word: String) -> [String] {
    var parts = word.map { String($0) }
    guard parts.count > 1 else { return parts }

    while true {
      var bestRank = Int.max
      var bestIndex = -1
      for i in 0..<(parts.count - 1) {
        if let rank = mergeRanks["\(parts[i]) \(parts[i + 1])"], rank < bestRank {
          bestRank = rank
          bestIndex = i
        }
      }
      guard bestIndex >= 0 else { break }
      parts[bestIndex] = parts[bestIndex] + parts[bestIndex + 1]
      parts.remove(at: bestIndex + 1)
      if parts.count == 1 { break }
    }
    return parts
  }
}
