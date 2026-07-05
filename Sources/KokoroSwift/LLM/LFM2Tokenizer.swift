//
//  KokoroSwift
//
//  Byte-level BPE tokenizer (GPT-2 style) loaded from a Hugging Face
//  tokenizer.json, for the LiquidAI LFM2 model. Uses LFM2's pre-tokenizer
//  split regex and resolves special-token ids by name from added_tokens.
//

import Foundation

/// Byte-level BPE tokenizer for the LFM2 chat model. Special tokens are not
/// split from text — build prompts by inserting their ids around encoded
/// segments (see LFM2Model.buildChatPrompt).
final class LFM2Tokenizer {
  enum TokenizerError: Error {
    case malformedTokenizerJSON
  }

  private let vocab: [String: Int]
  private let idToToken: [Int: String]
  private let mergeRanks: [String: Int]
  private let specialIds: Set<Int>

  // Named special tokens (resolved from the vocab)
  let bosId: Int          // <|startoftext|>
  let imStartId: Int      // <|im_start|>
  let imEndId: Int        // <|im_end|>  (also EOS)
  let toolCallStartId: Int
  let toolCallEndId: Int
  let toolResponseStartId: Int
  let toolResponseEndId: Int

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

  /// LFM2 pre-tokenizer split pattern (ByteLevel(use_regex:false) preceded by
  /// this Split regex).
  private static let splitRegex = try! NSRegularExpression(
    pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
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
    idToToken = Dictionary(fullVocab.map { ($1, $0) }, uniquingKeysWith: { a, _ in a })
    specialIds = special

    var ranks: [String: Int] = [:]
    for (rank, merge) in merges.enumerated() {
      if let s = merge as? String {
        ranks[s] = rank
      } else if let pair = merge as? [String], pair.count == 2 {
        ranks["\(pair[0]) \(pair[1])"] = rank
      }
    }
    mergeRanks = ranks

    func id(_ name: String) throws -> Int {
      guard let v = fullVocab[name] else { throw TokenizerError.malformedTokenizerJSON }
      return v
    }
    bosId = try id("<|startoftext|>")
    imStartId = try id("<|im_start|>")
    imEndId = try id("<|im_end|>")
    toolCallStartId = try id("<|tool_call_start|>")
    toolCallEndId = try id("<|tool_call_end|>")
    toolResponseStartId = try id("<|tool_response_start|>")
    toolResponseEndId = try id("<|tool_response_end|>")
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
