//
//  KokoroSwift
//
//  Tool-calling primitives for the LFM2 agent: the tool interface, and a
//  parser for the model's tool-call syntax
//  `[name(arg='value', arg2=123), ...]` emitted between <|tool_call_start|>
//  and <|tool_call_end|>.
//

import Foundation

/// A tool the LFM2 agent can call. `schemaJSON` is injected verbatim into the
/// system prompt's `List of tools: [...]`; `run` executes a parsed call.
public protocol AgentTool: Sendable {
  /// Function name the model uses to call this tool.
  var name: String { get }
  /// OpenAI-style JSON function schema (object with name/description/parameters).
  var schemaJSON: String { get }
  /// Executes the tool with parsed string arguments; returns a concise result.
  func run(arguments: [String: String]) async -> String
}

/// One parsed tool call.
public struct ToolCall: Equatable {
  public let name: String
  public let arguments: [String: String]
}

public enum ToolCallParser {
  /// Parses the inner text between the tool-call markers, e.g.
  /// `[get_weather(location='Cairo'), convert(amount=100, from='USD')]`.
  public static func parse(_ raw: String) -> [ToolCall] {
    var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if s.hasPrefix("[") { s.removeFirst() }
    if s.hasSuffix("]") { s.removeLast() }

    var calls: [ToolCall] = []
    let chars = Array(s)
    var i = 0
    let n = chars.count

    while i < n {
      // Skip separators / whitespace between calls
      while i < n, chars[i] == "," || chars[i] == " " || chars[i] == "\n" { i += 1 }
      // Read the function name up to '('
      var name = ""
      while i < n, chars[i] != "(" { name.append(chars[i]); i += 1 }
      name = name.trimmingCharacters(in: .whitespaces)
      guard i < n, chars[i] == "(" else { break }
      i += 1  // consume '('

      // Read the argument list up to the matching ')', tracking nesting/quotes
      var argText = ""
      var depth = 0
      var quote: Character? = nil
      while i < n {
        let c = chars[i]
        if let q = quote {
          argText.append(c)
          if c == q { quote = nil }
        } else if c == "'" || c == "\"" {
          quote = c
          argText.append(c)
        } else if c == "(" || c == "[" || c == "{" {
          depth += 1
          argText.append(c)
        } else if c == ")" || c == "]" || c == "}" {
          if depth == 0, c == ")" { break }
          depth -= 1
          argText.append(c)
        } else {
          argText.append(c)
        }
        i += 1
      }
      if i < n, chars[i] == ")" { i += 1 }  // consume ')'

      if !name.isEmpty {
        calls.append(ToolCall(name: name, arguments: parseArgs(argText)))
      }
    }
    return calls
  }

  /// Splits `key=value, key2='v2'` into a dictionary, unquoting string values.
  private static func parseArgs(_ text: String) -> [String: String] {
    var result: [String: String] = [:]
    for piece in splitTopLevel(text) {
      guard let eq = piece.firstIndex(of: "=") else { continue }
      let key = String(piece[..<eq]).trimmingCharacters(in: .whitespaces)
      var value = String(piece[piece.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
      if value.count >= 2,
         (value.hasPrefix("'") && value.hasSuffix("'")) ||
         (value.hasPrefix("\"") && value.hasSuffix("\"")) {
        value = String(value.dropFirst().dropLast())
      }
      if !key.isEmpty { result[key] = value }
    }
    return result
  }

  /// Splits on top-level commas (ignoring those inside quotes/brackets).
  private static func splitTopLevel(_ text: String) -> [String] {
    var pieces: [String] = []
    var current = ""
    var depth = 0
    var quote: Character? = nil
    for c in text {
      if let q = quote {
        current.append(c)
        if c == q { quote = nil }
      } else if c == "'" || c == "\"" {
        quote = c
        current.append(c)
      } else if c == "(" || c == "[" || c == "{" {
        depth += 1; current.append(c)
      } else if c == ")" || c == "]" || c == "}" {
        depth -= 1; current.append(c)
      } else if c == ",", depth == 0 {
        pieces.append(current); current = ""
      } else {
        current.append(c)
      }
    }
    if !current.trimmingCharacters(in: .whitespaces).isEmpty { pieces.append(current) }
    return pieces
  }
}
