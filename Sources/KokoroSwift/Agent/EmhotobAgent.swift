//
//  KokoroSwift
//
//  Agentic tool-calling loop for the Emhotob-50M Arabic model (a fine-tune of
//  the from-scratch 50M LLaMA, run via NawahLLM). The model emits calls as
//  <tool_call>{JSON}</tool_call> and reads results from <tool_response>{...}</tool_response>,
//  matching its glaive-function-calling training format.
//
//  A 50M model is trained ~one-tool-per-row, so it collapses when given many
//  tools at once. A lightweight Arabic keyword router therefore exposes only the
//  single relevant tool per turn (or none → plain chat).
//

import Foundation

/// A tool the Emhotob agent can call. `schemaJSON` is the exact training schema
/// (json.dumps with Arabic literal), injected into the tool-system prompt.
public protocol EmhotobTool: Sendable {
  var name: String { get }
  var schemaJSON: String { get }
  /// Words in the user's request that should route to this tool.
  var keywords: [String] { get }
  /// Executes the parsed call; returns a result object serialized to the
  /// <tool_response>.
  func run(_ arguments: [String: Any]) async -> [String: Any]
}

public final class EmhotobAgent {
  private let model: NawahLLM
  private let tools: [EmhotobTool]
  private let maxRounds: Int
  private let maxTokensPerTurn: Int

  /// Fires when a tool is about to run (name, parsed arguments).
  public var onToolUse: ((_ name: String, _ arguments: [String: Any]) -> Void)?

  // Training format constants (from the SFT format module).
  private static let defaultSystem = "أنت مساعد ذكي يجيب باللغة العربية الفصحى بدقة ووضوح."
  private static let toolSystemPrefix =
    "أنت مساعد ذكي قادر على استدعاء الأدوات. الأدوات المتاحة بصيغة JSON:\n"
  private static let toolSystemSuffix =
    "\nعند الحاجة لاستدعاء أداة، أرجِع الاستدعاء داخل <tool_call>...</tool_call> "
    + "على شكل JSON يحتوي على \"name\" و\"arguments\". ستعود نتيجة تنفيذ الأداة داخل "
    + "<tool_response>...</tool_response>، ثم أكمل ردك بالاعتماد عليها."

  public init(model: NawahLLM, tools: [EmhotobTool],
              maxRounds: Int = 3, maxTokensPerTurn: Int = 220) {
    self.model = model
    self.tools = tools
    self.maxRounds = maxRounds
    self.maxTokensPerTurn = maxTokensPerTurn
  }

  // MARK: - Public API

  public func respond(to userText: String) async -> String {
    if let tool = route(userText) {
      return await runToolLoop(userText, tool: tool)
    }
    return plainChat(userText)
  }

  /// Picks the single tool whose keywords appear in the request, or nil.
  private func route(_ text: String) -> EmhotobTool? {
    let hay = text.lowercased()
    return tools.first { $0.keywords.contains { hay.contains($0.lowercased()) } }
  }

  // MARK: - Tool loop

  private struct Turn { let role: String; let content: String }

  private func runToolLoop(_ userText: String, tool: EmhotobTool) async -> String {
    let toolSystem = Self.toolSystemPrefix + "[" + tool.schemaJSON + "]" + Self.toolSystemSuffix
    var messages = [Turn(role: "user", content: userText)]

    var lastText = ""
    for _ in 0..<maxRounds {
      let ids = buildPrompt(system: toolSystem, turns: messages)
      let gen = model.generate(promptIds: ids, maxTokens: maxTokensPerTurn)
      lastText = model.tokenizer.decode(gen).trimmingCharacters(in: .whitespacesAndNewlines)

      let calls = Self.parseToolCalls(lastText)
      if calls.isEmpty {
        return lastText  // model answered directly
      }

      // Feed the assistant's tool-call turn back verbatim, then the results.
      messages.append(Turn(role: "assistant", content: lastText))
      for call in calls where call.name == tool.name {
        onToolUse?(call.name, call.arguments)
        let result = await tool.run(call.arguments)
        let json = Self.jsonString(result)
        messages.append(Turn(role: "tool", content: "<tool_response>\(json)</tool_response>"))
      }
    }
    // Exhausted rounds without a plain answer — strip any tool-call markup.
    return stripToolCalls(lastText)
  }

  private func plainChat(_ userText: String) -> String {
    let ids = buildPrompt(system: Self.defaultSystem,
                          turns: [Turn(role: "user", content: userText)])
    let gen = model.generate(promptIds: ids, maxTokens: maxTokensPerTurn)
    return model.tokenizer.decode(gen).trimmingCharacters(in: .whitespacesAndNewlines)
  }

  // MARK: - Prompt building (ChatML, bos-prefixed, matching training)

  private func buildPrompt(system: String, turns: [Turn]) -> [Int] {
    let tok = model.tokenizer
    var ids: [Int] = [NawahTokenizer.bosId]
    ids.append(NawahTokenizer.imStartId)
    ids += tok.encode("system\n" + system)
    ids.append(NawahTokenizer.imEndId)
    ids += tok.encode("\n")
    for turn in turns {
      ids.append(NawahTokenizer.imStartId)
      ids += tok.encode(turn.role + "\n" + turn.content)
      ids.append(NawahTokenizer.imEndId)
      ids += tok.encode("\n")
    }
    ids.append(NawahTokenizer.imStartId)
    ids += tok.encode("assistant\n")
    return ids
  }

  // MARK: - Tool-call parsing

  struct ParsedCall { let name: String; let arguments: [String: Any] }

  /// Extracts <tool_call>{JSON}</tool_call> blocks and parses each.
  static func parseToolCalls(_ text: String) -> [ParsedCall] {
    var calls: [ParsedCall] = []
    var search = text.startIndex
    let open = "<tool_call>", close = "</tool_call>"
    while let o = text.range(of: open, range: search..<text.endIndex) {
      let innerStart = o.upperBound
      let innerEnd = text.range(of: close, range: innerStart..<text.endIndex)?.lowerBound ?? text.endIndex
      let inner = String(text[innerStart..<innerEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
      if let data = inner.data(using: .utf8),
         let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let name = obj["name"] as? String {
        calls.append(ParsedCall(name: name, arguments: obj["arguments"] as? [String: Any] ?? [:]))
      }
      search = innerEnd
    }
    return calls
  }

  private func stripToolCalls(_ text: String) -> String {
    var out = text
    while let o = out.range(of: "<tool_call>"),
          let c = out.range(of: "</tool_call>", range: o.upperBound..<out.endIndex) {
      out.removeSubrange(o.lowerBound..<c.upperBound)
    }
    return out.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// Serializes a tool result to compact JSON with Arabic left literal.
  static func jsonString(_ obj: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: obj,
                                                 options: [.withoutEscapingSlashes]),
          let s = String(data: data, encoding: .utf8) else { return "{}" }
    return s
  }
}
