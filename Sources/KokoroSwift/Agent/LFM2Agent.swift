//
//  KokoroSwift
//
//  Agentic loop over LFM2: inject tool schemas into the system prompt,
//  generate, detect and run tool calls, feed results back, and repeat until
//  the model produces a final answer.
//

import Foundation

/// Drives a tool-using conversation with LFM2. Works at the token level so the
/// assistant's tool-call turns (which contain special marker tokens) are fed
/// back verbatim, matching the model's chat template.
public final class LFM2Agent {
  private let model: LFM2Model
  private let tools: [AgentTool]
  private let system: String
  private let maxRounds: Int
  private let maxTokensPerTurn: Int
  private let fallback: String

  /// Notifies the UI that a tool is about to run (name, human-readable args).
  public var onToolUse: ((_ name: String, _ arguments: [String: String]) -> Void)?

  public init(model: LFM2Model, tools: [AgentTool], system: String,
              maxRounds: Int = 4, maxTokensPerTurn: Int = 256,
              fallback: String = "Sorry, I couldn't find an answer.") {
    self.model = model
    self.tools = tools
    self.system = system
    self.maxRounds = maxRounds
    self.maxTokensPerTurn = maxTokensPerTurn
    self.fallback = fallback
  }

  /// Runs the agentic loop for one user message and returns the final reply.
  /// `history` is prior turns (user/assistant) for multi-turn context.
  public func respond(to userText: String, history: [LFM2Model.Turn] = []) async -> String {
    let tok = model.tokenizer
    let schemas = tools.map { $0.schemaJSON }
    var ids = model.buildPrompt(
      system: system, tools: schemas,
      turns: history + [LFM2Model.Turn(role: "user", content: userText)],
      addGenerationPrompt: true)

    for _ in 0..<maxRounds {
      let gen = model.generate(promptIds: ids, maxTokens: maxTokensPerTurn, stop: [tok.imEndId])

      guard let (start, end) = toolCallSpan(in: gen, tok: tok) else {
        return tok.decode(gen).trimmingCharacters(in: .whitespacesAndNewlines)  // final answer
      }

      // Parse the calls from the inner text (marker tokens excluded).
      let calls = ToolCallParser.parse(tok.decode(Array(gen[(start + 1)..<end])))
      if calls.isEmpty { break }  // markers but unparseable → force a clean answer

      // Close the assistant tool-call turn verbatim, then append tool results.
      ids += gen + [tok.imEndId] + tok.encode("\n")
      for call in calls {
        onToolUse?(call.name, call.arguments)
        let result = await run(call)
        ids += [tok.imStartId] + tok.encode("tool\n" + result) + [tok.imEndId] + tok.encode("\n")
      }
      ids += [tok.imStartId] + tok.encode("assistant\n")
    }

    // Tool budget exhausted (the model kept calling tools). Force a text-only
    // answer by stopping the moment it tries another call — so we NEVER return
    // raw tool-call syntax to the user.
    let finalGen = model.generate(promptIds: ids, maxTokens: maxTokensPerTurn,
                                  stop: [tok.imEndId, tok.toolCallStartId])
    let text = tok.decode(finalGen).trimmingCharacters(in: .whitespacesAndNewlines)
    return text.isEmpty ? fallback : text
  }

  /// Finds the [start, end) token indices of the tool-call marker span, if any.
  private func toolCallSpan(in gen: [Int], tok: LFM2Tokenizer) -> (Int, Int)? {
    guard let start = gen.firstIndex(of: tok.toolCallStartId) else { return nil }
    let end = gen.firstIndex(of: tok.toolCallEndId) ?? gen.count
    guard end > start else { return nil }
    return (start, end)
  }

  private func run(_ call: ToolCall) async -> String {
    guard let tool = tools.first(where: { $0.name == call.name }) else {
      return "Error: unknown tool '\(call.name)'."
    }
    return await tool.run(arguments: call.arguments)
  }
}
