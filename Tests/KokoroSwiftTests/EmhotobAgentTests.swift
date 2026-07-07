import Foundation
import Testing
@testable import KokoroSwift

/// End-to-end smoke test of the Emhotob-50M tool-calling agent with the real
/// model + live exchange-rate API. Skipped when the model dir is absent.
@Suite struct EmhotobAgentTests {
  static let dir = URL(fileURLWithPath: NSString(
    string: "~/Documents/models/emhotob").expandingTildeInPath)

  @Test func parsesToolCall() {
    let calls = EmhotobAgent.parseToolCalls(
      #"<tool_call>{"name": "generate_password", "arguments": {"length": 14, "include_symbols": true}}</tool_call>"#)
    #expect(calls.count == 1)
    #expect(calls[0].name == "generate_password")
    #expect((calls[0].arguments["length"] as? NSNumber)?.intValue == 14)
  }

  @Test func toolCallingRoundTrip() async throws {
    let weights = Self.dir.appendingPathComponent("nawah_50m_fp16.safetensors")
    guard FileManager.default.fileExists(atPath: weights.path) else {
      print("emhotob model dir not found — skipping"); return
    }
    let model = try NawahLLM(
      modelPath: weights,
      tokenizerPath: Self.dir.appendingPathComponent("tokenizer.json"))
    let agent = EmhotobAgent(model: model, tools: [
      GeneratePasswordTool(), CalculateBMITool(), CalculateTipTool(), ExchangeRateTool(),
    ])
    var used: [String] = []
    agent.onToolUse = { name, args in used.append(name); print("TOOL: \(name)(\(args))") }

    for q in ["أنشئ لي كلمة مرور من 14 حرفًا مع رموز.",
              "احسب مؤشر كتلة جسمي، وزني 80 كيلوجرام وطولي 1.80 متر.",
              "عندي فاتورة 200 وأريد بقشيش 15 بالمئة."] {
      let reply = await agent.respond(to: q)
      print("Q: [\(q)]\n → [\(reply)]")
      #expect(!reply.isEmpty)
    }
    #expect(used.contains("generate_password"))
    #expect(used.contains("calculate_bmi"))
  }
}
