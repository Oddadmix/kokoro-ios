import Foundation
import Testing
@testable import KokoroSwift

/// End-to-end smoke test of the agentic loop with the real LFM2 model and live
/// tool APIs (needs network). Skipped when the model dir is absent.
@Suite struct LFM2AgentSmokeTests {
  static let modelDir = LFM2ModelTests.modelDir

  @Test func currencyToolRoundTrip() async throws {
    guard FileManager.default.fileExists(
      atPath: Self.modelDir.appendingPathComponent("lfm2_230m_fp16.safetensors").path) else {
      print("lfm2 model dir not found — skipping")
      return
    }
    let model = try LFM2Model(
      modelPath: Self.modelDir.appendingPathComponent("lfm2_230m_fp16.safetensors"),
      tokenizerPath: Self.modelDir.appendingPathComponent("tokenizer.json"))
    let agent = LFM2Agent(
      model: model,
      tools: [WeatherTool(), CurrencyTool(), WebSearchTool()],
      system: "You are a helpful assistant with access to tools. "
        + "Use get_weather for weather, convert_currency for money, web_search for facts.")

    var used: [String] = []
    agent.onToolUse = { name, args in
      used.append(name)
      print("TOOL: \(name)(\(args))")
    }

    let reply = await agent.respond(to: "Convert 100 US dollars to euros.")
    print("REPLY: \(reply)")
    #expect(!reply.isEmpty)
    #expect(used.contains("convert_currency"))
  }

  /// Reproduces the app's exact path: Arabic system prompt + Arabic query.
  @Test func arabicCurrencyRoundTrip() async throws {
    guard FileManager.default.fileExists(
      atPath: Self.modelDir.appendingPathComponent("lfm2_230m_fp16.safetensors").path) else {
      print("lfm2 model dir not found — skipping"); return
    }
    let model = try LFM2Model(
      modelPath: Self.modelDir.appendingPathComponent("lfm2_230m_fp16.safetensors"),
      tokenizerPath: Self.modelDir.appendingPathComponent("tokenizer.json"))
    let system = "أنت مساعد صوتي ذكي لديك أدوات: get_weather لأسئلة الطقس، "
      + "convert_currency لتحويل العملات، web_search للبحث عن الحقائق والمعلومات. "
      + "استخدم الأداة المناسبة عند الحاجة. أجب دائماً باللغة العربية الفصحى فقط بإيجاز، "
      + "حتى لو كانت نتائج الأدوات باللغة الإنجليزية."
    let agent = LFM2Agent(model: model,
                          tools: [WeatherTool(), CurrencyTool(), WebSearchTool()],
                          system: system)
    agent.onToolUse = { name, args in print("TOOL: \(name)(\(args))") }
    let reply = await agent.respond(to: "حول 100 دولار أمريكي إلى يورو")
    print("AR REPLY: [\(reply)]")
    #expect(!reply.isEmpty)
  }
}
