import Testing
@testable import KokoroSwift

@Suite struct ToolCallParserTests {
  @Test func singleStringArg() {
    let calls = ToolCallParser.parse("[get_weather(location='Cairo')]")
    #expect(calls == [ToolCall(name: "get_weather", arguments: ["location": "Cairo"])])
  }

  @Test func multipleMixedArgs() {
    let calls = ToolCallParser.parse("[convert_currency(amount=100, from='USD', to='EUR')]")
    #expect(calls == [ToolCall(name: "convert_currency",
                               arguments: ["amount": "100", "from": "USD", "to": "EUR"])])
  }

  @Test func doubleQuotesAndSpaces() {
    let calls = ToolCallParser.parse(#"[web_search(query="latest AI news, 2026")]"#)
    #expect(calls == [ToolCall(name: "web_search",
                               arguments: ["query": "latest AI news, 2026"])])
  }

  @Test func multipleCalls() {
    let calls = ToolCallParser.parse("[get_weather(location='Cairo'), get_weather(location='Paris')]")
    #expect(calls.count == 2)
    #expect(calls[0].arguments["location"] == "Cairo")
    #expect(calls[1].arguments["location"] == "Paris")
  }

  @Test func noArgs() {
    let calls = ToolCallParser.parse("[list_tools()]")
    #expect(calls == [ToolCall(name: "list_tools", arguments: [:])])
  }
}
