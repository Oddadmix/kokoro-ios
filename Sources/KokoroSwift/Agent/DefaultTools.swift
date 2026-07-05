//
//  KokoroSwift
//
//  The three built-in agent tools, all using free no-API-key services:
//  weather (Open-Meteo), currency (Frankfurter), web search (DuckDuckGo).
//

import Foundation

/// Fetches and JSON-decodes a URL; returns nil on any failure.
private func getJSON(_ url: String) async -> Any? {
  guard let encoded = url.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
        let u = URL(string: encoded) else { return nil }
  guard let (data, _) = try? await URLSession.shared.data(from: u) else { return nil }
  return try? JSONSerialization.jsonObject(with: data)
}

// MARK: - Weather (Open-Meteo)

public struct WeatherTool: AgentTool {
  public init() {}
  public let name = "get_weather"
  public let schemaJSON = """
  {"name": "get_weather", "description": "Get the current weather for a city.", \
  "parameters": {"type": "object", "properties": {"location": {"type": "string", \
  "description": "City name, e.g. 'Cairo'"}}, "required": ["location"]}}
  """

  public func run(arguments: [String: String]) async -> String {
    guard let location = arguments["location"], !location.isEmpty else {
      return "Error: missing location."
    }
    // 1. Geocode the city name to coordinates.
    guard let (lat, lon, place) = await geocode(location) else {
      return "Could not find location '\(location)'."
    }

    // 2. Fetch current conditions.
    guard let wx = await getJSON(
      "https://api.open-meteo.com/v1/forecast?latitude=\(lat)&longitude=\(lon)&current=temperature_2m,wind_speed_10m,weather_code") as? [String: Any],
      let current = wx["current"] as? [String: Any],
      let temp = current["temperature_2m"] as? Double
    else { return "Could not fetch weather for \(place)." }

    let code = (current["weather_code"] as? Int) ?? -1
    let wind = current["wind_speed_10m"] as? Double ?? 0
    return "Weather in \(place): \(temp)°C, \(Self.describe(code)), wind \(wind) km/h."
  }

  /// Geocodes a city name → (lat, lon, "City, Country"). Retries with the
  /// Arabic definite article prepended, since the small model often drops it
  /// (e.g. "قاهرة" → "القاهرة"), which Open-Meteo requires.
  private func geocode(_ name: String) async -> (Double, Double, String)? {
    if let hit = await lookup(name) { return hit }
    let isArabic = name.range(of: #"\p{Arabic}"#, options: .regularExpression) != nil
    if isArabic && !name.hasPrefix("ال") {
      return await lookup("ال" + name)
    }
    return nil
  }

  private func lookup(_ name: String) async -> (Double, Double, String)? {
    guard let geo = await getJSON(
      "https://geocoding-api.open-meteo.com/v1/search?name=\(name)&count=1") as? [String: Any],
      let results = geo["results"] as? [[String: Any]], let first = results.first,
      let lat = first["latitude"] as? Double, let lon = first["longitude"] as? Double
    else { return nil }
    let place = [first["name"] as? String, first["country"] as? String]
      .compactMap { $0 }.joined(separator: ", ")
    return (lat, lon, place)
  }

  /// WMO weather-code → short description.
  static func describe(_ code: Int) -> String {
    switch code {
    case 0: return "clear sky"
    case 1, 2, 3: return "partly cloudy"
    case 45, 48: return "foggy"
    case 51, 53, 55, 56, 57: return "drizzle"
    case 61, 63, 65, 66, 67: return "rain"
    case 71, 73, 75, 77: return "snow"
    case 80, 81, 82: return "rain showers"
    case 85, 86: return "snow showers"
    case 95, 96, 99: return "thunderstorm"
    default: return "unknown conditions"
    }
  }
}

// MARK: - Currency (Frankfurter)

public struct CurrencyTool: AgentTool {
  public init() {}
  public let name = "convert_currency"
  public let schemaJSON = """
  {"name": "convert_currency", "description": "Convert an amount from one \
  currency to another using live exchange rates.", "parameters": {"type": "object", \
  "properties": {"amount": {"type": "number", "description": "Amount to convert"}, \
  "from": {"type": "string", "description": "ISO currency code, e.g. 'USD'"}, \
  "to": {"type": "string", "description": "ISO currency code, e.g. 'EUR'"}}, \
  "required": ["amount", "from", "to"]}}
  """

  public func run(arguments: [String: String]) async -> String {
    let amount = Double(arguments["amount"] ?? "1") ?? 1
    guard let from = arguments["from"]?.uppercased(),
          let to = arguments["to"]?.uppercased(), !from.isEmpty, !to.isEmpty else {
      return "Error: missing currency codes."
    }
    guard let json = await getJSON(
      "https://api.frankfurter.app/latest?amount=\(amount)&from=\(from)&to=\(to)") as? [String: Any],
      let rates = json["rates"] as? [String: Any], let converted = rates[to] as? Double
    else { return "Could not convert \(from) to \(to)." }
    return "\(amount) \(from) = \(converted) \(to)."
  }
}

// MARK: - Web search (DuckDuckGo Instant Answer)

public struct WebSearchTool: AgentTool {
  public init() {}
  public let name = "web_search"
  public let schemaJSON = """
  {"name": "web_search", "description": "Search the web for a topic and return \
  a short summary.", "parameters": {"type": "object", "properties": {"query": \
  {"type": "string", "description": "The search query"}}, "required": ["query"]}}
  """

  public func run(arguments: [String: String]) async -> String {
    guard let query = arguments["query"], !query.isEmpty else {
      return "Error: missing query."
    }
    guard let json = await getJSON(
      "https://api.duckduckgo.com/?q=\(query)&format=json&no_html=1&skip_disambig=1") as? [String: Any]
    else { return "Search failed for '\(query)'." }

    if let abstract = json["AbstractText"] as? String, !abstract.isEmpty {
      let source = json["AbstractSource"] as? String
      return source.map { "\(abstract) (source: \($0))" } ?? abstract
    }
    if let answer = json["Answer"] as? String, !answer.isEmpty {
      return answer
    }
    // Fall back to the first few related-topic snippets.
    if let related = json["RelatedTopics"] as? [[String: Any]] {
      let snippets = related.compactMap { $0["Text"] as? String }.prefix(3)
      if !snippets.isEmpty { return snippets.joined(separator: " | ") }
    }
    return "No instant answer found for '\(query)'."
  }
}
