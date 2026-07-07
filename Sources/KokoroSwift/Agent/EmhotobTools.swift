//
//  KokoroSwift
//
//  The Emhotob-50M built-in tools. Each `schemaJSON` is the exact schema the
//  model was trained on (from glaive-function-calling-ar), so the tool-system
//  prompt is byte-identical to training.
//

import Foundation

// Args arrive parsed from the model's JSON (NSNumber/Bool/String).
private func intArg(_ v: Any?) -> Int? {
  if let n = v as? Int { return n }
  if let n = v as? NSNumber { return n.intValue }
  if let s = v as? String { return Int(s) }
  return nil
}
private func doubleArg(_ v: Any?) -> Double? {
  if let n = v as? NSNumber { return n.doubleValue }
  if let d = v as? Double { return d }
  if let s = v as? String { return Double(s) }
  return nil
}
private func round2(_ x: Double) -> Double { (x * 100).rounded() / 100 }

// MARK: - generate_password

public struct GeneratePasswordTool: EmhotobTool {
  public init() {}
  public let name = "generate_password"
  public let keywords = ["كلمة مرور", "كلمة السر", "كلمة سر", "باسورد", "باس ورد", "password", "passcode"]
  public let schemaJSON = #"{"name": "generate_password", "description": "إنشاء كلمة مرور عشوائية.", "parameters": {"type": "object", "properties": {"length": {"type": "integer", "description": "طول كلمة المرور"}, "include_symbols": {"type": "boolean", "description": "هل يجب تضمين رموز في كلمة المرور؟"}}, "required": ["length"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    let length = max(4, min(intArg(a["length"]) ?? 12, 64))
    let symbols = (a["include_symbols"] as? Bool) ?? (a["include_symbols"] as? NSNumber)?.boolValue ?? false
    var alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if symbols { alphabet += "!@#$%^&*" }
    let chars = Array(alphabet)
    let pwd = String((0..<length).map { _ in chars.randomElement()! })
    return ["password": pwd]
  }
}

// MARK: - calculate_bmi

public struct CalculateBMITool: EmhotobTool {
  public init() {}
  public let name = "calculate_bmi"
  public let keywords = ["مؤشر كتلة", "كتلة الجسم", "كتلة جسم", "bmi", "وزني", "الوزن المثالي"]
  public let schemaJSON = #"{"name": "calculate_bmi", "description": "احسب مؤشر كتلة الجسم (BMI).", "parameters": {"type": "object", "properties": {"weight": {"type": "number", "description": "وزن الشخص بالكيلوجرام."}, "height": {"type": "number", "description": "ارتفاع الشخص بالمتر"}}, "required": ["weight", "height"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let w = doubleArg(a["weight"]), let h = doubleArg(a["height"]), h > 0 else {
      return ["error": "الوزن والطول مطلوبان"]
    }
    let bmi = (w / (h * h) * 10).rounded() / 10
    let category = bmi < 18.5 ? "نقص وزن" : bmi < 25 ? "طبيعي" : bmi < 30 ? "زيادة وزن" : "سمنة"
    return ["bmi": bmi, "category": category]
  }
}

// MARK: - calculate_tip

public struct CalculateTipTool: EmhotobTool {
  public init() {}
  public let name = "calculate_tip"
  public let keywords = ["بقشيش", "بخشيش", "إكرامية", "اكرامية", "فاتورة", "tip", "حساب النسبة"]
  public let schemaJSON = #"{"name": "calculate_tip", "description": "احسب مبلغ النصيحة على فاتورة.", "parameters": {"type": "object", "properties": {"bill_amount": {"type": "number", "description": "إجمالي المبلغ المستحق"}, "tip_percentage": {"type": "number", "description": "نسبة الرِقَم"}}, "required": ["bill_amount", "tip_percentage"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let bill = doubleArg(a["bill_amount"]), let pct = doubleArg(a["tip_percentage"]) else {
      return ["error": "المبلغ والنسبة مطلوبان"]
    }
    let tip = round2(bill * pct / 100)
    return ["tip_amount": tip, "total": round2(bill + tip)]
  }
}

// MARK: - get_weather (live via Open-Meteo)

public struct EmhotobWeatherTool: EmhotobTool {
  public init() {}
  public let name = "get_weather"
  public let keywords = ["الطقس", "طقس", "الجو", "الحرارة", "درجة الحرارة", "weather", "أمطار", "مطر"]
  public let schemaJSON = #"{"name": "get_weather", "description": "احصل على حالة الطقس الحالية في موقع معين.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "المدينة، مثال: القاهرة"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "وحدة قياس درجة الحرارة"}}, "required": ["location"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let location = (a["location"] as? String)?.trimmingCharacters(in: .whitespaces),
          !location.isEmpty else { return ["error": "الموقع مطلوب"] }
    guard let (lat, lon, place) = await geocode(location) else {
      return ["error": "تعذر إيجاد الموقع '\(location)'"]
    }
    let urlStr = "https://api.open-meteo.com/v1/forecast?latitude=\(lat)&longitude=\(lon)&current=temperature_2m,weather_code"
    if let url = URL(string: urlStr),
       let (data, _) = try? await URLSession.shared.data(from: url),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let current = json["current"] as? [String: Any],
       let temp = doubleArg(current["temperature_2m"]) {
      let code = intArg(current["weather_code"]) ?? -1
      return ["location": place, "temperature": temp, "unit": "celsius",
              "condition": Self.describe(code)]
    }
    return ["error": "تعذر جلب الطقس لـ \(place)"]
  }

  /// Geocode with an Arabic definite-article retry ("قاهرة" → "القاهرة").
  private func geocode(_ name: String) async -> (Double, Double, String)? {
    if let hit = await lookup(name) { return hit }
    if name.range(of: #"\p{Arabic}"#, options: .regularExpression) != nil, !name.hasPrefix("ال") {
      return await lookup("ال" + name)
    }
    return nil
  }

  private func lookup(_ name: String) async -> (Double, Double, String)? {
    let enc = name.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? name
    guard let url = URL(string: "https://geocoding-api.open-meteo.com/v1/search?name=\(enc)&count=1"),
          let (data, _) = try? await URLSession.shared.data(from: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let results = json["results"] as? [[String: Any]], let first = results.first,
          let lat = doubleArg(first["latitude"]), let lon = doubleArg(first["longitude"])
    else { return nil }
    let place = (first["name"] as? String) ?? name
    return (lat, lon, place)
  }

  /// WMO weather-code → short Arabic description.
  static func describe(_ code: Int) -> String {
    switch code {
    case 0: return "صافٍ"
    case 1, 2, 3: return "غائم جزئيًا"
    case 45, 48: return "ضبابي"
    case 51, 53, 55, 56, 57: return "رذاذ"
    case 61, 63, 65, 66, 67, 80, 81, 82: return "ممطر"
    case 71, 73, 75, 77, 85, 86: return "ثلوج"
    case 95, 96, 99: return "عاصفة رعدية"
    default: return "غير معروف"
    }
  }
}

// MARK: - get_exchange_rate (live via Frankfurter)

public struct ExchangeRateTool: EmhotobTool {
  public init() {}
  public let name = "get_exchange_rate"
  public let keywords = ["سعر الصرف", "صرف", "عملة", "عمله", "دولار", "يورو", "جنيه", "ريال", "exchange", "currency"]
  public let schemaJSON = #"{"name": "get_exchange_rate", "description": "احصل على سعر الصرف بين عملتين.", "parameters": {"type": "object", "properties": {"base_currency": {"type": "string", "description": "العملة التي سيتم تحويلها من"}, "target_currency": {"type": "string", "description": "العملة التي سيتم تحويلها إليها."}}, "required": ["base_currency", "target_currency"]}}"#

  private static let iso: [String: String] = [
    "دولار": "USD", "يورو": "EUR", "جنيه": "EGP", "ريال": "SAR",
    "درهم": "AED", "دينار": "KWD", "ليرة": "TRY", "روبية": "INR",
  ]

  private func normalize(_ raw: Any?) -> String {
    guard let s = (raw as? String)?.trimmingCharacters(in: .whitespaces), !s.isEmpty else { return "" }
    if s.count == 3, s.allSatisfy({ $0.isLetter }) { return s.uppercased() }
    for (ar, code) in Self.iso where s.contains(ar) { return code }
    return s.uppercased()
  }

  public func run(_ a: [String: Any]) async -> [String: Any] {
    let base = normalize(a["base_currency"]), target = normalize(a["target_currency"])
    guard !base.isEmpty, !target.isEmpty else { return ["error": "العملتان مطلوبتان"] }
    let urlStr = "https://api.frankfurter.app/latest?from=\(base)&to=\(target)"
    if let url = URL(string: urlStr),
       let (data, _) = try? await URLSession.shared.data(from: url),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let rates = json["rates"] as? [String: Any],
       let rate = doubleArg(rates[target]) {
      return ["base_currency": base, "target_currency": target, "exchange_rate": rate]
    }
    return ["error": "تعذر جلب سعر الصرف بين \(base) و\(target)"]
  }
}
