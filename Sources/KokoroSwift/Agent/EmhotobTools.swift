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

// MARK: - convert_temperature

public struct ConvertTemperatureTool: EmhotobTool {
  public init() {}
  public let name = "convert_temperature"
  public let keywords = ["فهرنهايت", "مئوية", "كلفن", "celsius", "fahrenheit", "kelvin"]
  public let schemaJSON = #"{"name": "convert_temperature", "description": "حوّل درجة الحرارة بين الوحدات (مئوية، فهرنهايت، كلفن).", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "القيمة"}, "from_unit": {"type": "string", "description": "الوحدة المصدر"}, "to_unit": {"type": "string", "description": "الوحدة الهدف"}}, "required": ["value", "from_unit", "to_unit"]}}"#

  private func unit(_ s: String) -> String {
    let t = s.lowercased()
    if t.contains("fahren") || t.contains("فهرن") || t == "f" { return "F" }
    if t.contains("kelvin") || t.contains("كلفن") || t == "k" { return "K" }
    return "C"
  }
  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let v = doubleArg(a["value"]) else { return ["error": "القيمة مطلوبة"] }
    let from = unit((a["from_unit"] as? String) ?? "C"), to = unit((a["to_unit"] as? String) ?? "C")
    let celsius = from == "F" ? (v - 32) * 5 / 9 : from == "K" ? v - 273.15 : v
    let out = to == "F" ? celsius * 9 / 5 + 32 : to == "K" ? celsius + 273.15 : celsius
    return ["value": round2(out), "unit": to]
  }
}

// MARK: - calculate_percentage

public struct CalculatePercentageTool: EmhotobTool {
  public init() {}
  public let name = "calculate_percentage"
  public let keywords = ["بالمئة", "بالمائة", "في المئة", "نسبة مئوية", "النسبة المئوية", "percentage"]
  public let schemaJSON = #"{"name": "calculate_percentage", "description": "احسب نسبة مئوية من رقم.", "parameters": {"type": "object", "properties": {"percentage": {"type": "number", "description": "النسبة المئوية"}, "number": {"type": "number", "description": "الرقم"}}, "required": ["percentage", "number"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let p = doubleArg(a["percentage"]), let n = doubleArg(a["number"]) else {
      return ["error": "النسبة والرقم مطلوبان"]
    }
    return ["result": round2(n * p / 100)]
  }
}

// MARK: - calculate_discount

public struct CalculateDiscountTool: EmhotobTool {
  public init() {}
  public let name = "calculate_discount"
  public let keywords = ["خصم", "الخصم", "تخفيض", "discount"]
  public let schemaJSON = #"{"name": "calculate_discount", "description": "احسب السعر النهائي بعد الخصم.", "parameters": {"type": "object", "properties": {"original_price": {"type": "number", "description": "السعر الأصلي"}, "discount_percentage": {"type": "number", "description": "نسبة الخصم"}}, "required": ["original_price", "discount_percentage"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let price = doubleArg(a["original_price"]), let pct = doubleArg(a["discount_percentage"]) else {
      return ["error": "السعر ونسبة الخصم مطلوبان"]
    }
    let saved = round2(price * pct / 100)
    return ["discount_amount": saved, "final_price": round2(price - saved)]
  }
}

// MARK: - calculate_age

public struct CalculateAgeTool: EmhotobTool {
  public init() {}
  public let name = "calculate_age"
  public let keywords = ["عمري", "كم عمري", "احسب العمر", "مواليد", "سنة ميلادي", "age"]
  public let schemaJSON = #"{"name": "calculate_age", "description": "احسب العمر من سنة الميلاد.", "parameters": {"type": "object", "properties": {"birth_year": {"type": "integer", "description": "سنة الميلاد"}}, "required": ["birth_year"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let year = intArg(a["birth_year"]) else { return ["error": "سنة الميلاد مطلوبة"] }
    let now = Calendar(identifier: .gregorian).component(.year, from: Date())
    return ["age": max(0, now - year)]
  }
}

// MARK: - random_number

public struct RandomNumberTool: EmhotobTool {
  public init() {}
  public let name = "random_number"
  public let keywords = ["رقم عشوائي", "عدد عشوائي", "اختر رقم", "random"]
  public let schemaJSON = #"{"name": "random_number", "description": "اختر رقمًا عشوائيًا ضمن نطاق.", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "أصغر قيمة"}, "max": {"type": "integer", "description": "أكبر قيمة"}}, "required": ["min", "max"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    let lo = intArg(a["min"]) ?? 1, hi = intArg(a["max"]) ?? 100
    return ["result": Int.random(in: min(lo, hi)...max(lo, hi))]
  }
}

// MARK: - calculate_zakat

public struct CalculateZakatTool: EmhotobTool {
  public init() {}
  public let name = "calculate_zakat"
  public let keywords = ["زكاة", "الزكاة", "zakat"]
  public let schemaJSON = #"{"name": "calculate_zakat", "description": "احسب مقدار الزكاة على مبلغ من المال (2.5%).", "parameters": {"type": "object", "properties": {"amount": {"type": "number", "description": "المبلغ"}}, "required": ["amount"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let amount = doubleArg(a["amount"]) else { return ["error": "المبلغ مطلوب"] }
    return ["amount": amount, "zakat": round2(amount * 0.025)]
  }
}

// MARK: - calculate_vat

public struct CalculateVATTool: EmhotobTool {
  public init() {}
  public let name = "calculate_vat"
  public let keywords = ["ضريبة", "القيمة المضافة", "vat", "الضريبة"]
  public let schemaJSON = #"{"name": "calculate_vat", "description": "احسب القيمة المضافة والمبلغ الإجمالي.", "parameters": {"type": "object", "properties": {"amount": {"type": "number", "description": "المبلغ قبل الضريبة"}, "tax_percentage": {"type": "number", "description": "نسبة الضريبة"}}, "required": ["amount", "tax_percentage"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let amount = doubleArg(a["amount"]) else { return ["error": "المبلغ مطلوب"] }
    let pct = doubleArg(a["tax_percentage"]) ?? 15
    let vat = round2(amount * pct / 100)
    return ["vat": vat, "total": round2(amount + vat)]
  }
}

// MARK: - days_until

public struct DaysUntilTool: EmhotobTool {
  public init() {}
  public let name = "days_until"
  public let keywords = ["كم يوم", "كم يومًا", "كم باقي", "متبقي", "days until"]
  public let schemaJSON = #"{"name": "days_until", "description": "احسب عدد الأيام المتبقية حتى تاريخ معين (YYYY-MM-DD).", "parameters": {"type": "object", "properties": {"target_date": {"type": "string", "description": "التاريخ الهدف بصيغة YYYY-MM-DD"}}, "required": ["target_date"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let str = a["target_date"] as? String else { return ["error": "التاريخ مطلوب"] }
    let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"; df.timeZone = TimeZone(identifier: "UTC")
    guard let target = df.date(from: str) else { return ["error": "صيغة التاريخ يجب أن تكون YYYY-MM-DD"] }
    let cal = Calendar(identifier: .gregorian)
    let days = cal.dateComponents([.day], from: cal.startOfDay(for: Date()), to: target).day ?? 0
    return ["target_date": str, "days_remaining": days]
  }
}

// MARK: - get_prayer_times (live via Aladhan)

public struct PrayerTimesTool: EmhotobTool {
  public init() {}
  public let name = "get_prayer_times"
  public let keywords = ["مواقيت الصلاة", "مواعيد الصلاة", "الصلاة", "أذان", "اذان", "prayer"]
  public let schemaJSON = #"{"name": "get_prayer_times", "description": "احصل على مواقيت الصلاة في مدينة.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "المدينة"}}, "required": ["city"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let city = (a["city"] as? String)?.trimmingCharacters(in: .whitespaces), !city.isEmpty else {
      return ["error": "المدينة مطلوبة"]
    }
    let enc = city.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? city
    guard let url = URL(string: "https://api.aladhan.com/v1/timingsByAddress?address=\(enc)&method=5"),
          let (data, _) = try? await URLSession.shared.data(from: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let d = json["data"] as? [String: Any], let t = d["timings"] as? [String: Any]
    else { return ["error": "تعذر جلب مواقيت الصلاة لـ \(city)"] }
    return ["city": city,
            "الفجر": t["Fajr"] ?? "", "الظهر": t["Dhuhr"] ?? "", "العصر": t["Asr"] ?? "",
            "المغرب": t["Maghrib"] ?? "", "العشاء": t["Isha"] ?? ""]
  }
}

// MARK: - convert_to_hijri (live via Aladhan)

public struct HijriDateTool: EmhotobTool {
  public init() {}
  public let name = "convert_to_hijri"
  public let keywords = ["التاريخ الهجري", "هجري", "التقويم الهجري", "hijri"]
  public let schemaJSON = #"{"name": "convert_to_hijri", "description": "حوّل تاريخًا ميلاديًا إلى التقويم الهجري (يُستخدم تاريخ اليوم إن لم يُحدَّد).", "parameters": {"type": "object", "properties": {"gregorian_date": {"type": "string", "description": "التاريخ الميلادي بصيغة DD-MM-YYYY"}}, "required": []}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    let df = DateFormatter(); df.dateFormat = "dd-MM-yyyy"; df.timeZone = TimeZone(identifier: "UTC")
    let date = (a["gregorian_date"] as? String) ?? df.string(from: Date())
    guard let url = URL(string: "https://api.aladhan.com/v1/gToH/\(date)"),
          let (data, _) = try? await URLSession.shared.data(from: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let d = json["data"] as? [String: Any], let h = d["hijri"] as? [String: Any]
    else { return ["error": "تعذر تحويل التاريخ"] }
    let month = (h["month"] as? [String: Any])?["ar"] as? String ?? ""
    return ["gregorian": date,
            "hijri": "\(h["day"] ?? "") \(month) \(h["year"] ?? "")هـ"]
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
