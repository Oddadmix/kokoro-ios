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

// MARK: - split_bill

public struct SplitBillTool: EmhotobTool {
  public init() {}
  public let name = "split_bill"
  public let keywords = ["قسّم الفاتورة", "قسم الفاتورة", "نقسم", "التقسيم", "قسّم المبلغ", "split"]
  public let schemaJSON = #"{"name": "split_bill", "description": "قسّم الفاتورة بالتساوي على عدد من الأشخاص.", "parameters": {"type": "object", "properties": {"total_amount": {"type": "number", "description": "إجمالي الفاتورة"}, "number_of_people": {"type": "integer", "description": "عدد الأشخاص"}}, "required": ["total_amount", "number_of_people"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let total = doubleArg(a["total_amount"]), let people = intArg(a["number_of_people"]), people > 0 else {
      return ["error": "المبلغ وعدد الأشخاص مطلوبان"]
    }
    return ["per_person": round2(total / Double(people)), "people": people]
  }
}

// MARK: - convert_length

public struct ConvertLengthTool: EmhotobTool {
  public init() {}
  public let name = "convert_length"
  public let keywords = ["كيلومتر", "كيلو متر", "سنتيمتر", "سم إلى", "متر إلى", "ميل", "قدم", "الأطوال", "تحويل الطول"]
  public let schemaJSON = #"{"name": "convert_length", "description": "حوّل بين وحدات الطول (متر، كيلومتر، سنتيمتر، ميل، قدم).", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "القيمة"}, "from_unit": {"type": "string", "description": "الوحدة المصدر"}, "to_unit": {"type": "string", "description": "الوحدة الهدف"}}, "required": ["value", "from_unit", "to_unit"]}}"#

  // metres per unit — check longer/compound units before "متر"/"meter".
  private static let toM: [(keys: [String], m: Double)] = [
    (["كيلومتر", "كيلو متر", "kilometer", "kilometre", "km"], 1000),
    (["سنتيمتر", "centimeter", "centimetre", "cm"], 0.01),
    (["مليمتر", "ملم", "millimeter", "mm"], 0.001),
    (["ميل", "mile"], 1609.34),
    (["قدم", "foot", "feet", "ft"], 0.3048),
    (["بوصة", "انش", "inch"], 0.0254),
    (["متر", "meter", "metre", "m"], 1),
  ]
  private func metres(_ s: String) -> Double {
    let t = s.lowercased()
    for u in Self.toM where u.keys.contains(where: { t.contains($0) }) { return u.m }
    return 1
  }
  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let v = doubleArg(a["value"]) else { return ["error": "القيمة مطلوبة"] }
    let m = v * metres((a["from_unit"] as? String) ?? "m")
    return ["value": round2(m / metres((a["to_unit"] as? String) ?? "m")), "unit": (a["to_unit"] as? String) ?? ""]
  }
}

// MARK: - convert_weight

public struct ConvertWeightTool: EmhotobTool {
  public init() {}
  public let name = "convert_weight"
  public let keywords = ["جرام", "غرام", "رطل", "باوند", "أونصة", "طن", "الأوزان", "تحويل الوزن"]
  public let schemaJSON = #"{"name": "convert_weight", "description": "حوّل بين وحدات الوزن (كيلوغرام، جرام، رطل، أونصة).", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "القيمة"}, "from_unit": {"type": "string", "description": "الوحدة المصدر"}, "to_unit": {"type": "string", "description": "الوحدة الهدف"}}, "required": ["value", "from_unit", "to_unit"]}}"#

  private static let toKg: [(keys: [String], kg: Double)] = [
    (["كيلوغرام", "كيلوجرام", "كجم", "كيلو", "kilogram", "kg"], 1),
    (["ميليغرام", "ملغم", "milligram", "mg"], 1e-6),
    (["غرام", "جرام", "gram", "g"], 0.001),
    (["رطل", "باوند", "pound", "lb"], 0.453592),
    (["أونصة", "اونصة", "ounce", "oz"], 0.0283495),
    (["طن", "tonne", "ton"], 1000),
  ]
  private func kg(_ s: String) -> Double {
    let t = s.lowercased()
    for u in Self.toKg where u.keys.contains(where: { t.contains($0) }) { return u.kg }
    return 1
  }
  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let v = doubleArg(a["value"]) else { return ["error": "القيمة مطلوبة"] }
    let k = v * kg((a["from_unit"] as? String) ?? "kg")
    return ["value": round2(k / kg((a["to_unit"] as? String) ?? "kg")), "unit": (a["to_unit"] as? String) ?? ""]
  }
}

// MARK: - calculate_simple_interest

public struct SimpleInterestTool: EmhotobTool {
  public init() {}
  public let name = "calculate_simple_interest"
  public let keywords = ["فائدة", "الفائدة", "فوائد", "interest"]
  public let schemaJSON = #"{"name": "calculate_simple_interest", "description": "احسب الفائدة البسيطة على مبلغ.", "parameters": {"type": "object", "properties": {"principal": {"type": "number", "description": "المبلغ الأصلي"}, "rate": {"type": "number", "description": "نسبة الفائدة السنوية"}, "years": {"type": "number", "description": "عدد السنوات"}}, "required": ["principal", "rate", "years"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let p = doubleArg(a["principal"]), let r = doubleArg(a["rate"]), let y = doubleArg(a["years"]) else {
      return ["error": "المبلغ والنسبة والمدة مطلوبة"]
    }
    let interest = round2(p * r * y / 100)
    return ["interest": interest, "total": round2(p + interest)]
  }
}

// MARK: - count_words

public struct CountWordsTool: EmhotobTool {
  public init() {}
  public let name = "count_words"
  public let keywords = ["عدد الكلمات", "كم كلمة", "احسب الكلمات", "word count"]
  public let schemaJSON = #"{"name": "count_words", "description": "احسب عدد الكلمات في نص.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "النص"}}, "required": ["text"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    let text = (a["text"] as? String) ?? ""
    let words = text.split(whereSeparator: { $0.isWhitespace }).count
    return ["word_count": words]
  }
}

// MARK: - day_of_week

public struct DayOfWeekTool: EmhotobTool {
  public init() {}
  public let name = "day_of_week"
  public let keywords = ["أي يوم", "يوم الأسبوع", "اليوم يصادف", "day of week"]
  public let schemaJSON = #"{"name": "day_of_week", "description": "احصل على اسم يوم الأسبوع لتاريخ معين (YYYY-MM-DD).", "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "التاريخ بصيغة YYYY-MM-DD"}}, "required": ["date"]}}"#

  private static let days = ["الأحد", "الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت"]
  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let str = a["date"] as? String else { return ["error": "التاريخ مطلوب"] }
    let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"; df.timeZone = TimeZone(identifier: "UTC")
    guard let date = df.date(from: str) else { return ["error": "صيغة التاريخ يجب أن تكون YYYY-MM-DD"] }
    var cal = Calendar(identifier: .gregorian); cal.timeZone = TimeZone(identifier: "UTC")!
    return ["date": str, "day": Self.days[cal.component(.weekday, from: date) - 1]]
  }
}

// MARK: - flip_coin

public struct FlipCoinTool: EmhotobTool {
  public init() {}
  public let name = "flip_coin"
  public let keywords = ["اقلب عملة", "اقلب العملة", "صورة أم كتابة", "رمي العملة", "flip coin"]
  public let schemaJSON = #"{"name": "flip_coin", "description": "اقلب عملة معدنية.", "parameters": {"type": "object", "properties": {}, "required": []}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    return ["result": Bool.random() ? "صورة" : "كتابة"]
  }
}

// MARK: - calculate_speed

public struct CalculateSpeedTool: EmhotobTool {
  public init() {}
  public let name = "calculate_speed"
  public let keywords = ["السرعة", "احسب السرعة", "speed"]
  public let schemaJSON = #"{"name": "calculate_speed", "description": "احسب السرعة من المسافة والزمن.", "parameters": {"type": "object", "properties": {"distance": {"type": "number", "description": "المسافة"}, "time": {"type": "number", "description": "الزمن"}}, "required": ["distance", "time"]}}"#

  public func run(_ a: [String: Any]) async -> [String: Any] {
    guard let d = doubleArg(a["distance"]), let t = doubleArg(a["time"]), t > 0 else {
      return ["error": "المسافة والزمن مطلوبان"]
    }
    return ["speed": round2(d / t)]
  }
}
