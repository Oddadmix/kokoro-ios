//
//  KokoroSwift
//
//  On-device inference for the Nawah-50M Arabic chat LLM (LLaMA
//  architecture: GQA attention, RoPE, RMSNorm, SiLU MLP, tied embeddings).
//  Mirrors the MLX Python reference in kikiri-tts
//  scripts/convert_nawah_to_mlx.py, which is verified token-for-token
//  against Hugging Face transformers.
//

import Foundation
import MLX

/// Generates conversational Arabic replies with the Nawah-50M chat model.
/// Load with the converted fp16 weights (nawah_50m_fp16.safetensors) and the
/// model's tokenizer.json.
public final class NawahLLM {
  // MARK: - Model hyperparameters (Nawah-50M)

  private enum Config {
    static let nLayers = 12
    static let nHeads = 8
    static let nKVHeads = 4
    static let headDim = 64
    static let dModel = 512
    static let rmsEps: Float = 1e-6
    static let ropeTheta: Float = 10000
    static let maxContext = 2048
  }

  /// System prompt the model was trained with.
  public static let defaultSystemPrompt =
    "أنت مساعد ذكي يجيب باللغة العربية الفصحى بدقة ووضوح."

  private let weights: [String: MLXArray]
  let tokenizer: NawahTokenizer

  public init(modelPath: URL, tokenizerPath: URL) throws {
    weights = try MLX.loadArrays(url: modelPath)
    tokenizer = try NawahTokenizer(tokenizerPath: tokenizerPath)
  }

  // MARK: - Public API

  /// Generates a reply for a single user turn using the ChatML format the
  /// model was trained with. Greedy decoding, stops at <|im_end|>.
  public func reply(to userText: String,
                    system: String = NawahLLM.defaultSystemPrompt,
                    maxTokens: Int = 100) -> String {
    var promptIds: [Int] = []
    promptIds.append(NawahTokenizer.imStartId)
    promptIds += tokenizer.encode("system\n" + system)
    promptIds.append(NawahTokenizer.imEndId)
    promptIds += tokenizer.encode("\n")
    promptIds.append(NawahTokenizer.imStartId)
    promptIds += tokenizer.encode("user\n" + userText)
    promptIds.append(NawahTokenizer.imEndId)
    promptIds += tokenizer.encode("\n")
    promptIds.append(NawahTokenizer.imStartId)
    promptIds += tokenizer.encode("assistant\n")

    let outputIds = generate(promptIds: promptIds, maxTokens: maxTokens)
    return tokenizer.decode(outputIds).trimmingCharacters(in: .whitespacesAndNewlines)
  }

  // MARK: - Generation

  private typealias KVCache = [(k: MLXArray, v: MLXArray)]

  /// Greedy generation. With `repetitionPenalty` > 1 (default 1 = off, which
  /// preserves the verified greedy path) the logits of already-generated tokens
  /// are down-weighted, and a repeating 6-gram cycle terminates generation —
  /// both curb the small model's tendency to loop.
  func generate(promptIds: [Int], maxTokens: Int, repetitionPenalty: Float = 1.0) -> [Int] {
    guard promptIds.count < Config.maxContext - maxTokens else { return [] }

    var (logits, cache) = forward(tokenIds: promptIds, cache: nil)
    var output: [Int] = []
    var seen = Set<Int>()
    for _ in 0..<maxTokens {
      let next: Int
      if repetitionPenalty > 1.0, !seen.isEmpty {
        var scores = logits.asArray(Float.self)
        for t in seen { scores[t] = scores[t] > 0 ? scores[t] / repetitionPenalty : scores[t] * repetitionPenalty }
        var best = 0
        for i in 1..<scores.count where scores[i] > scores[best] { best = i }
        next = best
      } else {
        next = Int(MLX.argMax(logits).item(Int32.self))
      }
      if next == NawahTokenizer.imEndId { break }
      output.append(next)
      seen.insert(next)
      // Stop if the tail repeats a 6-gram cycle (a runaway loop).
      if output.count >= 12,
         Array(output.suffix(6)) == Array(output[(output.count - 12)..<(output.count - 6)]) {
        output.removeLast(6)
        break
      }
      (logits, cache) = forward(tokenIds: [next], cache: cache)
    }
    return output
  }

  /// One forward pass over `tokenIds`, extending `cache`. Returns logits for
  /// the last position and the updated KV cache. Attention scores and norms
  /// are computed in fp32 (weights are fp16), matching the Python reference.
  private func forward(tokenIds: [Int], cache: KVCache?) -> (MLXArray, KVCache) {
    let seqLen = tokenIds.count
    let offset = cache?[0].k.dim(1) ?? 0
    let embeddings = weights["model.embed_tokens.weight"]!

    var x = MLX.take(embeddings, MLXArray(tokenIds.map { Int32($0) }), axis: 0)
    var newCache: KVCache = []

    // Causal mask only needed when processing more than one position
    var causal: MLXArray?
    if seqLen > 1 {
      causal = MLX.triu(MLX.full([seqLen, seqLen + offset], values: MLXArray(-Float.infinity)),
                        k: offset + 1)
    }

    let kvRepeat = Config.nHeads / Config.nKVHeads
    let scale = MLXArray(1.0 / sqrtf(Float(Config.headDim)))

    for layer in 0..<Config.nLayers {
      let p = "model.layers.\(layer)."

      var h = rmsNorm(x, weight: weights[p + "input_layernorm.weight"]!)
      var q = MLX.matmul(h, weights[p + "self_attn.q_proj.weight"]!.transposed())
      var k = MLX.matmul(h, weights[p + "self_attn.k_proj.weight"]!.transposed())
      var v = MLX.matmul(h, weights[p + "self_attn.v_proj.weight"]!.transposed())

      q = q.reshaped([seqLen, Config.nHeads, Config.headDim]).transposed(1, 0, 2)
      k = k.reshaped([seqLen, Config.nKVHeads, Config.headDim]).transposed(1, 0, 2)
      v = v.reshaped([seqLen, Config.nKVHeads, Config.headDim]).transposed(1, 0, 2)

      q = rope(q, offset: offset)
      k = rope(k, offset: offset)

      if let cache {
        k = MLX.concatenated([cache[layer].k, k], axis: 1)
        v = MLX.concatenated([cache[layer].v, v], axis: 1)
      }
      newCache.append((k: k, v: v))

      // GQA: repeat KV heads to match query heads
      let kr = MLX.repeated(k, count: kvRepeat, axis: 0)
      let vr = MLX.repeated(v, count: kvRepeat, axis: 0)

      var scores = MLX.matmul(q.asType(.float32), kr.transposed(0, 2, 1).asType(.float32)) * scale
      if let causal {
        scores = scores + causal
      }
      var attn = MLX.matmul(MLX.softmax(scores, axis: -1).asType(x.dtype), vr)
      attn = attn.transposed(1, 0, 2).reshaped([seqLen, Config.dModel])
      x = x + MLX.matmul(attn, weights[p + "self_attn.o_proj.weight"]!.transposed())

      h = rmsNorm(x, weight: weights[p + "post_attention_layernorm.weight"]!)
      let gate = MLX.matmul(h, weights[p + "mlp.gate_proj.weight"]!.transposed())
      let up = MLX.matmul(h, weights[p + "mlp.up_proj.weight"]!.transposed())
      let activated = gate * MLX.sigmoid(gate) * up  // SiLU(gate) * up
      x = x + MLX.matmul(activated, weights[p + "mlp.down_proj.weight"]!.transposed())
    }

    x = rmsNorm(x, weight: weights["model.norm.weight"]!)
    // Tied embeddings serve as the LM head
    let logits = MLX.matmul(x[seqLen - 1], embeddings.transposed())
    MLX.eval(logits)
    return (logits, newCache)
  }

  private func rmsNorm(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let norm = xf * MLX.rsqrt(MLX.mean(xf * xf, axes: [-1], keepDims: true) + Config.rmsEps)
    return (norm * weight.asType(.float32)).asType(x.dtype)
  }

  /// Standard LLaMA half-split RoPE applied to [heads, T, headDim].
  private func rope(_ x: MLXArray, offset: Int) -> MLXArray {
    let T = x.dim(1)
    let d2 = Config.headDim / 2
    let invFreq = 1.0 / MLX.pow(MLXArray(Config.ropeTheta),
                                MLXArray(0..<d2).asType(.float32) / Float(d2))
    let positions = MLXArray(offset..<(offset + T)).asType(.float32)
    let angles = positions.reshaped([T, 1]) * invFreq.reshaped([1, d2])
    let cos = MLX.cos(angles)
    let sin = MLX.sin(angles)
    let x1 = x[.ellipsis, 0..<d2].asType(.float32)
    let x2 = x[.ellipsis, d2..<Config.headDim].asType(.float32)
    let rotated = MLX.concatenated([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1)
    return rotated.asType(x.dtype)
  }
}
