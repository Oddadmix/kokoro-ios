//
//  KokoroSwift
//
//  On-device inference for the LiquidAI LFM2.5-230M chat model — a HYBRID
//  architecture: each block is either a short-convolution operator or a GQA
//  attention operator (with per-head QK-norm), followed by a SwiGLU MLP.
//  Mirrors the verified MLX Python reference in kikiri-tts
//  scripts/convert_lfm2_to_mlx.py (matches transformers token-for-token).
//

import Foundation
import MLX

/// Generates chat/agentic replies with LFM2.5-230M on device. Load with the
/// converted fp16 weights (lfm2_230m_fp16.safetensors) and the tokenizer.json.
public final class LFM2Model {
  // MARK: - Model hyperparameters (LFM2.5-230M)

  private enum Config {
    static let nLayers = 14
    static let dModel = 1024
    static let nHeads = 16
    static let nKVHeads = 8
    static let headDim = 64
    static let normEps: Float = 1e-5
    static let ropeTheta: Float = 1e6
    static let maxContext = 8192
    /// Which block is attention vs short-conv, by layer index.
    static let isAttention: [Bool] = [
      false, false, true, false, true, false, true,
      false, true, false, true, false, true, false,
    ]
  }

  let tokenizer: LFM2Tokenizer
  private let weights: [String: MLXArray]

  public init(modelPath: URL, tokenizerPath: URL) throws {
    weights = try MLX.loadArrays(url: modelPath)
    tokenizer = try LFM2Tokenizer(tokenizerPath: tokenizerPath)
  }

  // MARK: - Cache

  /// Per-layer state carried across autoregressive steps: KV for attention
  /// layers, the last two `Bx` rows for conv layers.
  final class Cache {
    var kv: [Int: (k: MLXArray, v: MLXArray)] = [:]
    var conv: [Int: MLXArray] = [:]
    var offset = 0
  }

  // MARK: - Prompt building

  /// A single chat turn for the LFM2 ChatML template.
  public struct Turn {
    public let role: String    // "user", "assistant", "tool"
    public let content: String
    public init(role: String, content: String) {
      self.role = role
      self.content = content
    }
  }

  /// Builds the LFM2 ChatML prompt: `<|startoftext|>` then an optional system
  /// section (with any `tools` appended as `List of tools: [...]`), the turns,
  /// and the assistant generation prompt. Text segments are byte-level-BPE
  /// encoded as whole strings, exactly matching the model's chat template.
  public func buildPrompt(system: String?, tools: [String] = [],
                          turns: [Turn], addGenerationPrompt: Bool = true) -> [Int] {
    var ids: [Int] = [tokenizer.bosId]

    var systemContent = system ?? ""
    if !tools.isEmpty {
      if !systemContent.isEmpty { systemContent += "\n" }
      systemContent += "List of tools: [" + tools.joined(separator: ", ") + "]"
    }
    if !systemContent.isEmpty {
      ids.append(tokenizer.imStartId)
      ids += tokenizer.encode("system\n" + systemContent)
      ids.append(tokenizer.imEndId)
      ids += tokenizer.encode("\n")
    }
    for turn in turns {
      ids.append(tokenizer.imStartId)
      ids += tokenizer.encode(turn.role + "\n" + turn.content)
      ids.append(tokenizer.imEndId)
      ids += tokenizer.encode("\n")
    }
    if addGenerationPrompt {
      ids.append(tokenizer.imStartId)
      ids += tokenizer.encode("assistant\n")
    }
    return ids
  }

  // MARK: - Generation

  /// Greedy generation from `promptIds`, stopping at any id in `stop`
  /// (typically the EOS / im_end token). Returns the generated ids.
  func generate(promptIds: [Int], maxTokens: Int, stop: Set<Int>) -> [Int] {
    guard promptIds.count < Config.maxContext - maxTokens else { return [] }
    let cache = Cache()
    var logits = forward(tokenIds: promptIds, cache: cache)
    var output: [Int] = []
    for _ in 0..<maxTokens {
      let next = Int(MLX.argMax(logits).item(Int32.self))
      if stop.contains(next) { break }
      output.append(next)
      logits = forward(tokenIds: [next], cache: cache)
    }
    return output
  }

  /// One forward pass over `tokenIds`, extending `cache`; returns logits for
  /// the last position. Norms and attention run in fp32 (weights fp16).
  private func forward(tokenIds: [Int], cache: Cache) -> MLXArray {
    let T = tokenIds.count
    let offset = cache.offset
    let embed = weights["model.embed_tokens.weight"]!
    var x = MLX.take(embed, MLXArray(tokenIds.map { Int32($0) }), axis: 0)  // (T, D)

    let kvRepeat = Config.nHeads / Config.nKVHeads
    let scale = MLXArray(1.0 / sqrtf(Float(Config.headDim)))
    var causal: MLXArray?
    if T > 1 {
      causal = MLX.triu(MLX.full([T, T + offset], values: MLXArray(-Float.infinity)), k: offset + 1)
    }

    for layer in 0..<Config.nLayers {
      let p = "model.layers.\(layer)."
      let h = rmsNorm(x, weight: weights[p + "operator_norm.weight"]!)

      let y: MLXArray
      if Config.isAttention[layer] {
        y = attention(h, p: p, layer: layer, cache: cache, offset: offset,
                      T: T, kvRepeat: kvRepeat, scale: scale, causal: causal)
      } else {
        y = shortConv(h, p: p, layer: layer, cache: cache, T: T)
      }

      x = x + y
      x = x + swiGLU(rmsNorm(x, weight: weights[p + "ffn_norm.weight"]!), p: p)
    }

    cache.offset = offset + T
    x = rmsNorm(x, weight: weights["model.embedding_norm.weight"]!)
    let logits = MLX.matmul(x[T - 1], embed.transposed())  // tied embeddings
    MLX.eval(logits)
    return logits
  }

  // MARK: - Blocks

  private func attention(_ h: MLXArray, p: String, layer: Int, cache: Cache,
                         offset: Int, T: Int, kvRepeat: Int,
                         scale: MLXArray, causal: MLXArray?) -> MLXArray {
    let hd = Config.headDim
    var q = MLX.matmul(h, weights[p + "self_attn.q_proj.weight"]!.transposed())
      .reshaped([T, Config.nHeads, hd])
    var k = MLX.matmul(h, weights[p + "self_attn.k_proj.weight"]!.transposed())
      .reshaped([T, Config.nKVHeads, hd])
    var v = MLX.matmul(h, weights[p + "self_attn.v_proj.weight"]!.transposed())
      .reshaped([T, Config.nKVHeads, hd])

    // Per-head QK RMSNorm over head_dim, before RoPE.
    q = rmsNorm(q, weight: weights[p + "self_attn.q_layernorm.weight"]!)
    k = rmsNorm(k, weight: weights[p + "self_attn.k_layernorm.weight"]!)

    q = rope(q.transposed(1, 0, 2), offset: offset)
    k = rope(k.transposed(1, 0, 2), offset: offset)
    v = v.transposed(1, 0, 2)

    if let cached = cache.kv[layer] {
      k = MLX.concatenated([cached.k, k], axis: 1)
      v = MLX.concatenated([cached.v, v], axis: 1)
    }
    cache.kv[layer] = (k, v)

    let kr = MLX.repeated(k, count: kvRepeat, axis: 0)
    let vr = MLX.repeated(v, count: kvRepeat, axis: 0)
    var scores = MLX.matmul(q.asType(.float32), kr.transposed(0, 2, 1).asType(.float32)) * scale
    if let causal {
      scores = scores + causal
    }
    var attn = MLX.matmul(MLX.softmax(scores, axis: -1).asType(h.dtype), vr)
    attn = attn.transposed(1, 0, 2).reshaped([T, Config.nHeads * hd])
    return MLX.matmul(attn, weights[p + "self_attn.out_proj.weight"]!.transposed())
  }

  /// Double-gated short convolution: in_proj → (B, C, x) → Bx → depthwise
  /// causal conv (kernel 3) → *C → out_proj.
  private func shortConv(_ h: MLXArray, p: String, layer: Int, cache: Cache, T: Int) -> MLXArray {
    let d = Config.dModel
    let proj = MLX.matmul(h, weights[p + "conv.in_proj.weight"]!.transposed())  // (T, 3D)
    let b = proj[0..., 0..<d]
    let c = proj[0..., d..<(2 * d)]
    let xx = proj[0..., (2 * d)..<(3 * d)]
    let bx = b * xx  // (T, D)

    // conv weight (D,1,3) → three per-channel taps (oldest, mid, current)
    let cw = weights[p + "conv.conv.weight"]!.reshaped([d, 3]).transposed()  // (3, D)
    let w0 = cw[0], w1 = cw[1], w2 = cw[2]

    let convOut: MLXArray
    if let prev = cache.conv[layer] {  // decode: prev holds last 2 Bx rows
      convOut = (w0 * prev[0] + w1 * prev[1] + w2 * bx[0]).reshaped([1, d])
      cache.conv[layer] = MLX.concatenated([prev[1..<2], bx], axis: 0)
    } else {  // prefill: left-pad 2 zeros, causal kernel-3 conv over time
      let pad = MLX.concatenated([MLX.zeros([2, d], dtype: bx.dtype), bx], axis: 0)
      convOut = w0 * pad[0..<T] + w1 * pad[1..<(T + 1)] + w2 * pad[2..<(T + 2)]
      cache.conv[layer] = T >= 2
        ? bx[(T - 2)..<T]
        : MLX.concatenated([MLX.zeros([2 - T, d], dtype: bx.dtype), bx], axis: 0)
    }
    return MLX.matmul(c * convOut, weights[p + "conv.out_proj.weight"]!.transposed())
  }

  private func swiGLU(_ h: MLXArray, p: String) -> MLXArray {
    let gate = MLX.matmul(h, weights[p + "feed_forward.w1.weight"]!.transposed())
    let up = MLX.matmul(h, weights[p + "feed_forward.w3.weight"]!.transposed())
    let activated = gate * MLX.sigmoid(gate) * up  // SiLU(gate) * up
    return MLX.matmul(activated, weights[p + "feed_forward.w2.weight"]!.transposed())
  }

  private func rmsNorm(_ x: MLXArray, weight: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let norm = xf * MLX.rsqrt(MLX.mean(xf * xf, axes: [-1], keepDims: true) + Config.normEps)
    return (norm * weight.asType(.float32)).asType(x.dtype)
  }

  /// Half-split RoPE (HF rotate_half convention) on [heads, T, headDim].
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
