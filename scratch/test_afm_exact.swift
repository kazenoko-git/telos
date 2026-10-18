
import Foundation
import FoundationModels

@main
struct AFMExactBenchmark {
    static func main() async throws {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            print("SystemLanguageModel is not available!")
            return
        }

        print("=== Apple Foundation Model (SystemLanguageModel) Exact Benchmark ===")
        
        let testPrompts = [
            ("Warmup", "Write a 50-word summary of quantum mechanics.", 128),
            ("Code Generation (Python QuickSort & Binary Search)", "Write a complete, highly optimized Python module implementing QuickSort with dual-pivot partitioning and binary search with edge case handling, including extensive unit tests using unittest.", 512),
            ("Creative Reasoning (System Architecture)", "Design an ultra-low-latency distributed cache system in C++20. Explain cache coherence protocols, write-back vs write-through strategies, lock-free ring buffers, and NUMA-aware memory allocation in exhaustive detail.", 768),
            ("Long Form Analysis (Transformers)", "Provide a deep mathematical derivation of Multi-Head Self-Attention, Rotary Positional Embeddings (RoPE), and FlashAttention v2 memory tiling algorithms. Write down the equations and step-by-step matrix decompositions.", 1024)
        ]
        
        for (name, prompt, maxTokens) in testPrompts {
            let session = LanguageModelSession(model: model, instructions: "You are an expert computer scientist and engineer. Provide detailed, rigorous, and exhaustive responses without hesitation or unnecessary apologies.")
            let options = GenerationOptions(
                temperature: 0.0,
                maximumResponseTokens: maxTokens
            )
            
            var chunkCount = 0
            var firstTokenTime: CFAbsoluteTime? = nil
            var lastSnapshot: LanguageModelSession.ResponseStream<String>.Snapshot? = nil
            
            let startTime = CFAbsoluteTimeGetCurrent()
            
            for try await snapshot in session.streamResponse(to: prompt, options: options) {
                if firstTokenTime == nil {
                    firstTokenTime = CFAbsoluteTimeGetCurrent()
                }
                chunkCount += 1
                lastSnapshot = snapshot
            }
            
            let endTime = CFAbsoluteTimeGetCurrent()
            let totalTime = endTime - startTime
            let ttft = (firstTokenTime ?? endTime) - startTime
            let decodeTime = endTime - (firstTokenTime ?? endTime)
            
            let outputTokens = lastSnapshot?.usage.output.totalTokenCount ?? 0
            let inputTokens = lastSnapshot?.usage.input.totalTokenCount ?? 0
            let decodeTokPerSec = decodeTime > 0 ? Double(outputTokens) / decodeTime : 0.0
            let totalTokPerSec = totalTime > 0 ? Double(outputTokens) / totalTime : 0.0
            
            print("
--- [\(name)] ---")
            print(String(format: "Input Tokens:   %d", inputTokens))
            print(String(format: "Output Tokens:  %d", outputTokens))
            print(String(format: "Chunks / Steps: %d", chunkCount))
            print(String(format: "TTFT:           %.4f s", ttft))
            print(String(format: "Decode Time:    %.4f s", decodeTime))
            print(String(format: "Total Time:     %.4f s", totalTime))
            print(String(format: "Decode Speed:   %.2f tok/s", decodeTokPerSec))
            print(String(format: "Overall Speed:  %.2f tok/s", totalTokPerSec))
        }
    }
}
