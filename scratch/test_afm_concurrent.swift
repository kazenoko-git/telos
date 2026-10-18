import Foundation
import FoundationModels

@main
struct AFMConcurrencyBenchmark {
    static func main() async {
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return }

        print("=== Evaluating AFM Concurrency & Aggregate Throughput ===")

        let concurrencyLevels = [1, 2, 4]
        let prompt = "Explain the difference between TCP and UDP networking protocols in bullet points."

        for concurrency in concurrencyLevels {
            let t0 = CFAbsoluteTimeGetCurrent()
            
            await withTaskGroup(of: (Int, Double).self) { group in
                for i in 0..<concurrency {
                    group.addTask {
                        let session = LanguageModelSession(model: model)
                        let options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 256)
                        var lastSnapshot: LanguageModelSession.ResponseStream<String>.Snapshot? = nil
                        do {
                            for try await snap in session.streamResponse(to: prompt, options: options) {
                                lastSnapshot = snap
                            }
                            let outTok = lastSnapshot?.usage.output.totalTokenCount ?? 0
                            return (outTok, 0.0)
                        } catch {
                            return (0, 0.0)
                        }
                    }
                }
                
                var totalTokens = 0
                for await (tokens, _) in group {
                    totalTokens += tokens
                }
                
                let wallTime = CFAbsoluteTimeGetCurrent() - t0
                let aggregateSpeed = wallTime > 0 ? Double(totalTokens) / wallTime : 0.0
                print(String(format: "[Concurrency = %d] Total Tokens: %4d | Wall Time: %.3fs | Aggregate Throughput: %6.2f tok/s",
                             concurrency, totalTokens, wallTime, aggregateSpeed))
            }
        }
    }
}
