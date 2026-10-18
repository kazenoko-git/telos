import Foundation
import FoundationModels

@main
struct AFMHeadToHeadBenchmark {
    static func main() async {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            print("AFM unavailable")
            return
        }

        let prompts: [(String, String, Int)] = [
            ("QuickSort Python", "Write a Python implementation of QuickSort with comments.", 512),
            ("Counting Run", "Count from 1 to 100, one number per line.", 512),
            ("OS Reasoning", "Explain how virtual memory pages and TLB work in modern operating systems.", 512)
        ]

        print("=== Apple AFM-3 On-Device Head-to-Head Benchmark ===")

        for (label, prompt, maxTokens) in prompts {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(temperature: 0.0, maximumResponseTokens: maxTokens)

            var firstTokenTime: CFAbsoluteTime? = nil
            var lastSnapshot: LanguageModelSession.ResponseStream<String>.Snapshot? = nil
            let t0 = CFAbsoluteTimeGetCurrent()

            do {
                for try await snapshot in session.streamResponse(to: prompt, options: options) {
                    if firstTokenTime == nil {
                        firstTokenTime = CFAbsoluteTimeGetCurrent()
                    }
                    lastSnapshot = snapshot
                }
                let t1 = CFAbsoluteTimeGetCurrent()
                let ttft = (firstTokenTime ?? t1) - t0
                let decodeTime = t1 - (firstTokenTime ?? t1)
                let outTokens = lastSnapshot?.usage.output.totalTokenCount ?? 0
                let inTokens = lastSnapshot?.usage.input.totalTokenCount ?? 0
                let decodeSpeed = decodeTime > 0 ? Double(outTokens) / decodeTime : 0.0

                print(String(format: "[%-16s] Tokens: %4d | TTFT: %.3fs | Decode: %.3fs | Speed: %6.2f tok/s",
                             label, outTokens, ttft, decodeTime, decodeSpeed))
                fflush(stdout)
            } catch {
                print("Error in [\(label)]: \(error)")
            }
        }
    }
}
