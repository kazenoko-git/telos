import Foundation
import FoundationModels

@main
struct AFMLongBenchmark {
    static func main() async {
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return }

        let prompts = [
            "Write an extremely detailed, 1000-word science fiction story about an astronomer on Mars who discovers a hidden ancient computational archive beneath the ice. Describe the architecture, the optical storage crystals, and the historical records.",
            "Write the complete text of a comprehensive, 800-word tutorial on low-level memory management in C, covering stack vs heap, virtual memory pages, mmap, brk/sbrk, and custom free-list allocators with full explanations.",
            "Generate an exhaustive numbered list from 1 to 100 of the most significant engineering breakthroughs in human history, explaining the physics principle behind each."
        ]

        for (i, p) in prompts.enumerated() {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                temperature: 0.2,
                maximumResponseTokens: 2048
            )

            var firstTokenTime: CFAbsoluteTime? = nil
            var lastSnapshot: LanguageModelSession.ResponseStream<String>.Snapshot? = nil
            let t0 = CFAbsoluteTimeGetCurrent()

            do {
                for try await snapshot in session.streamResponse(to: p, options: options) {
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

                print("Test \(i+1): In: \(inTokens) | Out: \(outTokens) | TTFT: \(String(format: "%.3f", ttft))s | Decode: \(String(format: "%.3f", decodeTime))s | Decode Speed: \(String(format: "%.2f", decodeSpeed)) tok/s")
                fflush(stdout)
            } catch {
                print("Error in Test \(i+1): \(error)")
            }
        }
    }
}
