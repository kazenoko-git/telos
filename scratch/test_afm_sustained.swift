import Foundation
import FoundationModels

@main
struct AFMSustainedBenchmark {
    static func main() async {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            print("SystemLanguageModel is not available!")
            return
        }

        print("=== Evaluating AFM Peak & Sustained Decode Throughput ===")
        
        let prompts = [
            ("Warmup 1", "Count from 1 to 50, one number per line.", 200),
            ("Warmup 2", "List 30 common algorithms in computer science with a one-sentence description each.", 500),
            ("Code Long", "Write a complete C++20 implementation of a B-tree including node splitting, insertion, and traversal methods with zero external dependencies.", 1000),
            ("Reasoning Long", "Explain the full lifecycle of a thread in an operating system kernel from creation, scheduling across runqueues, context switching via register saves, handling page faults, to termination and zombie reclamation.", 1000),
            ("Python Long", "Write a Python script that implements an event-driven async reactor pattern from scratch using standard socket and selectors modules, with full error handling.", 1000)
        ]

        var decodeSpeeds: [Double] = []
        
        for (label, prompt, maxTokens) in prompts {
            let session = LanguageModelSession(model: model)
            let options = GenerationOptions(
                temperature: 0.0,
                maximumResponseTokens: maxTokens
            )
            
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
                decodeSpeeds.append(decodeSpeed)
                
                print("[\(label)] In: \(inTokens) | Out: \(outTokens) | TTFT: \(String(format: "%.3f", ttft))s | Decode: \(String(format: "%.3f", decodeTime))s | Speed: \(String(format: "%.2f", decodeSpeed)) tok/s")
                fflush(stdout)
            } catch {
                print("Error in [\(label)]: \(error)")
                fflush(stdout)
            }
        }
        
        if !decodeSpeeds.isEmpty {
            let avgSpeed = decodeSpeeds.reduce(0.0, +) / Double(decodeSpeeds.count)
            let maxSpeed = decodeSpeeds.max() ?? 0.0
            print(String(format: "\nAverage Decode Speed: %.2f tok/s | Peak Decode Speed: %.2f tok/s", avgSpeed, maxSpeed))
        }
    }
}
