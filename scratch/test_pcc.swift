import Foundation
import FoundationModels

@main
struct App {
    static func main() async {
        let model = PrivateCloudComputeLanguageModel()
        print("Availability: \(model.availability)")
        
        let session = LanguageModelSession(model: model, instructions: "You are a helpful assistant. Reply directly.")
        let options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 64)
        
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            print("Sending request to PrivateCloudComputeLanguageModel...")
            let resp = try await session.respond(to: "Say hello in exactly 5 words.", options: options)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            print("Response received in \(String(format: "%.3f", elapsed))s:")
            print(resp.content)
            
            // Calculate tokens / sec approx
            let words = resp.content.split(separator: " ").count
            let approxTokens = Double(words) * 1.33
            let tokPerSec = approxTokens / elapsed
            print("Approx Tok/s: \(String(format: "%.2f", tokPerSec))")
        } catch {
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            print("Error after \(String(format: "%.3f", elapsed))s: \(error)")
        }
    }
}
