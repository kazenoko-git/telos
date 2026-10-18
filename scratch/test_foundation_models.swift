import Foundation
import FoundationModels

@main
struct App {
    static func main() async {
        let model = SystemLanguageModel.default
        do {
            let session = LanguageModelSession(model: model, instructions: "You are a helpful coding assistant. Write direct code without conversational preamble.")
            let options = GenerationOptions(temperature: 0.0, maximumResponseTokens: 256)
            let response = try await session.respond(to: "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n", options: options)
            print("=== Response Content ===")
            print(response.content)
            print("========================")
        } catch {
            print("Error: \(error)")
        }
    }
}
