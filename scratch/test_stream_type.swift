
import Foundation
import FoundationModels

@main
struct StreamTest {
    static func main() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(model: model)
        for try await chunk in session.streamResponse(to: "Hi") {
            print("Chunk type:", type(of: chunk))
            print("Chunk:", chunk)
            break
        }
    }
}
