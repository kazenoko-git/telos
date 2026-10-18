
import Foundation
import FoundationModels

let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)

Task {
    for try await chunk in session.streamResponse(to: "Hello") {
        print(chunk)
    }
}
