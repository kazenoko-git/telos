
import Foundation
import FoundationModels

let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)
print(type(of: session))
