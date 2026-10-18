
import Foundation
import FoundationModels

@main
struct StreamTest {
    static func main() async throws {
        let model = SystemLanguageModel.default
        let session = LanguageModelSession(model: model)
        var chunkCount = 0
        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime? = nil
        var fullText = ""
        
        for try await chunk in session.streamResponse(to: "Count from 1 to 30. One number per line.") {
            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
            chunkCount += 1
            fullText += chunk
        }
        let end = CFAbsoluteTimeGetCurrent()
        let ttft = (firstTokenTime ?? end) - start
        let decodeTime = end - (firstTokenTime ?? end)
        print("Chunks:", chunkCount)
        print("Full length (chars):", fullText.count)
        print("TTFT (s):", ttft)
        print("Decode time (s):", decodeTime)
        print("Total time (s):", end - start)
        print("Chunks/sec (decode):", Double(chunkCount) / max(decodeTime, 0.0001))
    }
}
