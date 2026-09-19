import Foundation
import FoundationModels

struct GenerationRequest: Codable {
    let id: Int?
    let prompt: String
    let instructions: String?
    let maxTokens: Int?
    let temperature: Double?
    let stopTokens: [String]?
}

struct GenerationResponse: Codable {
    let id: Int?
    let content: String
    let error: String?
}

@main
struct AFMBridge {
    static func main() async {
        let model = SystemLanguageModel.default
        guard model.isAvailable else {
            let resp = GenerationResponse(id: nil, content: "", error: "SystemLanguageModel is not available on this system.")
            if let data = try? JSONEncoder().encode(resp) {
                print(String(data: data, encoding: .utf8)!)
            }
            return
        }

        let stdin = FileHandle.standardInput
        while let lineData = try? stdin.readLine() {
            guard !lineData.isEmpty else { continue }
            guard let req = try? JSONDecoder().decode(GenerationRequest.self, from: lineData) else {
                continue
            }

            let instructions = req.instructions ?? "You are an expert programming and reasoning assistant. Provide direct, concise, and accurate output without conversational filler."
            let session = LanguageModelSession(model: model, instructions: instructions)
            let options = GenerationOptions(
                temperature: req.temperature ?? 0.0,
                maximumResponseTokens: req.maxTokens ?? 256
            )

            do {
                let response = try await session.respond(to: req.prompt, options: options)
                var text = response.content
                if let stopTokens = req.stopTokens {
                    for stop in stopTokens {
                        if let range = text.range(of: stop) {
                            text = String(text[..<range.lowerBound])
                        }
                    }
                }
                let resp = GenerationResponse(id: req.id, content: text, error: nil)
                if let outData = try? JSONEncoder().encode(resp),
                   let outStr = String(data: outData, encoding: .utf8) {
                    print(outStr)
                    fflush(stdout)
                }
            } catch {
                let resp = GenerationResponse(id: req.id, content: "", error: "\(error)")
                if let outData = try? JSONEncoder().encode(resp),
                   let outStr = String(data: outData, encoding: .utf8) {
                    print(outStr)
                    fflush(stdout)
                }
            }
        }
    }
}

extension FileHandle {
    func readLine() throws -> Data? {
        var buffer = Data()
        while true {
            guard let byteData = try self.read(upToCount: 1), !byteData.isEmpty else {
                return buffer.isEmpty ? nil : buffer
            }
            if byteData[0] == 10 { // newline \n
                return buffer
            }
            buffer.append(byteData)
        }
    }
}
