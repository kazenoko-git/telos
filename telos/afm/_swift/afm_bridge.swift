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

/// One-shot availability report, emitted for `afm_bridge --status`.
///
/// `FoundationModels` exposes no public selector for a specific on-device
/// model, only `SystemLanguageModel.default`. The `check` field therefore
/// reports *why* the default model is or is not usable, and the Python side
/// maps that onto the Core / Core Advanced preference.
struct StatusResponse: Codable {
    let available: Bool
    /// One of: "available", "device_not_eligible",
    /// "apple_intelligence_not_enabled", "model_not_ready", "unknown".
    let check: String
    let detail: String?
}

func describeAvailability(_ model: SystemLanguageModel) -> StatusResponse {
    switch model.availability {
    case .available:
        return StatusResponse(available: true, check: "available", detail: nil)
    case .unavailable(let reason):
        switch reason {
        case .deviceNotEligible:
            return StatusResponse(
                available: false,
                check: "device_not_eligible",
                detail: "This device is not eligible for Apple Intelligence."
            )
        case .appleIntelligenceNotEnabled:
            return StatusResponse(
                available: false,
                check: "apple_intelligence_not_enabled",
                detail: "Apple Intelligence is not enabled in System Settings."
            )
        case .modelNotReady:
            return StatusResponse(
                available: false,
                check: "model_not_ready",
                detail: "The on-device model is still downloading or preparing."
            )
        @unknown default:
            return StatusResponse(available: false, check: "unknown", detail: "\(reason)")
        }
    @unknown default:
        return StatusResponse(available: false, check: "unknown", detail: nil)
    }
}

func emitJSON<T: Encodable>(_ value: T) {
    if let data = try? JSONEncoder().encode(value),
       let str = String(data: data, encoding: .utf8) {
        print(str)
        fflush(stdout)
    }
}

@main
struct AFMBridge {
    static func main() async {
        let model = SystemLanguageModel.default

        // One-shot status probe: `afm_bridge --status`. Exits immediately so
        // the Python side can report availability without opening a session.
        if CommandLine.arguments.contains("--status") {
            emitJSON(describeAvailability(model))
            return
        }

        guard model.isAvailable else {
            let resp = GenerationResponse(id: nil, content: "", error: "SystemLanguageModel is not available on this system.")
            emitJSON(resp)
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
                emitJSON(GenerationResponse(id: req.id, content: text, error: nil))
            } catch {
                emitJSON(GenerationResponse(id: req.id, content: "", error: "\(error)"))
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
