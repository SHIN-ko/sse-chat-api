package com.midasit.hdot.ssechatapi.service;

import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Component
public class ModelClient {

    private final WebClient webClient;

    @Value("${app.model.base-url}")
    private String baseUrl;

    @Value("${app.model.endpoint:/chat/completions}")
    private String endpoint;

    @Value("${app.model.name}")
    private String modelName;

    @Value("${app.model.api-key:}")
    private String apiKey;

    @Value("${app.model.temperature:0.2}")
    private double temperature;

    @Value("${app.model.max-tokens:64}")
    private int maxTokens;

    public ModelClient(WebClient webClient) {
        this.webClient = webClient;
    }

    private String url() {
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length()-1) : baseUrl;
        return base + endpoint;
    }

    private boolean isChat()        { return endpoint.contains("/chat/"); }
    private boolean isV1Compl()     { return endpoint.contains("/v1/completions") || endpoint.endsWith("/completions"); }
    private boolean isLegacyCompl() { return endpoint.endsWith("/completion"); } // llama.cpp 고유

    /** 공통 요청 바디 생성 (stream:true로 요청; 업스트림이 무시하고 JSON 한 번에 줘도 처리함) */
    private JSONObject buildBody(String system, String userPrompt, boolean stream) {
        JSONObject body = new JSONObject().put("temperature", temperature).put("stream", stream);
        if (isChat()) {
            JSONArray messages = new JSONArray();
            if (system != null && !system.isBlank()) {
                messages.put(new JSONObject().put("role","system").put("content", system));
            }
            messages.put(new JSONObject().put("role","user").put("content", userPrompt));
            body.put("model", modelName).put("messages", messages).put("max_tokens", maxTokens);
        } else if (isV1Compl()) {
            String prompt = (system == null || system.isBlank())
                    ? userPrompt
                    : "[SYSTEM]\n" + system + "\n\n[USER]\n" + userPrompt + "\n\n[ASSISTANT]\n";
            body.put("model", modelName).put("prompt", prompt).put("max_tokens", maxTokens);
        } else if (isLegacyCompl()) {
            String prompt = (system == null || system.isBlank())
                    ? userPrompt
                    : "[SYSTEM]\n" + system + "\n\n[USER]\n" + userPrompt + "\n\n[ASSISTANT]\n";
            body.remove("stream"); // /completion은 stream 키 없어도 됨(있어도 무시)
            body.put("prompt", prompt).put("n_predict", Math.max(8, Math.min(maxTokens, 256)));
        } else {
            // 기본: /v1/completions 취급
            String prompt = (system == null || system.isBlank())
                    ? userPrompt
                    : "[SYSTEM]\n" + system + "\n\n[USER]\n" + userPrompt + "\n\n[ASSISTANT]\n";
            body.put("model", modelName).put("prompt", prompt).put("max_tokens", maxTokens);
        }
        return body;
    }

    /** 스트리밍: 가능하면 SSE로, 아니면 JSON 한방 응답을 파싱해서 Flux로 변환 */
    public Flux<String> streamDeltas(String system, String userPrompt) {
        JSONObject body = buildBody(system, userPrompt, true);

        return webClient.post()
                .uri(url())
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .headers(h -> { if (!apiKey.isBlank()) h.setBearerAuth(apiKey); })
                .bodyValue(body.toString())
                .exchangeToFlux(resp -> {
                    MediaType ct = resp.headers().contentType().orElse(MediaType.APPLICATION_JSON);
                    // 1) SSE (text/event-stream)
                    if (MediaType.TEXT_EVENT_STREAM.isCompatibleWith(ct)) {
                        return resp.bodyToFlux(String.class)
                                .flatMap(chunk -> Flux.fromArray(chunk.split("\\r?\\n")))
                                .filter(line -> line.startsWith("data: "))
                                .map(line -> line.substring(6).trim())
                                .takeUntil("[DONE]"::equals)
                                .filter(payload -> !"[DONE]".equals(payload))
                                .map(this::extractTextSafely)
                                .filter(s -> s != null && !s.isBlank());
                    }
                    // 2) JSON 한 번에
                    return resp.bodyToMono(String.class)
                            .flatMapMany(json -> {
                                String text = extractTextFromNonStream(json);
                                return text == null || text.isBlank()
                                        ? Flux.empty()
                                        : Flux.just(text);
                            });
                });
    }

    /** 비-스트리밍: 필요 시 JSON 한방 호출 */
    public Mono<String> complete(String system, String userPrompt) {
        // 스트림 시도 → 없으면 논스트림 재시도
        return streamDeltas(system, userPrompt)
                .reduce(new StringBuilder(), StringBuilder::append)
                .map(StringBuilder::toString)
                .flatMap(s -> {
                    if (s != null && !s.isBlank()) return Mono.just(s);
                    // 스트림에서 아무 것도 못 뽑았으면 논스트림으로 재호출
                    JSONObject nonStreamBody = buildBody(system, userPrompt, false);
                    return webClient.post()
                            .uri(url())
                            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                            .headers(h -> { if (!apiKey.isBlank()) h.setBearerAuth(apiKey); })
                            .bodyValue(nonStreamBody.toString())
                            .retrieve()
                            .bodyToMono(String.class)
                            .map(this::extractTextFromNonStream)
                            .defaultIfEmpty("");
                });
    }

    /** 스트림 조각에서 안전 추출 */
    private String extractTextSafely(String payload) {
        try {
            var obj = new JSONObject(payload);

            // chat stream: choices[0].delta.content / reasoning_content
            if (obj.has("choices")) {
                var ch0 = obj.getJSONArray("choices").optJSONObject(0);
                if (ch0 != null) {
                    if (ch0.has("delta")) {
                        var delta = ch0.optJSONObject("delta");
                        if (delta != null) {
                            String c = delta.optString("content", "");
                            if (!c.isBlank()) return c;
                            String rc = delta.optString("reasoning_content", "");
                            if (!rc.isBlank()) return rc;
                        }
                    }
                    // v1/completions stream: choices[0].text
                    String t = ch0.optString("text", "");
                    if (!t.isBlank()) return t;
                }
            }
            // /completion stream: top-level content
            String top = obj.optString("content", "");
            if (!top.isBlank()) return top;

            // 일부 구현체: { "delta":"..." }
            String plainDelta = obj.optString("delta", "");
            if (!plainDelta.isBlank()) return plainDelta;

            return "";
        } catch (Exception ignore) {
            return "";
        }
    }

    /** 논스트림 JSON에서 안전 추출 */
    private String extractTextFromNonStream(String json) {
        try {
            var obj = new JSONObject(json);

            // /v1/chat/completions (non-stream): choices[0].message.content
            if (obj.has("choices")) {
                var ch0 = obj.getJSONArray("choices").optJSONObject(0);
                if (ch0 != null) {
                    if (ch0.has("message")) {
                        var msg = ch0.optJSONObject("message");
                        if (msg != null) {
                            String c = msg.optString("content", "");
                            if (!c.isBlank()) return c;
                        }
                    }
                    // /v1/completions (non-stream): choices[0].text
                    String t = ch0.optString("text", "");
                    if (!t.isBlank()) return t;
                }
            }
            // /completion (non-stream): top-level content
            String top = obj.optString("content", "");
            if (!top.isBlank()) return top;

            return "";
        } catch (Exception ignore) {
            return "";
        }
    }
}
