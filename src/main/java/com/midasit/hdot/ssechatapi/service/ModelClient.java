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

import java.util.List;

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

    @Value("${app.model.max-tokens:512}") // ← RAG에선 64는 너무 짧아서 기본 512 권장
    private int maxTokens;

    public ModelClient(WebClient webClient) {
        this.webClient = webClient;
    }

    /* -------------------------
       URL/엔드포인트 정규화
       ------------------------- */
    private String url() {
        String base = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        String ep   = endpoint.startsWith("/") ? endpoint : ("/" + endpoint);
        return base + ep;
    }

    private boolean isChat()        { return endpoint.contains("/chat/"); }
    private boolean isV1Compl()     { return endpoint.contains("/v1/completions") || endpoint.endsWith("/completions"); }
    private boolean isLegacyCompl() { return endpoint.endsWith("/completion"); } // llama.cpp 고유

    /* -------------------------
       공통 요청 바디 생성
       ------------------------- */
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

    /* -------------------------
       스트리밍 호출
       ------------------------- */
    public Flux<String> streamDeltas(String system, String userPrompt) {
        JSONObject body = buildBody(system, userPrompt, true);

        return webClient.post()
                .uri(url())
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .headers(h -> { if (!apiKey.isBlank()) h.setBearerAuth(apiKey); })
                .bodyValue(body.toString())
                .exchangeToFlux(resp -> {
                    MediaType ct = resp.headers().contentType().orElse(MediaType.APPLICATION_JSON);
                    // 1) SSE
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
                                return text.isBlank() ? Flux.empty() : Flux.just(text);
                            });
                });
    }

    /* -------------------------
       논-스트리밍 호출
       ------------------------- */
    public Mono<String> complete(String system, String userPrompt) {
        return streamDeltas(system, userPrompt)
                .reduce(new StringBuilder(), StringBuilder::append)
                .map(StringBuilder::toString)
                .flatMap(s -> {
                    if (s != null && !s.isBlank()) return Mono.just(s);
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

    /* -------------------------
       RAG 전용: 검색 컨텍스트 포함 완성 호출
       contexts: 검색 결과 text들의 리스트(이미 정렬된 top-k)
       maxContextChars: 컨텍스트 총 글자수 제한(토큰 가드)
       ------------------------- */
    public Mono<String> completeWithContext(String system, String question, List<String> contexts, int maxContextChars) {
        String prompt = buildRagPrompt(question, contexts, maxContextChars);
        return complete(system, prompt);
    }

    /* -------------------------
       RAG 프롬프트 빌더
       ------------------------- */
    private String buildRagPrompt(String question, List<String> contexts, int maxContextChars) {
        StringBuilder ctx = new StringBuilder();
        int remain = Math.max(0, maxContextChars);
        int i = 1;
        for (String c : contexts) {
            String chunk = "### Document " + (i++) + "\n" + c + "\n\n";
            if (chunk.length() > remain) break;
            ctx.append(chunk);
            remain -= chunk.length();
        }
        // 시스템 메시지/사용자 메시지로 들어갈 userPrompt를 구성
        return """
                아래는 질의에 답하기 위한 참고 문서들입니다. 문서 내용만 근거로 삼아 답변하세요.
                모르면 모른다고 말하세요. 답변은 한국어로 간결히, 필요 시 코드/명령 예시를 포함하세요.

                [CONTEXT]
                %s
                [QUESTION]
                %s
                """.formatted(ctx.toString(), question);
    }

    /* -------------------------
       스트림 파서 보강
       ------------------------- */
    private String extractTextSafely(String payload) {
        try {
            var obj = new JSONObject(payload);

            // OpenAI chat stream: choices[0].delta.content / reasoning_content
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
                    // 일부 구현체: choices[0].message.content
                    var msg = ch0.optJSONObject("message");
                    if (msg != null) {
                        String c = msg.optString("content", "");
                        if (!c.isBlank()) return c;
                        String rc = msg.optString("reasoning_content", "");
                        if (!rc.isBlank()) return rc;
                    }
                }
            }
            // /completion stream 또는 커스텀
            String top = obj.optString("content", "");
            if (!top.isBlank()) return top;

            // 일부 서버: { "response": "..." }
            String resp = obj.optString("response", "");
            if (!resp.isBlank()) return resp;

            // 일부 구현체: { "delta":"..." }
            String plainDelta = obj.optString("delta", "");
            if (!plainDelta.isBlank()) return plainDelta;

            return "";
        } catch (Exception ignore) {
            return "";
        }
    }

    /* -------------------------
       논-스트림 파서 보강
       ------------------------- */
    private String extractTextFromNonStream(String json) {
        try {
            var obj = new JSONObject(json);

            // /v1/chat/completions
            if (obj.has("choices")) {
                var ch0 = obj.getJSONArray("choices").optJSONObject(0);
                if (ch0 != null) {
                    var msg = ch0.optJSONObject("message");
                    if (msg != null) {
                        String c = msg.optString("content", "");
                        if (!c.isBlank()) return c;
                        String rc = msg.optString("reasoning_content", "");
                        if (!rc.isBlank()) return rc;
                    }
                    // /v1/completions
                    String t = ch0.optString("text", "");
                    if (!t.isBlank()) return t;
                }
            }
            // /completion or custom
            String top = obj.optString("content", "");
            if (!top.isBlank()) return top;

            // 일부 서버: { "response": "..." }
            String resp = obj.optString("response", "");
            if (!resp.isBlank()) return resp;

            return "";
        } catch (Exception ignore) {
            return "";
        }
    }
}
