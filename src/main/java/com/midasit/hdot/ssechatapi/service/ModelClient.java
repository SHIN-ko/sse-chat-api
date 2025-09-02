
package com.midasit.hdot.ssechatapi.service;

import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

@Component
public class ModelClient {

    private final WebClient webClient;

    @Value("${app.model.base-url}")
    private String baseUrl;               // 예: http://localhost:8000/v1

    @Value("${app.model.endpoint:/chat/completions}")
    private String endpoint;              // 예: /chat/completions

    @Value("${app.model.name}")
    private String modelName;             // 예: local-llama

    @Value("${app.model.api-key:}")
    private String apiKey;

    public ModelClient(WebClient webClient) {
        this.webClient = webClient;
    }

    public Flux<String> streamDeltas(String prompt) {
        var messages = new JSONArray()
                .put(new JSONObject().put("role", "user").put("content", prompt));

        var body = new JSONObject()
                .put("model", modelName)
                .put("stream", true)
                .put("messages", messages);

        // llama.cpp OpenAI-compat: /v1/chat/completions (SSE: data: ... , [DONE] 종료)
        return webClient.post()
                .uri(baseUrl + endpoint)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .header(HttpHeaders.ACCEPT, "text/event-stream")   // 스트리밍 힌트
                .headers(h -> {
                    if (apiKey != null && !apiKey.isBlank()) {
                        h.setBearerAuth(apiKey);
                    }
                })
                .bodyValue(body.toString())
                .retrieve()
                .bodyToFlux(String.class)               // 업스트림 SSE 원문 청크
                .flatMap(chunk -> Flux.fromArray(chunk.split("\\r?\\n")))
                .filter(line -> line.startsWith("data: "))
                .map(line -> line.substring(6).trim())  // "data: " 제거
                .takeUntil(payload -> payload.equals("[DONE]"))
                .filter(payload -> !payload.equals("[DONE]"))
                .map(payload -> {
                    try {
                        var obj = new JSONObject(payload);
                        var choices = obj.getJSONArray("choices");
                        if (choices.isEmpty()) return "";
                        var delta = choices.getJSONObject(0).optJSONObject("delta");
                        if (delta == null) return "";
                        return delta.optString("content", "");
                    } catch (Exception e) {
                        return "";
                    }
                })
                .filter(s -> !s.isEmpty());
    }
}