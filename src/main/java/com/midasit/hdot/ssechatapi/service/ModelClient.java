
package com.midasit.hdot.ssechatapi.service;

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

    @Value("${app.model.url}")
    private String modelUrl;

    @Value("${app.model.name}")
    private String modelName;

    @Value("${app.model.api-key:}")
    private String apiKey;

    public ModelClient(WebClient webClient) {
        this.webClient = webClient;
    }

    public Flux<String> streamDeltas(String prompt) {
        // OpenAI 호환 요청 바디
        var body = new JSONObject()
                .put("model", modelName)
                .put("stream", true)
                .put("messages", new org.json.JSONArray()
                        .put(new JSONObject().put("role", "user").put("content", prompt)));

        return webClient.post()
                .uri(modelUrl)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .headers(h -> { if (!apiKey.isBlank()) h.setBearerAuth(apiKey); })
                .bodyValue(body.toString())
                .retrieve()
                .bodyToFlux(String.class)      // 업스트림 SSE 원문 청크
                .flatMap(chunk -> Flux.fromArray(chunk.split("\\r?\\n")))
                .filter(line -> line.startsWith("data: "))
                .map(line -> line.substring(6).trim())
                .takeUntil(payload -> payload.equals("[DONE]"))
                .filter(payload -> !payload.equals("[DONE]"))
                .map(payload -> {
                    try {
                        var obj = new JSONObject(payload);
                        return obj.getJSONArray("choices")
                                .getJSONObject(0)
                                .getJSONObject("delta")
                                .optString("content", "");
                    } catch (Exception e) {
                        return "";
                    }
                })
                .filter(s -> !s.isEmpty());
    }
}
