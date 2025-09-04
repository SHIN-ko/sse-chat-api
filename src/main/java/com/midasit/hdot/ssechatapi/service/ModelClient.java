
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

    public ModelClient(WebClient webClient) {
        this.webClient = webClient;
    }

    private String url() {
        // 최종 호출 URL = base-url + endpoint
        return (baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length()-1) : baseUrl) + endpoint;
    }

    public Flux<String> streamDeltas(String system, String userPrompt) {
        var messages = new JSONArray();
        if (system != null && !system.isBlank()) {
            messages.put(new JSONObject().put("role", "system").put("content", system));
        }
        messages.put(new JSONObject().put("role", "user").put("content", userPrompt));

        var body = new JSONObject()
                .put("model", modelName)
                .put("stream", true)
                .put("temperature", temperature)
                .put("messages", messages);

        return webClient.post()
                .uri(url())
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .headers(h -> { if (!apiKey.isBlank()) h.setBearerAuth(apiKey); })
                .bodyValue(body.toString())
                .retrieve()
                .bodyToFlux(String.class)                 // OpenAI 호환 SSE 원문을 라인 단위로 파싱
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

    public Mono<String> complete(String system, String userPrompt) {
        // 스트림을 모두 모아 최종 문자열로
        return streamDeltas(system, userPrompt)
                .reduce(new StringBuilder(), StringBuilder::append)
                .map(StringBuilder::toString);
    }
}