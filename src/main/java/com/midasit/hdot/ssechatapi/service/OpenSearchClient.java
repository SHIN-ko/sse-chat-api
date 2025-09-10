package com.midasit.hdot.ssechatapi.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.midasit.hdot.ssechatapi.dto.OsHit;
import com.midasit.hdot.ssechatapi.dto.OsResult;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class OpenSearchClient {

    private final WebClient wc;
    private final String index;
    private final ObjectMapper om = new ObjectMapper();

    public OpenSearchClient(
            @Value("${app.opensearch.base-url}") String baseUrl,
            @Value("${app.opensearch.index}") String index,
            @Value("${app.opensearch.username}") String user,
            @Value("${app.opensearch.password}") String pass
    ) {
        this.index = index;
        HttpClient http = HttpClient.create().secure(ssl -> {
            try {
                ssl.sslContext(
                        SslContextBuilder.forClient()
                                .trustManager(InsecureTrustManagerFactory.INSTANCE)
                                .build()
                );
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
        this.wc = WebClient.builder()
                .baseUrl(baseUrl)
                .defaultHeaders(h -> h.setBasicAuth(user, pass))
                .clientConnector(new ReactorClientHttpConnector(http))
                .build();
    }

    public Mono<Void> indexDoc(String id, String title, String text, float[] embedding, Map<String,Object> metadata) {
        var body = Map.of(
                "doc_id", id,
                "title", title,
                "text", text,
                "metadata", metadata == null ? Map.of() : metadata,
                "embedding", embedding
        );
        return wc.put()
                .uri("/" + index + "/_doc/{id}", id)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(String.class)
                .then();
    }

    public Mono<String> knnSearch(float[] queryVec, int k) {
        var body = Map.of(
                "size", k,
                "query", Map.of(
                        "knn", Map.of(
                                "embedding", Map.of(
                                        "vector", queryVec,
                                        "k", k
                                )
                        )
                ),
                "_source", new String[]{"doc_id","title","text","metadata"}
        );
        return wc.post()
                .uri("/" + index + "/_search")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(String.class);
    }

    public Mono<OsResult> knnSearchParsed(float[] queryVec, int k) {
        var body = Map.of(
                "size", k,
                "query", Map.of(
                        "knn", Map.of(
                                "embedding", Map.of("vector", queryVec, "k", k)
                        )
                ),
                "_source", new String[]{"doc_id","title","text","metadata"}
        );

        return wc.post()
                .uri("/" + index + "/_search")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(String.class)
                .map(this::toOsResult);
    }

    private OsResult toOsResult(String json) {
        try {
            JsonNode root = om.readTree(json);
            int took = root.path("took").asInt(0);

            List<OsHit> hits = new ArrayList<>();
            for (JsonNode h : root.path("hits").path("hits")) {
                String id = h.path("_id").asText();
                float score = (float) h.path("_score").asDouble(0.0);

                JsonNode src = h.path("_source");
                String title = src.path("title").asText(null);
                String text  = src.path("text").asText(null);

                Map<String, Object> metadata = om.convertValue(
                        src.path("metadata"),
                        new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {}
                );

                hits.add(new OsHit(id, score, title, text, metadata));
            }
            return new OsResult(hits, took);
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse OpenSearch response", e);
        }
    }

}