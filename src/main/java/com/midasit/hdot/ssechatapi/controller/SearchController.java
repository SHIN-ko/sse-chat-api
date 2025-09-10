package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.service.EmbeddingService;
import com.midasit.hdot.ssechatapi.service.ModelClient;
import com.midasit.hdot.ssechatapi.service.OpenSearchClient;
import com.midasit.hdot.ssechatapi.service.RerankerService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class SearchController {

    private final EmbeddingService embedding;
    private final OpenSearchClient os;
    private final RerankerService reranker;
    private final ModelClient modelClient;

    public SearchController(EmbeddingService embedding,
                            OpenSearchClient os,
                            RerankerService reranker,
                            ModelClient modelClient) {
        this.embedding = embedding;
        this.os = os;
        this.reranker = reranker;
        this.modelClient = modelClient;
    }

    @GetMapping("/search")
    public Mono<String> search(@RequestParam String q, @RequestParam(defaultValue = "5") int k) {
        try {
            float[] vec = embedding.embedOne(q);
            return os.knnSearch(vec, k);
        } catch (Exception e) {
            return Mono.error(e);
        }
    }

    @GetMapping("/search2")
    public Mono<Map<String, Object>> search(
            @RequestParam String q,
            @RequestParam(defaultValue = "10") int k,
            @RequestParam(defaultValue = "true") boolean rerank
    ) {
        float[] qVec = embedding.embedOne(q); // 기존 EmbeddingService

        return os.knnSearchParsed(qVec, k) // OsResult { hits, tookMs }
                .flatMap(res -> {
                    if (!rerank) {
                        return Mono.just(Map.of("took", res.tookMs(), "hits", res.hits()));
                    }

                    return Flux.fromIterable(res.hits())
                            .map(hit -> {
                                float s = reranker.score(q, hit.text()); // 👈 ONNX 리랭커
                                return Map.of(
                                        "_id", hit.id(),
                                        "osScore", hit.score(),
                                        "rerankScore", s,
                                        "title", hit.title(),
                                        "text", hit.text(),
                                        "metadata", hit.metadata()
                                );
                            })
                            .sort((a, b) -> Float.compare(
                                    ((Number) b.get("rerankScore")).floatValue(),
                                    ((Number) a.get("rerankScore")).floatValue()
                            ))
                            .collectList()
                            .map(hits -> Map.of("took", res.tookMs(), "hits", hits));
                });
    }

    @GetMapping(value = "/ask", produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<ResponseEntity<Map<String, Object>>> ask(
            @RequestParam String q,
            @RequestParam(defaultValue = "8") int k,
            @RequestParam(defaultValue = "true") boolean rerank
    ) {
        // 입력 검증(바로 400 JSON)
        if (q == null || q.isBlank()) {
            return Mono.just(ResponseEntity.badRequest().body(Map.of(
                    "error", "bad_request",
                    "message", "q must not be blank"
            )));
        }
        if (k < 1 || k > 50) {
            return Mono.just(ResponseEntity.badRequest().body(Map.of(
                    "error", "bad_request",
                    "message", "k must be between 1 and 50"
            )));
        }

        float[] qVec;
        try {
            qVec = embedding.embedOne(q);
        } catch (Exception e) {
            return Mono.just(ResponseEntity.status(500).body(Map.of(
                    "error", "embedding_failed",
                    "message", e.getMessage()
            )));
        }

        return os.knnSearchParsed(qVec, k)
                .flatMap(res -> {
                    List<Map<String, Object>> baseHits = res.hits().stream()
                            .map(h -> Map.of(
                                    "_id", h.id(),
                                    "osScore", h.score(),
                                    "title", h.title(),
                                    "text", h.text(),
                                    "metadata", h.metadata()
                            ))
                            .toList();

                    Mono<List<Map<String, Object>>> hitsMono = !rerank
                            ? Mono.just(baseHits)
                            : Flux.fromIterable(baseHits)
                            .map(hit -> {
                                Map<String, Object> copy = new HashMap<>(hit);
                                float s = reranker.score(q, String.valueOf(hit.get("text")));
                                copy.put("rerankScore", s);
                                return copy; // 타입을 Map으로 유지
                            })
                            .sort((a, b) -> Float.compare(
                                    ((Number)b.getOrDefault("rerankScore", 0f)).floatValue(),
                                    ((Number)a.getOrDefault("rerankScore", 0f)).floatValue()
                            ))
                            .collectList();

                    return hitsMono.flatMap(hits -> {
                        List<String> contexts = hits.stream()
                                .map(h -> {
                                    String title = String.valueOf(h.getOrDefault("title",""));
                                    String text  = String.valueOf(h.getOrDefault("text",""));
                                    return title.isBlank() ? text : ("Title: " + title + "\n" + text);
                                })
                                .toList();

                        String system = "Use ONLY provided context. Answer in Korean. Say you don't know if unsure.";
                        return modelClient.completeWithContext(system, q, contexts, 4000)
                                .map(answer -> ResponseEntity.ok(Map.of(
                                        "question", q,
                                        "tookMs", res.tookMs(),
                                        "answer", answer,
                                        "sources", hits
                                )));
                    });
                })
                .onErrorResume(e -> {
                    // 여기로 떨어지면 내부 예외(리랭커/LLM/OS 등)
                    return Mono.just(ResponseEntity.status(500).body(Map.of(
                            "error", "internal_error",
                            "message", e.getMessage()
                    )));
                });
    }
}