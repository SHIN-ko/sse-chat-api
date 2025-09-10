package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.service.EmbeddingService;
import com.midasit.hdot.ssechatapi.service.OpenSearchClient;
import com.midasit.hdot.ssechatapi.service.RerankerService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class SearchController {

    private final EmbeddingService embedding;
    private final OpenSearchClient os;
    private final RerankerService reranker;

    public SearchController(EmbeddingService embedding, OpenSearchClient os, RerankerService reranker) {
        this.embedding = embedding;
        this.os = os;
        this.reranker = reranker;
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
}