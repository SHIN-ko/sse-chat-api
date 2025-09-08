package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.service.EmbeddingService;
import com.midasit.hdot.ssechatapi.service.OpenSearchClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api")
public class SearchController {

    private final EmbeddingService embedding;
    private final OpenSearchClient os;

    public SearchController(EmbeddingService embedding, OpenSearchClient os) {
        this.embedding = embedding;
        this.os = os;
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
}