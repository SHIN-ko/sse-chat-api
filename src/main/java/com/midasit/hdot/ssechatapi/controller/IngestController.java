package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.dto.DocIngestRq;
import com.midasit.hdot.ssechatapi.repository.RawDocRepository;
import com.midasit.hdot.ssechatapi.service.EmbeddingService;
import com.midasit.hdot.ssechatapi.service.OpenSearchClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api")
public class IngestController {

    private final EmbeddingService embedding;
    private final RawDocRepository rawRepo;
    private final OpenSearchClient os;

    public IngestController(EmbeddingService embedding, RawDocRepository rawRepo, OpenSearchClient os) {
        this.embedding = embedding;
        this.rawRepo = rawRepo;
        this.os = os;
    }

    @PostMapping("/docs")
    public Mono<String> ingest(@RequestBody DocIngestRq rq) {
        try {
            float[] vec = embedding.embedOne(rq.text());
            rawRepo.save(rq);
            return os.indexDoc(rq.docId(), rq.title(), rq.text(), vec, rq.metadata())
                    .thenReturn("OK");
        } catch (Exception e) {
            return Mono.error(e);
        }
    }
}