package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.service.ModelClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.time.Duration;

@RestController
@RequestMapping("/chat")
public class ChatStreamController {

    private final ModelClient modelClient;

    @Value("${app.sse.heartbeat-seconds:20}")
    private long heartbeatSec;

    public ChatStreamController(ModelClient modelClient) {
        this.modelClient = modelClient;
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> stream(@RequestParam("q") String q) {
        var dataFlux = modelClient.streamDeltas(q)
                .map(delta -> ServerSentEvent.<String>builder(delta)
                        .event("message")
                        .build());

        var heartbeat = Flux.interval(Duration.ofSeconds(heartbeatSec))
                .map(i -> ServerSentEvent.<String>builder("")
                        .comment("ping")
                        .build());

        // 데이터 + 주석형 하트비트 interleave
        return Flux.merge(dataFlux, heartbeat)
                .concatWith(Flux.just(ServerSentEvent.<String>builder("")
                        .event("done")
                        .build()));
    }
}
