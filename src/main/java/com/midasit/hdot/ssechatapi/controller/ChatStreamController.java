package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.dto.ChatRequest;
import com.midasit.hdot.ssechatapi.dto.ChatResponse;
import com.midasit.hdot.ssechatapi.service.ModelClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;

@RestController
@RequestMapping("/api/chat")
public class ChatStreamController {

    private final ModelClient modelClient;

    @Value("${app.sse.heartbeat-seconds:20}")
    private long heartbeatSec;

    public ChatStreamController(ModelClient modelClient) {
        this.modelClient = modelClient;
    }

    // 1) 비스트리밍: 최종 텍스트 한 번에 반환
    @PostMapping(produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<ChatResponse> chat(@RequestBody ChatRequest req) {
        return modelClient.complete(req.system(), req.userPrompt())
                .map(ChatResponse::new);
    }

    // 2-A) 스트리밍(SSE, GET) — EventSource 사용 시 (POST 불가하므로 쿼리파라미터)
    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> streamGet(
            @RequestParam(required = false, defaultValue = "") String system,
            @RequestParam("userPrompt") String userPrompt
    ) {
        return sse(system, userPrompt);
    }

    // 2-B) 스트리밍(SSE, POST) — fetch(POST) 스트림 사용 시
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> streamPost(@RequestBody ChatRequest req) {
        return sse(req.system(), req.userPrompt());
    }

    /** 공통 SSE 구현 */
    private Flux<ServerSentEvent<String>> sse(String system, String userPrompt) {
        var dataFlux = modelClient.streamDeltas(system, userPrompt)
                .map(delta -> ServerSentEvent.<String>builder(delta)
                        .event("message")
                        .build());

        var heartbeat = Flux.interval(Duration.ofSeconds(heartbeatSec))
                .map(i -> ServerSentEvent.<String>builder("")
                        .comment("ping")
                        .build());

        // heartbeat는 dataFlux가 끝나면 자동 종료되도록 제어하고, 마지막에 done 이벤트 전송
        return Flux.merge(
                        dataFlux,
                        heartbeat.takeUntilOther(dataFlux.ignoreElements().then())
                )
                .concatWith(Mono.just(ServerSentEvent.<String>builder("")
                        .event("done")
                        .build()));
    }
}